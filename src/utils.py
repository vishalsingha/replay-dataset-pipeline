"""
Shared utilities for the replay dataset pipeline.

Provides checkpointing helpers, JSONL I/O, deduplication, and common functions
used across multiple pipeline scripts.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import tempfile
import unicodedata
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

_log = logging.getLogger("replay")


# ---------------------------------------------------------------------------
# JSONL I/O + Checkpointing
# ---------------------------------------------------------------------------

def count_jsonl_lines(path: str | Path) -> int:
    """Count valid non-empty JSON lines in a file.

    Ignores trailing partial lines (e.g. from a crash mid-write).

    Args:
        path: Path to the JSONL file.

    Returns:
        Number of valid JSON lines. 0 if file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return 0
    count = 0
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                break  # partial line at end from crash — stop counting
    return count


def sanitize_jsonl(path: str | Path) -> int:
    """Remove any trailing partial/corrupt line from a JSONL file.

    Important for resume: if a crash happened mid-append, the last line
    may be truncated JSON. This function rewrites the file keeping only
    valid lines.

    Args:
        path: Path to the JSONL file.

    Returns:
        Number of valid lines kept.
    """
    p = Path(path)
    if not p.exists():
        return 0
    valid_lines: list[str] = []
    with open(p) as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
                valid_lines.append(stripped)
            except json.JSONDecodeError:
                _log.warning(f"Removing partial/corrupt line at end of {p}")
                break
    with open(p, "w") as f:
        for line in valid_lines:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return len(valid_lines)


def load_jsonl(path: str | Path, required_key: str | None = None) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Skips malformed lines with a warning instead of crashing.

    Args:
        path: Path to the JSONL file.
        required_key: If set, skip entries missing this key.

    Returns:
        List of parsed dicts. Empty list if file doesn't exist.
    """
    p = Path(path)
    if not p.exists():
        return []
    data: list[dict] = []
    skipped = 0
    with open(p) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                if skipped <= 5:
                    _log.warning(f"malformed JSON at {p}:{line_num}, skipping")
                continue
            if required_key and required_key not in entry:
                skipped += 1
                if skipped <= 5:
                    _log.warning(f"missing key '{required_key}' at {p}:{line_num}, skipping")
                continue
            data.append(entry)
    if skipped > 0:
        _log.warning(f"skipped {skipped} bad lines in {p}")
    return data


def append_jsonl(path: str | Path, items: list[dict]) -> None:
    """Append dicts as JSON lines to a file with fsync.

    Creates parent directories and the file if needed. No-op if items is empty.

    Args:
        path: Path to the JSONL file.
        items: List of dicts to append.
    """
    if not items:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def truncate_file(path: str | Path) -> None:
    """Create or truncate a file to zero bytes."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    open(p, "w").close()


def atomic_write_jsonl(path: str | Path, items: list[dict]) -> None:
    """Write dicts as JSONL atomically via temp-file-then-rename.

    The target path is either the old complete file or the new complete
    file — never partial. Safe against crashes.

    Args:
        path: Target JSONL file path.
        items: List of dicts to write.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp", prefix=p.stem + "_")
    try:
        with os.fdopen(fd, "w") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, p)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_progress(path: str | Path, value: int) -> None:
    """Atomically write a progress counter to a file.

    Args:
        path: Progress file path.
        value: Integer progress value.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w") as f:
        f.write(str(value))
        f.flush()
        os.fsync(f.fileno())
    os.rename(tmp, p)


def read_progress(path: str | Path) -> int:
    """Read a progress counter from a file.

    Args:
        path: Progress file path.

    Returns:
        Integer progress value. 0 if file is missing or corrupt.
    """
    p = Path(path)
    if not p.exists():
        return 0
    try:
        with open(p) as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        _log.warning(f"Corrupt progress file {p}, resetting to 0")
        return 0


def init_output_with_resume(output_path: str | Path, resume: bool) -> int:
    """Initialize output for a step where 1 output line = 1 input entry.

    For steps with no drops (e.g. generate_responses where every instruction
    produces exactly one output line).

    Args:
        output_path: Path to the output JSONL file.
        resume: If True, sanitize and count existing lines. If False, truncate.

    Returns:
        Number of already-completed entries (0 on fresh run).
    """
    p = Path(output_path)
    if resume and p.exists():
        n = sanitize_jsonl(p)
        _log.info(f"Resuming: {n} valid entries in {p}")
        return n
    truncate_file(p)
    return 0


# ---------------------------------------------------------------------------
# Text normalization + deduplication
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def dedupe_exact(instructions: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for inst in instructions:
        key = normalize_text(inst)
        if key not in seen:
            seen.add(key)
            deduped.append(inst)
    return deduped


def dedupe_minhash(instructions: list[str], threshold: float = 0.7, num_perm: int = 128) -> list[str]:
    """Near-duplicate removal via MinHash LSH."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep: list[str] = []
    for idx, inst in enumerate(tqdm(instructions, desc="MinHash dedupe")):
        tokens = normalize_text(inst).split()
        if len(tokens) < 3:
            keep.append(inst)
            continue
        mh = MinHash(num_perm=num_perm)
        for token in tokens:
            mh.update(token.encode("utf-8"))
        key = f"inst_{idx}"
        if not lsh.query(mh):
            lsh.insert(key, mh)
            keep.append(inst)
    return keep


# ---------------------------------------------------------------------------
# System prompt sampling
# ---------------------------------------------------------------------------

def sample_system_prompt(pool: list[dict]) -> str:
    """Sample a system prompt from the weighted pool.

    Args:
        pool: List of dicts with ``prompt`` and ``weight`` keys.

    Returns:
        A sampled prompt string. Empty string if total weight is zero.
    """
    prompts = [p["prompt"] for p in pool]
    weights = [p["weight"] for p in pool]
    total = sum(weights)
    if total <= 0:
        return ""
    return random.choices(prompts, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Chat template helpers
# ---------------------------------------------------------------------------

SPLIT_MARKER = "<<__REPLAY_SPLIT_MARKER__>>"


def build_prequery_prefix(tokenizer) -> str:
    """Build the raw text prefix ending at the user-turn content start.

    Uses a dummy marker to find the split point in the rendered chat
    template, making this model-agnostic.

    Args:
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.

    Returns:
        The template prefix string up to where user content begins.
    """
    dummy = [{"role": "user", "content": SPLIT_MARKER}]
    full_text = tokenizer.apply_chat_template(
        dummy, tokenize=False, add_generation_prompt=False
    )
    return full_text.split(SPLIT_MARKER)[0]


def build_user_turn_prefix(tokenizer, conversation: list[dict[str, str]]) -> str:
    """Build the full conversation text ending where the next user turn begins.

    Used in multi-turn generation: the model continues from this prefix
    to generate a follow-up user message.

    Args:
        tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
        conversation: List of message dicts (role + content) so far.

    Returns:
        Rendered text up to the next user content boundary.
    """
    conv_with_dummy = conversation + [{"role": "user", "content": SPLIT_MARKER}]
    full_text = tokenizer.apply_chat_template(
        conv_with_dummy, tokenize=False, add_generation_prompt=False
    )
    return full_text.split(SPLIT_MARKER)[0]


def extract_instruction(raw_text: str) -> str:
    """Extract just the user instruction from raw model output.

    Truncates at assistant-turn markers and special tokens to isolate
    the instruction portion.

    Args:
        raw_text: Raw text output from the model.

    Returns:
        Cleaned instruction string.
    """
    for marker in ["<|im_start|>assistant", "<|im_start|>system", "\nassistant\n"]:
        idx = raw_text.find(marker)
        if idx != -1:
            raw_text = raw_text[:idx]
    idx = raw_text.find("<|im_end|>")
    if idx != -1:
        raw_text = raw_text[:idx]
    return raw_text.strip()
