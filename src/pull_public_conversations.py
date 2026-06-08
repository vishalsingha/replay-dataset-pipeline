"""
Pull complete conversations (system + instruction + response) from public datasets.

Unlike pull_public_instructions which only extracts the user query, this pulls
the full conversation ready for SFT — no generate_responses step needed.

Checkpointing: same pattern as pull_public_instructions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from src.config_schema import load_and_validate_config
from src.log import setup_logging
from src.utils import (
    append_jsonl,
    atomic_write_jsonl,
    dedupe_minhash_indices,
    load_jsonl,
    normalize_text,
    truncate_file,
)

DEFAULT_SYSTEM = "You are a helpful assistant."


# ---------------------------------------------------------------------------
# Per-dataset extraction functions
# Each returns a messages list [system, user, assistant, ...] or None.
# ---------------------------------------------------------------------------

def extract_openorca(example: dict) -> list[dict[str, str]] | None:
    q = (example.get("question") or "").strip()
    r = (example.get("response") or "").strip()
    if not q or not r:
        return None
    sys = (example.get("system_prompt") or "").strip() or DEFAULT_SYSTEM
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": q},
        {"role": "assistant", "content": r},
    ]


def extract_openhermes(example: dict) -> list[dict[str, str]] | None:
    convs = example.get("conversations", [])
    if not convs:
        return None

    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    messages: list[dict[str, str]] = []
    for turn in convs:
        role = role_map.get(turn.get("from", ""))
        val = (turn.get("value") or "").strip()
        if role and val:
            messages.append({"role": role, "content": val})

    has_user = any(m["role"] == "user" for m in messages)
    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not has_user or not has_assistant:
        return None

    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    return messages


def extract_ultrachat(example: dict) -> list[dict[str, str]] | None:
    msgs = example.get("messages", [])
    if not msgs:
        return None

    messages: list[dict[str, str]] = []
    for m in msgs:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role in ("system", "user", "assistant") and content:
            messages.append({"role": role, "content": content})

    has_user = any(m["role"] == "user" for m in messages)
    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not has_user or not has_assistant:
        return None

    if not messages or messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM})
    return messages


def extract_codefeedback(example: dict) -> list[dict[str, str]] | None:
    q = (example.get("query") or "").strip()
    r = (example.get("answer") or "").strip()
    if not q or not r:
        return None
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": r},
    ]


def extract_mathinstruct(example: dict) -> list[dict[str, str]] | None:
    q = (example.get("instruction") or "").strip()
    r = (example.get("output") or "").strip()
    if not q or not r:
        return None
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": r},
    ]


EXTRACTORS = {
    "Open-Orca/OpenOrca": extract_openorca,
    "teknium/OpenHermes-2.5": extract_openhermes,
    "HuggingFaceH4/ultrachat_200k": extract_ultrachat,
    "m-a-p/CodeFeedback-Filtered-Instruction": extract_codefeedback,
    "TIGER-Lab/MathInstruct": extract_mathinstruct,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MAX_STREAM_ITERATIONS = 10_000_000


def pull_public_conversations(
    config_path: str,
    output_override: str | None = None,
    resume: bool = False,
) -> None:
    cfg = load_and_validate_config(config_path)
    log = setup_logging(cfg.get("log_level", "INFO"))

    pcfg = cfg.get("public_conversations") or cfg.get("public_instructions")
    if not pcfg:
        log.error("public_conversations (or public_instructions) section missing in config.")
        return

    sources = pcfg["sources"]
    min_len = pcfg.get("min_length", 10)
    max_len = pcfg.get("max_length", 2048)
    min_response_len = pcfg.get("min_response_length", 10)

    output_path = (
        Path(output_override) if output_override
        else Path(cfg["paths"].get("public_conversations", "data/public_conversations/conversations.jsonl"))
    )
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    progress_path = output_path.with_suffix(".source_progress.json")
    source_progress: dict[str, dict[str, int]] = {}
    if resume and progress_path.exists():
        try:
            with open(progress_path) as f:
                source_progress = json.load(f)
            log.info(f"Resuming: {source_progress}")
        except (json.JSONDecodeError, OSError):
            log.warning("Corrupt source progress file, starting fresh")
            source_progress = {}
    else:
        truncate_file(raw_path)
        if progress_path.exists():
            progress_path.unlink()

    for source in sources:
        name = source["dataset"]
        split = source.get("split", "train")
        n = source.get("n", 5000)
        streaming = source.get("streaming", True)

        prog = source_progress.get(name, {"rows_seen": 0, "collected": 0})
        already_collected = prog.get("collected", 0)
        rows_to_skip = prog.get("rows_seen", 0)

        if already_collected >= n:
            log.info(f"Skipping {name} — already collected {already_collected}/{n}")
            continue

        extractor = EXTRACTORS.get(name)
        if extractor is None:
            log.warning(f"No extractor for {name}, skipping")
            continue

        log.info(f"Loading {name} (split={split}, target={n}, collected={already_collected}, "
                 f"rows_to_skip={rows_to_skip}, streaming={streaming})...")
        ds = load_dataset(name, split=split, streaming=streaming)

        collected: list[dict] = []
        rows_seen = 0
        iterations = 0

        for example in tqdm(ds, desc=f"  {name}", total=n):
            iterations += 1
            if iterations > MAX_STREAM_ITERATIONS:
                log.warning(f"  Hit max stream iterations for {name}, stopping")
                break

            rows_seen += 1
            if rows_seen <= rows_to_skip:
                continue

            messages = extractor(example)
            if messages is None:
                continue

            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_content = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            if not (min_len <= len(user_content) <= max_len):
                continue
            if len(assistant_content) < min_response_len:
                continue

            collected.append({"messages": messages})

            if already_collected + len(collected) >= n:
                break

        append_jsonl(raw_path, collected)

        source_progress[name] = {
            "rows_seen": rows_to_skip + rows_seen,
            "collected": already_collected + len(collected),
        }
        with open(progress_path, "w") as f:
            json.dump(source_progress, f)

        log.info(f"  Collected {len(collected)} conversations from {name} "
                 f"(total: {source_progress[name]['collected']})")

    # --- Dedupe on user content ---
    dedupe_cfg = cfg.get("dedupe", {})
    mh_threshold = dedupe_cfg.get("minhash_threshold", 0.7)
    mh_num_perm = dedupe_cfg.get("minhash_num_perm", 128)

    log.info(f"Loading raw conversations from {raw_path}...")
    raw_data = load_jsonl(raw_path, required_key="messages")

    def _user_content(entry: dict) -> str:
        for msg in entry.get("messages", []):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    log.info(f"Total raw conversations: {len(raw_data)}")

    log.info("Exact deduplication...")
    seen: set[str] = set()
    exact_deduped: list[dict] = []
    for entry in raw_data:
        key = normalize_text(_user_content(entry))
        if key and key not in seen:
            seen.add(key)
            exact_deduped.append(entry)
    log.info(f"After exact dedupe: {len(exact_deduped)}")

    log.info(f"MinHash near-duplicate removal (threshold={mh_threshold})...")
    deduped_texts = [_user_content(d) for d in exact_deduped]
    keep_indices = dedupe_minhash_indices(deduped_texts, threshold=mh_threshold, num_perm=mh_num_perm)
    final_entries = [exact_deduped[i] for i in keep_indices]
    log.info(f"After MinHash dedupe: {len(final_entries)}")

    log.info(f"Writing {len(final_entries)} conversations to {output_path}")
    atomic_write_jsonl(output_path, final_entries)
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Pull complete conversations from public HuggingFace datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None, help="Override output JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    pull_public_conversations(args.config, output_override=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
