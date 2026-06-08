"""
Pull instructions from public datasets on HuggingFace.

Checkpointing: per-source progress tracks ROWS SEEN in the stream (not valid
collected count) so resume skips exactly the right number of raw rows (#3).
Progress and raw file are both cleared on non-resume runs (#4).
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


# ---------------------------------------------------------------------------
# Per-dataset extraction functions
# Each returns (instruction, system_prompt). system_prompt may be "".
# ---------------------------------------------------------------------------

def extract_openorca(example: dict) -> tuple[str | None, str]:
    q = example.get("question", "")
    sys = example.get("system_prompt", "")
    inst = q.strip() if q and q.strip() else None
    return inst, sys.strip()


def extract_openhermes(example: dict) -> tuple[str | None, str]:
    convs = example.get("conversations", [])
    sys = ""
    inst = None
    for turn in convs:
        role = turn.get("from", "")
        val = turn.get("value", "")
        if role == "system" and val and val.strip():
            sys = val.strip()
        elif role == "human" and val and val.strip() and inst is None:
            inst = val.strip()
    return inst, sys


def extract_ultrachat(example: dict) -> tuple[str | None, str]:
    p = example.get("prompt", "")
    inst = p.strip() if p and p.strip() else None
    # ultrachat stores messages list; check for system turn
    sys = ""
    for msg in example.get("messages", []):
        if msg.get("role") == "system":
            sys = msg.get("content", "").strip()
            break
    return inst, sys


def extract_codefeedback(example: dict) -> tuple[str | None, str]:
    q = example.get("query", "")
    inst = q.strip() if q and q.strip() else None
    return inst, ""


def extract_mathinstruct(example: dict) -> tuple[str | None, str]:
    inst_text = example.get("instruction", "")
    inst = inst_text.strip() if inst_text and inst_text.strip() else None
    return inst, ""


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

MAX_STREAM_ITERATIONS = 10_000_000  # #14: safety cap to prevent infinite loop


def pull_public_instructions(config_path: str, output_override: str | None = None, resume: bool = False) -> None:
    cfg = load_and_validate_config(config_path)
    log = setup_logging(cfg.get("log_level", "INFO"))

    # #15: check public_instructions is not None
    pcfg = cfg.get("public_instructions")
    if not pcfg:
        log.error("public_instructions section missing or null in config. Nothing to do.")
        return

    sources = pcfg["sources"]
    min_len = pcfg.get("min_length", 10)
    max_len = pcfg.get("max_length", 2048)

    output_path = Path(output_override) if output_override else Path(cfg["paths"]["public_instructions"])
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-source progress: tracks {dataset_name: {rows_seen: N, collected: N}}
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
        # #4: clear BOTH raw file AND progress on non-resume
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
            # #14: safety cap
            if iterations > MAX_STREAM_ITERATIONS:
                log.warning(f"  Hit max stream iterations ({MAX_STREAM_ITERATIONS}) for {name}, stopping")
                break

            rows_seen += 1

            # #3: skip by rows_seen, not collected count
            if rows_seen <= rows_to_skip:
                continue

            inst, sys_prompt = extractor(example)
            if inst and min_len <= len(inst) <= max_len:
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": sys_prompt or "You are a helpful assistant."},
                    {"role": "user", "content": inst},
                ]
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

        log.info(f"  Collected {len(collected)} new instructions from {name} "
                 f"(total: {source_progress[name]['collected']})")

    # --- Dedupe ---
    dedupe_cfg = cfg.get("dedupe", {})
    mh_threshold = dedupe_cfg.get("minhash_threshold", 0.7)
    mh_num_perm = dedupe_cfg.get("minhash_num_perm", 128)

    log.info(f"Loading raw instructions from {raw_path}...")
    raw_data = load_jsonl(raw_path, required_key="messages")

    def _user_content(entry: dict) -> str:
        for msg in entry.get("messages", []):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    all_texts = [_user_content(d) for d in raw_data]
    log.info(f"Total raw instructions: {len(all_texts)}")

    log.info("Exact deduplication...")
    seen: set[str] = set()
    exact_deduped: list[dict] = []
    for entry, text in zip(raw_data, all_texts):
        key = normalize_text(text)
        if key and key not in seen:
            seen.add(key)
            exact_deduped.append(entry)
    log.info(f"After exact dedupe: {len(exact_deduped)}")

    log.info(f"MinHash near-duplicate removal (threshold={mh_threshold}, num_perm={mh_num_perm})...")
    deduped_texts = [_user_content(d) for d in exact_deduped]
    keep_indices = dedupe_minhash_indices(deduped_texts, threshold=mh_threshold, num_perm=mh_num_perm)
    final_entries = [exact_deduped[i] for i in keep_indices]
    log.info(f"After MinHash dedupe: {len(final_entries)}")

    log.info(f"Writing {len(final_entries)} instructions to {output_path} (atomic)")
    atomic_write_jsonl(output_path, final_entries)

    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Pull instructions from public HuggingFace datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None, help="Override output instructions JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    pull_public_instructions(args.config, output_override=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
