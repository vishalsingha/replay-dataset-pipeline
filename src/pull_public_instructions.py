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
    dedupe_exact,
    dedupe_minhash,
    load_jsonl,
    truncate_file,
)


# ---------------------------------------------------------------------------
# Per-dataset extraction functions
# ---------------------------------------------------------------------------

def extract_openorca(example: dict) -> str | None:
    q = example.get("question", "")
    return q.strip() if q and q.strip() else None


def extract_openhermes(example: dict) -> str | None:
    convs = example.get("conversations", [])
    for turn in convs:
        if turn.get("from") == "human":
            val = turn.get("value", "")
            return val.strip() if val and val.strip() else None
    return None


def extract_ultrachat(example: dict) -> str | None:
    p = example.get("prompt", "")
    return p.strip() if p and p.strip() else None


def extract_codefeedback(example: dict) -> str | None:
    q = example.get("query", "")
    return q.strip() if q and q.strip() else None


def extract_mathinstruct(example: dict) -> str | None:
    inst = example.get("instruction", "")
    return inst.strip() if inst and inst.strip() else None


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


def pull_public_instructions(config_path: str, resume: bool = False) -> None:
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

    output_path = Path(cfg["paths"]["public_instructions"])
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

            inst = extractor(example)
            if inst and min_len <= len(inst) <= max_len:
                collected.append({"instruction": inst})

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
    raw_data = load_jsonl(raw_path)
    all_instructions = [d.get("instruction", "") for d in raw_data if d.get("instruction")]
    log.info(f"Total raw instructions: {len(all_instructions)}")

    log.info("Exact deduplication...")
    all_instructions = dedupe_exact(all_instructions)
    log.info(f"After exact dedupe: {len(all_instructions)}")

    log.info(f"MinHash near-duplicate removal (threshold={mh_threshold}, num_perm={mh_num_perm})...")
    all_instructions = dedupe_minhash(all_instructions, threshold=mh_threshold, num_perm=mh_num_perm)
    log.info(f"After MinHash dedupe: {len(all_instructions)}")

    log.info(f"Writing {len(all_instructions)} instructions to {output_path} (atomic)")
    atomic_write_jsonl(output_path, [{"instruction": inst} for inst in all_instructions])

    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Pull instructions from public HuggingFace datasets")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    pull_public_instructions(args.config, resume=args.resume)


if __name__ == "__main__":
    main()
