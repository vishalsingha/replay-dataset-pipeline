"""
Step 3: Filtering High-Quality Responses

Checkpointing: uses a .progress file to track input index (since output
lines != input entries due to drops). Progress is written atomically.
On --resume, corrupt .progress defaults to 0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm import SamplingParams

from src.backends import build_backend, parse_judge_score
from src.config_schema import load_and_validate_config
from src.log import setup_logging
from src.prompts import format_judge_prompt
from src.utils import append_jsonl, load_jsonl, read_progress, truncate_file, write_progress


def filter_responses(
    config_path: str,
    candidates_override: str | None = None,
    output_override: str | None = None,
    resume: bool = False,
) -> None:
    cfg = load_and_validate_config(config_path)
    log = setup_logging(cfg.get("log_level", "INFO"))

    fcfg = cfg["filtering"]
    chunk_size = fcfg.get("chunk_size", 1000)
    min_score = fcfg.get("min_score", 3.0)
    samp = fcfg["sampling"]

    candidates_path = Path(candidates_override) if candidates_override else Path(cfg["paths"]["candidates"])
    output_path = Path(output_override) if output_override else Path(cfg["paths"]["replay"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not candidates_path.exists():
        log.error(f"Candidates file not found: {candidates_path}")
        sys.exit(1)

    log.info(f"Loading candidates from {candidates_path}...")
    entries = load_jsonl(candidates_path, required_key="candidates")
    log.info(f"Loaded {len(entries)} entries")

    if not entries:
        log.error("No valid candidate entries found.")
        sys.exit(1)

    # --- Checkpoint: #1 atomic progress, #2 corrupt-safe read ---
    progress_path = output_path.with_suffix(".progress")
    done = 0
    if resume:
        done = read_progress(progress_path)  # #2: returns 0 on corrupt/missing
        log.info(f"Resuming: {done} entries already processed")
    else:
        truncate_file(output_path)
        write_progress(progress_path, 0)

    if done >= len(entries):
        log.info(f"All {len(entries)} entries already processed. Nothing to do.")
        return

    tp = cfg.get("tensor_parallel_size", 1)
    judges = [build_backend(j, tensor_parallel_size=tp) for j in cfg["committee"]["judges"]]
    log.info(f"Committee judges: {[j.model for j in judges]}")

    sampling_params = SamplingParams(
        temperature=samp["temperature"],
        top_p=samp["top_p"],
        top_k=samp["top_k"],
        max_tokens=samp["max_tokens"],
    )

    remaining_entries = entries[done:]
    total_kept = 0
    total_dropped = 0
    total_scored = 0
    parse_failures = 0
    sample_failures: list[str] = []

    log.info(f"Processing {len(remaining_entries)} entries in chunks of {chunk_size}...")

    for chunk_start in range(0, len(remaining_entries), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(remaining_entries))
        chunk = remaining_entries[chunk_start:chunk_end]
        chunk_num = chunk_start // chunk_size + 1

        scores: list[list[list[float]]] = [
            [[] for _ in entry["candidates"]] for entry in chunk
        ]

        for judge in judges:
            log.info(f"  Chunk {chunk_num}: judging with {judge.model} ({len(chunk)} entries)...")

            flat_index: list[tuple[int, int]] = []
            conversations: list[list[dict[str, str]]] = []
            for e_idx, entry in enumerate(chunk):
                for c_idx, cand in enumerate(entry["candidates"]):
                    prompt = format_judge_prompt(
                        instruction=entry.get("instruction", ""),
                        response=cand.get("text", ""),
                    )
                    conversations.append([{"role": "user", "content": prompt}])
                    flat_index.append((e_idx, c_idx))

            results = judge.chat(conversations, sampling_params)

            # #15: check result count
            if len(results) != len(flat_index):
                log.warning(
                    f"  Result count mismatch: expected {len(flat_index)}, got {len(results)}. "
                    f"{abs(len(flat_index) - len(results))} entries affected."
                )

            for idx in range(min(len(flat_index), len(results))):
                e_idx, c_idx = flat_index[idx]
                total_scored += 1
                # #13: guard empty result list
                if not results[idx]:
                    parse_failures += 1
                    if len(sample_failures) < 10:
                        sample_failures.append(f"  entry={e_idx} cand={c_idx}: empty result list")
                    continue
                score, error = parse_judge_score(results[idx][0])
                if score is not None:
                    scores[e_idx][c_idx].append(score)
                else:
                    parse_failures += 1
                    if len(sample_failures) < 10:
                        sample_failures.append(f"  entry={e_idx} cand={c_idx}: {error}")

        chunk_results: list[dict] = []
        for e_idx, entry in enumerate(chunk):
            best_score = -1.0
            best_c_idx = -1

            for c_idx in range(len(entry["candidates"])):
                if scores[e_idx][c_idx]:
                    avg = sum(scores[e_idx][c_idx]) / len(scores[e_idx][c_idx])
                else:
                    avg = 0.0
                if avg > best_score:
                    best_score = avg
                    best_c_idx = c_idx

            if best_score < min_score or best_c_idx < 0:
                total_dropped += 1
                continue

            best_text = entry["candidates"][best_c_idx].get("text", "")
            sys_prompt = entry.get("system_prompt", "")

            messages: list[dict[str, str]] = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": entry.get("instruction", "")})
            messages.append({"role": "assistant", "content": best_text})

            chunk_results.append({"messages": messages})
            total_kept += 1

        # #1: append output THEN update progress atomically
        append_jsonl(output_path, chunk_results)
        progress_so_far = done + chunk_end
        write_progress(progress_path, progress_so_far)

        log.info(f"  Checkpoint: {progress_so_far}/{len(entries)} entries processed "
                 f"(kept={total_kept}, dropped={total_dropped}, parse_failures={parse_failures})")

    log.info("=" * 60)
    log.info("FILTERING SUMMARY")
    log.info("=" * 60)
    log.info(f"  Total entries processed: {len(entries)}")
    log.info(f"  Total judge calls: {total_scored}")
    log.info(f"  Successful score parses: {total_scored - parse_failures}")
    log.info(f"  Parse failures: {parse_failures} ({parse_failures / max(total_scored, 1) * 100:.1f}%)")
    log.info(f"  Entries kept (score >= {min_score}): {total_kept}")
    log.info(f"  Entries dropped: {total_dropped}")
    if sample_failures:
        log.info(f"  Sample parse failures (first {len(sample_failures)}):")
        for msg in sample_failures:
            log.info(msg)
    log.info("=" * 60)
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Step 3: Filter candidates and build replay dataset")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--candidates", default=None, help="Override candidates JSONL path")
    parser.add_argument("--output", default=None, help="Override output replay JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    filter_responses(args.config, candidates_override=args.candidates, output_override=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
