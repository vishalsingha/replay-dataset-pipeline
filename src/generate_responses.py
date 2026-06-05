"""
Step 2: Multi-Response Generation

Checkpointing: candidates are flushed after each chunk. On --resume,
output is sanitized (partial lines removed) and valid line count is used
to skip already-processed instructions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm import SamplingParams

from src.backends import build_backend
from src.config_schema import load_and_validate_config
from src.log import setup_logging
from src.utils import append_jsonl, init_output_with_resume, load_jsonl


def generate_responses(
    config_path: str,
    instructions_override: str | None = None,
    output_override: str | None = None,
    resume: bool = False,
) -> None:
    cfg = load_and_validate_config(config_path)
    log = setup_logging(cfg.get("log_level", "INFO"))

    rcfg = cfg["response_generation"]
    L = rcfg["L"]
    chunk_size = rcfg.get("chunk_size", 1000)
    samp = rcfg["sampling"]

    instructions_path = Path(instructions_override) if instructions_override else Path(cfg["paths"]["instructions"])
    output_path = Path(output_override) if output_override else Path(cfg["paths"]["candidates"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instructions_path.exists():
        log.error(f"Instructions file not found: {instructions_path}")
        sys.exit(1)

    log.info(f"Loading instructions from {instructions_path}...")
    entries = load_jsonl(instructions_path, required_key="messages")
    log.info(f"Loaded {len(entries)} instructions")

    if not entries:
        log.error("No valid instructions found.")
        sys.exit(1)

    done = init_output_with_resume(output_path, resume)
    if done >= len(entries):
        log.info(f"All {len(entries)} instructions already processed. Nothing to do.")
        return
    if done > 0:
        log.info(f"Skipping first {done} instructions (already processed)")

    tp = cfg.get("tensor_parallel_size", 1)
    generators = [build_backend(g, tensor_parallel_size=tp) for g in cfg["committee"]["generators"]]
    log.info(f"Committee generators: {[g.model for g in generators]}")

    sp_kwargs: dict = dict(
        n=L,
        temperature=samp["temperature"],
        top_p=samp["top_p"],
        top_k=samp["top_k"],
        max_tokens=samp["max_tokens"],
        presence_penalty=samp.get("presence_penalty", 0.0),
    )
    if samp.get("min_p", 0) > 0:
        sp_kwargs["min_p"] = samp["min_p"]
    sampling_params = SamplingParams(**sp_kwargs)

    remaining = entries[done:]
    total_remaining = len(remaining)

    log.info(f"Processing {total_remaining} instructions in chunks of {chunk_size}...")

    for chunk_start in range(0, total_remaining, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_remaining)
        chunk_entries = remaining[chunk_start:chunk_end]
        chunk_num = chunk_start // chunk_size + 1

        # Use the stored messages (system + user) directly as the conversation context.
        conversations: list[list[dict[str, str]]] = [e["messages"] for e in chunk_entries]

        chunk_candidates: list[list[dict[str, str]]] = [[] for _ in chunk_entries]

        for gen in generators:
            log.info(f"  Chunk {chunk_num}: generating with {gen.model} ({len(chunk_entries)} instructions, L={L})...")
            results = gen.chat(conversations, sampling_params)

            if len(results) != len(conversations):
                log.warning(f"  Expected {len(conversations)} results, got {len(results)}")

            for i in range(min(len(results), len(conversations))):
                for text in results[i]:
                    chunk_candidates[i].append({"model": gen.model, "text": text.strip()})

        chunk_items: list[dict] = []
        for i, entry in enumerate(chunk_entries):
            chunk_items.append({
                "messages": entry["messages"],
                "candidates": chunk_candidates[i],
            })
        append_jsonl(output_path, chunk_items)

        processed = done + chunk_end
        log.info(f"  Checkpoint: {processed}/{len(entries)} instructions done")

    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate candidate responses from committee")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--instructions", default=None, help="Override instructions JSONL path")
    parser.add_argument("--output", default=None, help="Override output candidates JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    generate_responses(args.config, instructions_override=args.instructions, output_override=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
