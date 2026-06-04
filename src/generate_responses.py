"""
Step 2: Multi-Response Generation

Checkpointing: candidates are flushed after each chunk. On --resume,
output is sanitized (partial lines removed) and valid line count is used
to skip already-processed instructions.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from vllm import SamplingParams

from src.backends import build_backend
from src.config_schema import load_and_validate_config
from src.log import setup_logging
from src.utils import append_jsonl, init_output_with_resume, load_jsonl, sample_system_prompt


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
    system_prompt_pool = cfg["replay_system_prompts"]
    seed = cfg.get("seed", 42)

    instructions_path = Path(instructions_override) if instructions_override else Path(cfg["paths"]["instructions"])
    output_path = Path(output_override) if output_override else Path(cfg["paths"]["candidates"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instructions_path.exists():
        log.error(f"Instructions file not found: {instructions_path}")
        sys.exit(1)

    log.info(f"Loading instructions from {instructions_path}...")
    raw = load_jsonl(instructions_path, required_key="instruction")
    instructions = [d["instruction"] for d in raw]
    log.info(f"Loaded {len(instructions)} instructions")

    if not instructions:
        log.error("No valid instructions found.")
        sys.exit(1)

    random.seed(seed)
    sampled_sys_prompts = [sample_system_prompt(system_prompt_pool) for _ in instructions]

    # #7/#22: sanitize_jsonl removes partial last line, count_jsonl_lines for accuracy
    done = init_output_with_resume(output_path, resume)
    if done >= len(instructions):
        log.info(f"All {len(instructions)} instructions already processed. Nothing to do.")
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

    remaining_instructions = instructions[done:]
    remaining_sys = sampled_sys_prompts[done:]
    total_remaining = len(remaining_instructions)

    log.info(f"Processing {total_remaining} instructions in chunks of {chunk_size}...")

    for chunk_start in range(0, total_remaining, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_remaining)
        chunk_inst = remaining_instructions[chunk_start:chunk_end]
        chunk_sys = remaining_sys[chunk_start:chunk_end]
        chunk_num = chunk_start // chunk_size + 1

        conversations: list[list[dict[str, str]]] = []
        for inst, sys_prompt in zip(chunk_inst, chunk_sys):
            messages: list[dict[str, str]] = []
            if sys_prompt:
                messages.append({"role": "system", "content": sys_prompt})
            messages.append({"role": "user", "content": inst})
            conversations.append(messages)

        chunk_candidates: list[list[dict[str, str]]] = [[] for _ in chunk_inst]

        for gen in generators:
            log.info(f"  Chunk {chunk_num}: generating with {gen.model} ({len(chunk_inst)} instructions, L={L})...")
            results = gen.chat(conversations, sampling_params)

            # #15: check result count
            if len(results) != len(conversations):
                log.warning(f"  Expected {len(conversations)} results, got {len(results)}")

            for i in range(min(len(results), len(conversations))):
                for text in results[i]:
                    chunk_candidates[i].append({"model": gen.model, "text": text.strip()})

        chunk_items: list[dict] = []
        for i, inst in enumerate(chunk_inst):
            chunk_items.append({
                "instruction": inst,
                "system_prompt": chunk_sys[i],
                "candidates": chunk_candidates[i],
            })
        append_jsonl(output_path, chunk_items)

        processed = done + chunk_end
        log.info(f"  Checkpoint: {processed}/{len(instructions)} instructions done")

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
