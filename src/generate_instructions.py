"""
Step 1: Instruction Generation

Reconstructs the model's latent instruction distribution by feeding only the
user-turn chat-template prefix and letting the model continue.

Checkpointing: raw instructions are flushed to a .raw file after each chunk.
Budget counts LINES WRITTEN, not prompts sent. Dedupe runs at the end.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from vllm import LLM, SamplingParams

from src.config_schema import load_and_validate_config
from src.log import setup_logging
from src.utils import (
    append_jsonl,
    atomic_write_jsonl,
    build_prequery_prefix,
    count_jsonl_lines,
    dedupe_exact,
    dedupe_minhash,
    extract_instruction,
    load_jsonl,
    sanitize_jsonl,
    truncate_file,
)


def generate_instructions(config_path: str, quick: bool = False, resume: bool = False) -> None:
    cfg = load_and_validate_config(config_path)
    log = setup_logging(cfg.get("log_level", "INFO"))

    model_name = cfg["model"]
    seed = cfg.get("seed", 42)
    icfg = cfg["instruction_generation"]
    n = icfg.get("quick_n", 1000) if quick else icfg["n"]
    chunk_size = icfg.get("chunk_size", 5000)
    samp = icfg["sampling"]
    min_len = icfg.get("min_length", 10)
    max_len = icfg.get("max_length", 2048)
    dedupe_cfg = cfg.get("dedupe", {})
    mh_threshold = dedupe_cfg.get("minhash_threshold", 0.7)
    mh_num_perm = dedupe_cfg.get("minhash_num_perm", 128)
    stop_tokens = cfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"])

    output_path = Path(cfg["paths"]["instructions"])
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # --- Checkpoint: count WRITTEN lines in raw file (#6) ---
    already_written = 0
    if resume and raw_path.exists():
        already_written = sanitize_jsonl(raw_path)  # also fixes partial last line
        log.info(f"Resuming: {already_written} raw instructions already in {raw_path}")
    else:
        truncate_file(raw_path)

    remaining = max(0, n - already_written)
    if remaining == 0:
        log.info(f"All {n} raw instructions already written. Skipping to dedupe.")
    else:
        tp = cfg.get("tensor_parallel_size", 1)
        log.info(f"Loading model: {model_name} (tensor_parallel_size={tp})")
        llm = LLM(model=model_name, tensor_parallel_size=tp)
        tokenizer = llm.get_tokenizer()
        prefix = build_prequery_prefix(tokenizer)
        log.info(f"Pre-query prefix ({len(prefix)} chars): {repr(prefix[:80])}...")

        sampling_params = SamplingParams(
            temperature=samp["temperature"],
            top_p=samp["top_p"],
            top_k=samp["top_k"],
            max_tokens=samp["max_tokens"],
            stop=stop_tokens,
        )

        # #6: Budget counts LINES WRITTEN, not prompts sent.
        # Over-generate per chunk to compensate for length filtering.
        written_this_run = 0
        chunk_num = 0
        max_empty_chunks = 5  # #1: abort if this many consecutive chunks yield 0 valid instructions
        consecutive_empty = 0
        log.info(f"Generating until {remaining} more instructions are written (chunk_size={chunk_size})...")

        while written_this_run < remaining:
            chunk_num += 1
            prompts_to_send = min(chunk_size, remaining - written_this_run + chunk_size // 2)
            prompts = [prefix] * prompts_to_send

            log.info(f"  Chunk {chunk_num}: sending {prompts_to_send} prompts...")
            outputs = llm.generate(prompts, sampling_params)

            chunk_items: list[dict] = []
            for out in outputs:
                text = extract_instruction(out.outputs[0].text)
                if min_len <= len(text) <= max_len:
                    chunk_items.append({"instruction": text})
                    if len(chunk_items) + written_this_run >= remaining:
                        break

            append_jsonl(raw_path, chunk_items)
            written_this_run += len(chunk_items)
            total = already_written + written_this_run
            log.info(f"  Flushed {len(chunk_items)} instructions (total raw: {total}/{n})")

            if len(chunk_items) == 0:
                consecutive_empty += 1
                if consecutive_empty >= max_empty_chunks:
                    log.error(
                        f"  {max_empty_chunks} consecutive chunks yielded 0 valid instructions. "
                        f"Check model output, min_length/max_length, and stop_tokens. Aborting."
                    )
                    break
            else:
                consecutive_empty = 0

    # --- Dedupe (#16: use required_key for safety) ---
    log.info(f"Loading raw instructions from {raw_path}...")
    raw_data = load_jsonl(raw_path, required_key="instruction")
    all_instructions = [d["instruction"] for d in raw_data]
    log.info(f"Raw instructions: {len(all_instructions)}")

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
    parser = argparse.ArgumentParser(description="Step 1: Generate instructions from model's latent distribution")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--quick", action="store_true", help="Quick mode: generate fewer instructions")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    generate_instructions(args.config, quick=args.quick, resume=args.resume)


if __name__ == "__main__":
    main()
