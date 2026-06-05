"""
Step 1b: Multi-Turn Conversation Generation

Checkpointing: uses a .progress file (like filter) since quality drops mean
output lines != input count. On --resume, re-processes from last progress.

Requires a vllm_local backend (needs tokenizer + generate_raw for follow-ups).
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
from src.utils import (
    append_jsonl,
    build_user_turn_prefix,
    extract_instruction,
    load_jsonl,
    read_progress,
    truncate_file,
    write_progress,
)


def generate_multiturn(
    config_path: str,
    instructions_override: str | None = None,
    output_override: str | None = None,
    resume: bool = False,
) -> None:
    cfg = load_and_validate_config(config_path)
    log = setup_logging(cfg.get("log_level", "INFO"))

    mcfg = cfg["multiturn"]
    fraction = mcfg["fraction"]
    min_turns = mcfg["min_turns"]
    max_turns = mcfg["max_turns"]
    chunk_size = mcfg.get("chunk_size", 500)
    samp_resp = cfg["response_generation"]["sampling"]
    samp_followup = mcfg.get("followup_sampling", {})
    seed = cfg.get("seed", 42)
    stop_tokens = cfg.get("stop_tokens", ["<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"])

    instructions_path = Path(instructions_override) if instructions_override else Path(cfg["paths"]["instructions"])
    output_path = Path(output_override) if output_override else Path(cfg["paths"]["multiturn"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading instructions...")
    entries = load_jsonl(instructions_path, required_key="messages")
    log.info(f"Loaded {len(entries)} instructions")

    if not entries:
        log.error("No instructions found. Run Step 1 first.")
        return

    random.seed(seed)
    n_multiturn = int(len(entries) * fraction)
    if n_multiturn <= 0:
        log.info("multiturn.fraction=0 → no multi-turn conversations to generate.")
        return

    selected = random.sample(entries, min(n_multiturn, len(entries)))
    log.info(f"Selected {len(selected)} instructions for multi-turn extension")

    turn_counts = [random.randint(min_turns, max_turns) for _ in selected]

    # --- Checkpoint: #8 use .progress file (output lines != input count due to drops) ---
    progress_path = output_path.with_suffix(".progress")
    done = 0
    if resume:
        done = read_progress(progress_path)
        log.info(f"Resuming: {done} conversations already processed")
    else:
        truncate_file(output_path)
        write_progress(progress_path, 0)

    if done >= len(selected):
        log.info(f"All {len(selected)} conversations already generated. Nothing to do.")
        return

    remaining_selected = selected[done:]
    remaining_turns = turn_counts[done:]

    # #9-10: require vllm_local (needs tokenizer + generate_raw)
    tp = cfg.get("tensor_parallel_size", 1)
    gen_cfg = cfg["committee"]["generators"][0]
    backend = build_backend(gen_cfg, tensor_parallel_size=tp)

    if not backend.supports_raw_generation:
        log.error(
            f"Multi-turn generation requires a vllm_local backend (needs tokenizer + generate_raw). "
            f"Got: {gen_cfg['type']} ({gen_cfg['model']})"
        )
        sys.exit(1)

    tokenizer = backend.tokenizer
    log.info(f"Using backend: {backend.model} (tensor_parallel_size={tp})")

    response_params = SamplingParams(
        temperature=samp_resp["temperature"],
        top_p=samp_resp["top_p"],
        top_k=samp_resp["top_k"],
        max_tokens=samp_resp["max_tokens"],
        presence_penalty=samp_resp.get("presence_penalty", 0.0),
    )

    followup_params = SamplingParams(
        temperature=samp_followup.get("temperature", 0.9),
        top_p=samp_followup.get("top_p", 0.95),
        top_k=samp_followup.get("top_k", 30),
        max_tokens=samp_followup.get("max_tokens", 512),
        stop=stop_tokens,
    )

    total_kept = 0
    total_dropped = 0

    log.info(f"Processing {len(remaining_selected)} conversations in chunks of {chunk_size}...")

    for chunk_start in range(0, len(remaining_selected), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(remaining_selected))
        chunk_entries = remaining_selected[chunk_start:chunk_end]
        chunk_turns = list(remaining_turns[chunk_start:chunk_end])
        chunk_num = chunk_start // chunk_size + 1

        # Start each conversation from the stored messages (system + user).
        conversations: list[list[dict[str, str]]] = [
            list(e["messages"]) for e in chunk_entries
        ]

        max_needed = max(chunk_turns) if chunk_turns else 0
        current_turn = 1

        while current_turn <= max_needed:
            active_indices = [i for i, tc in enumerate(chunk_turns) if current_turn <= tc]
            if not active_indices:
                break

            # --- Assistant responses ---
            log.info(f"  Chunk {chunk_num}, turn {current_turn}/{max_needed}: "
                     f"assistant responses for {len(active_indices)} conversations...")
            active_convs = [conversations[i] for i in active_indices]
            results = backend.chat(active_convs, response_params)

            if len(results) != len(active_indices):
                log.warning(f"  Expected {len(active_indices)} results, got {len(results)}")

            for j, idx in enumerate(active_indices):
                if j < len(results) and results[j]:  # #11: guard empty result
                    conversations[idx].append({"role": "assistant", "content": results[j][0].strip()})
                else:
                    conversations[idx].append({"role": "assistant", "content": ""})

            # --- Follow-up user messages ---
            need_followup = [i for i in active_indices if current_turn < chunk_turns[i]]
            if not need_followup:
                current_turn += 1
                continue

            log.info(f"  Chunk {chunk_num}, turn {current_turn}/{max_needed}: "
                     f"follow-up user messages for {len(need_followup)} conversations...")
            followup_prefixes = [
                build_user_turn_prefix(tokenizer, conversations[i])
                for i in need_followup
            ]
            followup_texts = backend.generate_raw(followup_prefixes, followup_params)

            if len(followup_texts) != len(need_followup):
                log.warning(f"  Expected {len(need_followup)} follow-ups, got {len(followup_texts)}")

            for j, idx in enumerate(need_followup):
                if j < len(followup_texts):
                    followup_text = extract_instruction(followup_texts[j])
                else:
                    followup_text = ""
                if followup_text:
                    conversations[idx].append({"role": "user", "content": followup_text})
                else:
                    chunk_turns[idx] = current_turn

            current_turn += 1

        # Quality filter + flush
        chunk_results: list[dict] = []
        for conv in conversations:
            assistant_turns = [m for m in conv if m["role"] == "assistant"]
            if not assistant_turns or any(not t["content"].strip() for t in assistant_turns):
                total_dropped += 1
                continue
            chunk_results.append({"messages": conv})
            total_kept += 1

        # #8: append output, THEN update progress atomically
        append_jsonl(output_path, chunk_results)
        progress_so_far = done + chunk_end
        write_progress(progress_path, progress_so_far)

        log.info(f"  Checkpoint: {progress_so_far}/{len(selected)} conversations "
                 f"(kept={total_kept}, dropped={total_dropped})")

    log.info(f"Final: kept {total_kept}, dropped {total_dropped}")
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Step 1b: Generate multi-turn conversations")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--instructions", default=None, help="Override input instructions JSONL path")
    parser.add_argument("--output", default=None, help="Override output multi-turn JSONL path")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    generate_multiturn(args.config, instructions_override=args.instructions, output_override=args.output, resume=args.resume)


if __name__ == "__main__":
    main()
