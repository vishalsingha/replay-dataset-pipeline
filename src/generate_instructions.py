"""
Step 1: Instruction Generation (MAGPIE / Replay)

Extracts the model's latent instruction distribution by feeding only the
pre-query chat-template prefix and letting the model continue. The model
generates instruction + response as one stream; we post-process to extract
just the instruction (truncate at the first turn-boundary marker).

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
    dedupe_minhash_indices,
    load_jsonl,
    normalize_text,
    sample_system_prompt,
    sanitize_jsonl,
    truncate_file,
)


FILTER_PROMPT = (
    "The following text was generated as a user message, but it may contain "
    "both the user's question/instruction AND an answer or explanation mixed in.\n\n"
    "Extract ONLY the user's original question or instruction. "
    "Remove any answer, explanation, elaboration, or assistant-like content.\n\n"
    "If the text is already a clean question or instruction, return it as-is.\n\n"
    "Text:\n{text}\n\n"
    "Extracted instruction:"
)


def generate_instructions(
    config_path: str,
    output_override: str | None = None,
    quick: bool = False,
    resume: bool = False,
) -> None:
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

    output_path = Path(output_override) if output_override else Path(cfg["paths"]["instructions"])
    raw_path = output_path.with_suffix(".raw.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    already_written = 0
    if resume and raw_path.exists():
        already_written = sanitize_jsonl(raw_path)
        log.info(f"Resuming: {already_written} raw instructions already in {raw_path}")
    else:
        truncate_file(raw_path)

    llm = None
    remaining = max(0, n - already_written)
    if remaining == 0:
        log.info(f"All {n} raw instructions already written. Skipping to dedupe.")
    else:
        tp = cfg.get("tensor_parallel_size", 1)
        log.info(f"Loading model: {model_name} (tp={tp})")
        llm = LLM(model=model_name, tensor_parallel_size=tp)
        tokenizer = llm.get_tokenizer()

        sys_prompt_pool = cfg.get("replay_system_prompts", [])
        log.info(f"System prompt pool: {len(sys_prompt_pool)} entries" if sys_prompt_pool
                 else "No replay_system_prompts in config")

        sampling_params = SamplingParams(
            temperature=samp["temperature"],
            top_p=samp["top_p"],
            top_k=samp["top_k"],
            max_tokens=samp["max_tokens"],
            skip_special_tokens=False,
        )

        _prefix_cache: dict[str, str] = {}

        def _get_prefix(sys_prompt: str) -> str:
            if sys_prompt not in _prefix_cache:
                _prefix_cache[sys_prompt] = build_prequery_prefix(
                    tokenizer, system_prompt=sys_prompt or None,
                )
            return _prefix_cache[sys_prompt]

        example = _get_prefix(
            sys_prompt_pool[0]["prompt"] if sys_prompt_pool else "",
        )
        log.info(f"Example prefix ({len(example)} chars): {repr(example[:120])}...")

        written_this_run = 0
        chunk_num = 0
        max_empty_chunks = 5
        consecutive_empty = 0

        while written_this_run < remaining:
            chunk_num += 1
            needed = remaining - written_this_run
            batch_size = min(chunk_size, int(needed))

            # Sample a system prompt per-prompt and build the raw prefix.
            chunk_sys = [
                sample_system_prompt(sys_prompt_pool) if sys_prompt_pool else ""
                for _ in range(batch_size)
            ]
            prompts = [_get_prefix(sp) for sp in chunk_sys]

            log.info(f"  Chunk {chunk_num}: sending {batch_size} prompts...")
            outputs = llm.generate(prompts, sampling_params)

            chunk_items: list[dict] = []
            for i, out in enumerate(outputs):
                text = out.outputs[0].text
                messages: list[dict[str, str]] = []
                if chunk_sys[i]:
                    messages.append({"role": "system", "content": chunk_sys[i]})
                messages.append({"role": "user", "content": text})
                chunk_items.append({"messages": messages})
                if len(chunk_items) + written_this_run >= remaining:
                    break

            append_jsonl(raw_path, chunk_items)
            written_this_run += len(chunk_items)
            total = already_written + written_this_run
            log.info(f"  Flushed {len(chunk_items)} instructions (total raw: {total}/{n})")

            if len(chunk_items) == 0:
                consecutive_empty += 1
                if consecutive_empty >= max_empty_chunks:
                    log.error(f"  {max_empty_chunks} consecutive empty chunks. Aborting.")
                    break
            else:
                consecutive_empty = 0

    # --- LLM filter: separate instruction from leaked answer ---
    log.info(f"Loading raw instructions from {raw_path}...")
    raw_data = load_jsonl(raw_path, required_key="messages")

    def _user_content(entry: dict) -> str:
        for msg in entry["messages"]:
            if msg["role"] == "user":
                return msg["content"]
        return ""

    log.info(f"Raw instructions: {len(raw_data)}")

    if not raw_data:
        log.error("No raw instructions found. Aborting.")
        return

    if llm is None:
        tp = cfg.get("tensor_parallel_size", 1)
        log.info(f"Loading model for filtering: {model_name} (tp={tp})")
        llm = LLM(model=model_name, tensor_parallel_size=tp)

    filter_sp = SamplingParams(temperature=0.0, max_tokens=max_len)

    filter_chunk_size = icfg.get("chunk_size", 5000)
    filtered_data: list[dict] = []
    n_unchanged = 0
    n_cleaned = 0
    n_dropped = 0

    log.info(f"Filtering {len(raw_data)} instructions via LLM (separating instruction from answer)...")
    for chunk_start in range(0, len(raw_data), filter_chunk_size):
        chunk_end = min(chunk_start + filter_chunk_size, len(raw_data))
        chunk = raw_data[chunk_start:chunk_end]

        conversations = [
            [{"role": "user", "content": FILTER_PROMPT.format(text=_user_content(e))}]
            for e in chunk
        ]
        outputs = llm.chat(conversations, filter_sp)

        for i, out in enumerate(outputs):
            original = _user_content(chunk[i])
            cleaned = out.outputs[0].text.strip()

            if not (min_len <= len(cleaned) <= max_len):
                n_dropped += 1
                continue
            if normalize_text(cleaned) == normalize_text(original):
                n_unchanged += 1
                filtered_data.append(chunk[i])
            else:
                n_cleaned += 1
                messages = [m for m in chunk[i]["messages"] if m["role"] == "system"]
                messages.append({"role": "user", "content": cleaned})
                filtered_data.append({"messages": messages})

        log.info(f"  Filtered {chunk_end}/{len(raw_data)} "
                 f"(kept {len(filtered_data)} so far)")

    log.info(f"LLM filter results: {n_unchanged} unchanged, {n_cleaned} cleaned, {n_dropped} dropped")
    raw_data = filtered_data

    # --- Dedupe (on user-turn content) ---

    log.info("Exact deduplication...")
    seen: set[str] = set()
    exact_deduped: list[dict] = []
    for entry in raw_data:
        key = normalize_text(_user_content(entry))
        if key not in seen:
            seen.add(key)
            exact_deduped.append(entry)
    log.info(f"After exact dedupe: {len(exact_deduped)}")

    log.info(f"MinHash near-duplicate removal (threshold={mh_threshold})...")
    deduped_texts = [_user_content(d) for d in exact_deduped]
    keep_indices = dedupe_minhash_indices(deduped_texts, threshold=mh_threshold, num_perm=mh_num_perm)
    final_entries = [exact_deduped[i] for i in keep_indices]
    log.info(f"After MinHash dedupe: {len(final_entries)}")

    log.info(f"Writing {len(final_entries)} instructions to {output_path}")
    atomic_write_jsonl(output_path, final_entries)
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate instructions (MAGPIE/Replay)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None, help="Override output path")
    parser.add_argument("--quick", action="store_true", help="Quick mode: fewer instructions")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()
    generate_instructions(args.config, output_override=args.output, quick=args.quick, resume=args.resume)


if __name__ == "__main__":
    main()
