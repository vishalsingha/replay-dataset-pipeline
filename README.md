# Replay Dataset Pipeline

Replay (rehearsal) dataset generation for mitigating catastrophic forgetting during SFT.

Instead of mixing public SFT datasets (which are distributionally mismatched), this pipeline **reconstructs the base model's own latent instruction distribution** and synthesizes high-quality responses from it. It also supports pulling instructions from public datasets as a complementary source.

**Target model:** `Qwen/Qwen3-4B-Instruct-2507`

**Documentation:** https://vishalsingha.github.io/replay-dataset-pipeline/

---

## Table of Contents

- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration Reference](#configuration-reference)
- [Pipeline Walkthrough](#pipeline-walkthrough)
- [Checkpointing and Resume](#checkpointing-and-resume)
- [Common Recipes](#common-recipes)
- [Output Format](#output-format)
- [Expanding the Committee](#expanding-the-committee)
- [CLI Reference](#cli-reference)
- [Troubleshooting](#troubleshooting)
- [Benchmark Evaluation](#benchmark-evaluation)
- [Experiment Results](#experiment-results)
- [Reference](#reference)

---

## How It Works

The pipeline has two independent instruction sources (Track A and Track B) that feed into a shared response generation and filtering pipeline:

```
Track A (self-generated)              Track B (public datasets)
─────────────────────────             ─────────────────────────
Step 1: generate_instructions         pull_public_instructions
  │  (pre-query template trick)         │  (HuggingFace datasets)
  ▼                                     ▼
instructions.jsonl                    public_instructions/instructions.jsonl
  │                                     │
  ├──► Step 1b: generate_multiturn      │
  │      │                              │
  │      ▼                              │
  │    multiturn.jsonl                  │
  │                                     │
  ▼                                     ▼
Step 2: generate_responses ◄──── (same script, --instructions flag)
  │                                     │
  ▼                                     ▼
candidates.jsonl              public_instructions/candidates.jsonl
  │                                     │
  ▼                                     ▼
Step 3: filter_responses ◄──── (same script, --candidates flag)
  │                                     │
  ▼                                     ▼
replay.jsonl                  public_instructions/replay.jsonl
  │                                     │
  └──────────────┬──────────────────────┘
                 ▼
Step 4: mix_datasets
  │  (auto-loads all replay sources + domain data)
  ▼
train.jsonl
```

**Step 1 — Instruction Generation:** Feed the model only its user-turn template prefix and let it generate instructions from its own latent distribution. Deduplicate via exact + MinHash near-dup removal.

**Step 1b — Multi-Turn Generation:** A configurable fraction of instructions are extended into multi-turn conversations (2-5 turns) by alternating model-generated follow-up user messages and assistant responses.

**Public Instructions:** Pull prompts only (discard original responses) from 5 public datasets covering instruction-following, chat, reasoning, code, and math.

**Step 2 — Multi-Response Generation:** A committee of generators (default: the same model) produces L candidate responses per instruction, with a system prompt sampled from a diverse generic pool.

**Step 3 — Filtering:** A committee of judges scores each candidate on a 5-point rubric (helpfulness, relevance, clarity, AI-persona). The best-scoring response is selected per instruction. A detailed summary of parse failures and score distributions is printed at the end.

**Step 4 — Mixing:** All replay sources (single-turn + public + multi-turn) are **automatically loaded** and combined with your domain SFT data at a configurable ratio.

---

## Project Structure

```
replay/
├── config.yaml                          # All configuration (validated by pydantic at startup)
├── requirements.txt                     # Python dependencies
├── run.sh                               # Main pipeline script (public track)
├── run_all_evals.sh                     # Batch evaluate all fine-tuned models
├── README.md                            # This file
├── src/
│   ├── __init__.py
│   ├── config_schema.py                 # Pydantic config validation
│   ├── log.py                           # Structured logging (stdout + file)
│   ├── utils.py                         # Shared utilities (JSONL I/O, checkpointing, dedupe)
│   ├── backends.py                      # vLLM + OpenAI API abstraction
│   ├── prompts.py                       # Judge rubric for quality scoring
│   ├── generate_instructions.py         # Step 1: self-generate instructions
│   ├── pull_public_instructions.py      # Pull instructions from public datasets
│   ├── pull_public_conversations.py     # Pull full conversations from public datasets
│   ├── generate_multiturn.py            # Step 1b: multi-turn conversations
│   ├── generate_responses.py            # Step 2: generate candidate responses
│   ├── filter_responses.py              # Step 3: judge + filter
│   └── mix_datasets.py                  # Step 4: mix replay + domain
├── eval/
│   ├── __init__.py
│   ├── eval_config.yaml                 # Benchmark suite definitions + model config
│   ├── run_eval.py                      # Evaluation runner (lm-eval-harness + vLLM)
│   ├── compare.py                       # Compare results across models
│   └── results/                         # JSON output from evaluation runs (gitignored)
├── logs/                                # Timestamped log files (gitignored)
└── data/
    ├── replay/                          # Self-generated replay (Track A)
    ├── public/                          # Public dataset instructions + replay (Track B)
    ├── public_conversations/            # Full conversations from public datasets
    └── final/                           # Mixed training data (Step 4)
```

---

## Setup

```bash
cd /path/to/replay

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, GPU(s) with enough VRAM for Qwen3-4B-Instruct-2507, pydantic for config validation.

---

## Configuration Reference

All settings live in `config.yaml`. The config is **validated at startup** by pydantic — typos, missing fields, and invalid ranges produce clear error messages immediately, not cryptic KeyErrors mid-run.

### Top-level

| Field | Description | Default |
|---|---|---|
| `model` | HuggingFace model ID | `Qwen/Qwen3-4B-Instruct-2507` |
| `tensor_parallel_size` | Number of GPUs for vLLM tensor parallelism | `2` |
| `seed` | Global random seed for reproducibility across all steps | `42` |
| `log_level` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |

### `dedupe`

| Field | Description | Default |
|---|---|---|
| `minhash_threshold` | MinHash LSH similarity threshold for near-dup removal (0-1) | `0.7` |
| `minhash_num_perm` | Number of permutations for MinHash | `128` |

### `stop_tokens`

Model-specific stop tokens used in instruction generation and multi-turn follow-up generation. Change these when switching to a non-Qwen model.

Default: `["<|im_end|>", "<|endoftext|>", "<|im_start|>assistant"]`

### `instruction_generation` (Step 1)

| Field | Description | Default |
|---|---|---|
| `n` | Target number of raw instructions to write (before dedupe) | `20000` |
| `quick_n` | N used when `--quick` flag is passed | `1000` |
| `chunk_size` | Instructions per checkpoint flush | `5000` |
| `sampling.temperature` | Higher = more diverse instructions | `1.0` |
| `sampling.top_p` | Nucleus sampling threshold | `0.98` |
| `sampling.top_k` | Top-k sampling | `50` |
| `sampling.max_tokens` | Max tokens per instruction | `512` |
| `min_length` | Drop instructions shorter than this (chars) | `10` |
| `max_length` | Drop instructions longer than this (chars) | `2048` |

### `response_generation` (Step 2)

| Field | Description | Default |
|---|---|---|
| `L` | Candidate responses per generator per instruction | `3` |
| `chunk_size` | Instructions per checkpoint flush | `1000` |
| `sampling.temperature` | Official Qwen recommended | `0.7` |
| `sampling.top_p` | Official Qwen recommended | `0.8` |
| `sampling.top_k` | Official Qwen recommended | `20` |
| `sampling.min_p` | Official Qwen recommended | `0.0` |
| `sampling.max_tokens` | Maximum response length | `16384` |
| `sampling.presence_penalty` | Reduce repetition (0-2, higher risks language mixing) | `0.5` |

### `filtering` (Step 3)

| Field | Description | Default |
|---|---|---|
| `chunk_size` | Candidates per checkpoint flush | `1000` |
| `min_score` | Drop instructions whose best candidate scores below this (1-5) | `3.0` |
| `sampling.temperature` | Lower = more deterministic judge | `0.3` |
| `sampling.max_tokens` | For rubric justification + score | `512` |

### `committee`

| Field | Description |
|---|---|
| `generators[].type` | `vllm_local` or `openai_api` |
| `generators[].model` | Model name / HuggingFace ID |
| `judges[].type` | `vllm_local` or `openai_api` |
| `judges[].model` | Model name / HuggingFace ID |

**Note:** Multi-turn generation (Step 1b) requires the first generator to be `vllm_local` (needs tokenizer + raw text generation). API backends work for Steps 2-3 only.

### `replay_system_prompts`

A weighted pool of system prompts. For each instruction, one is sampled by weight during response generation. Your domain persona (e.g. PW Guru) is intentionally excluded to keep the replay persona-neutral and the final model easily re-steerable. Total weight must be > 0.

| Field | Description |
|---|---|
| `prompt` | The system prompt text (empty string = no system prompt) |
| `weight` | Relative sampling weight (>= 0) |

### `public_instructions`

| Field | Description | Default |
|---|---|---|
| `min_length` | Drop instructions shorter than this (chars) | `10` |
| `max_length` | Drop instructions longer than this (chars) | `2048` |
| `sources[].dataset` | HuggingFace dataset ID | -- |
| `sources[].split` | Dataset split to use | `train` |
| `sources[].n` | Number of instructions to sample from this source | `5000` |
| `sources[].streaming` | Stream instead of full download | `true` |

### `public_conversations`

Pulls complete conversations (system + instruction + response) from public datasets, ready for SFT without a separate response generation step.

| Field | Description | Default |
|---|---|---|
| `min_length` | Drop instructions shorter than this (chars) | `10` |
| `max_length` | Drop instructions longer than this (chars) | `2048` |
| `min_response_length` | Drop responses shorter than this (chars) | `10` |
| `sources[].dataset` | HuggingFace dataset ID | -- |
| `sources[].split` | Dataset split to use | `train` |
| `sources[].n` | Number of conversations to sample from this source | `5000` |
| `sources[].streaming` | Stream instead of full download | `true` |

### `multiturn` (Step 1b)

| Field | Description | Default |
|---|---|---|
| `fraction` | Fraction of instructions to extend to multi-turn (0 = skip) | `0.3` |
| `min_turns` | Minimum user-assistant turn pairs | `2` |
| `max_turns` | Maximum user-assistant turn pairs | `5` |
| `chunk_size` | Conversations per checkpoint flush | `500` |
| `followup_sampling.temperature` | Higher diversity for follow-up questions | `0.9` |
| `followup_sampling.max_tokens` | Follow-up questions are short | `512` |

### `mixing` (Step 4)

| Field | Description | Default |
|---|---|---|
| `domain_fraction` | Fraction of domain data in final mix (0-1 exclusive) | `0.17` |
| `domain_data_path` | Path to your domain SFT dataset (JSONL with `messages` key) | **must set** |
| `val_fraction` | Validation split fraction (0 = no split) | `0.0` |
| `seed` | Random seed for sampling/shuffling | `42` |

### `paths`

| Field | Default path |
|---|---|
| `instructions` | `data/instructions/instructions.jsonl` |
| `public_instructions` | `data/public_instructions/instructions.jsonl` |
| `candidates` | `data/candidates/candidates.jsonl` |
| `replay` | `data/replay/replay.jsonl` |
| `public_replay` | `data/public_instructions/replay.jsonl` |
| `multiturn` | `data/multiturn/multiturn.jsonl` |
| `train` | `data/final/train.jsonl` |
| `val` | `data/final/val.jsonl` |

---

## Pipeline Walkthrough

All commands assume you are in the project root with the venv activated:

```bash
cd /path/to/replay
source venv/bin/activate
```

### Track A: Self-Generated Instructions

```bash
# Step 1: Generate instructions (full run or --quick for testing)
python -m src.generate_instructions --config config.yaml

# Step 1b: Multi-turn (optional, can run in parallel with Step 2)
python -m src.generate_multiturn --config config.yaml

# Step 2: Generate candidate responses
python -m src.generate_responses --config config.yaml

# Step 3: Filter and select best responses
python -m src.filter_responses --config config.yaml
```

### Track B: Public Dataset Instructions

```bash
# Pull instructions from HuggingFace datasets
python -m src.pull_public_instructions --config config.yaml

# Generate + filter responses (use --instructions and --output overrides)
python -m src.generate_responses --config config.yaml \
    --instructions data/public_instructions/instructions.jsonl \
    --output data/public_instructions/candidates.jsonl

python -m src.filter_responses --config config.yaml \
    --candidates data/public_instructions/candidates.jsonl \
    --output data/public_instructions/replay.jsonl
```

### Track C: Public Conversations (No Generation Needed)

Pulls complete conversations (system + user + assistant) directly from public datasets. Since responses are included, no generate_responses or filter step is needed — data is ready for SFT immediately.

```bash
python -m src.pull_public_conversations --config config.yaml \
    --output data/public_conversations/conversations.jsonl
```

### Mixing

Set `mixing.domain_data_path` in config.yaml, then:

```bash
python -m src.mix_datasets --config config.yaml
```

The mixer **automatically loads** all replay sources:
- `data/replay/replay.jsonl` (self-generated single-turn)
- `data/public_instructions/replay.jsonl` (public-track single-turn, via `paths.public_replay`)
- `data/multiturn/multiturn.jsonl` (multi-turn)

Missing files are silently skipped (count as 0 examples).

---

## Checkpointing and Resume

Every script supports `--resume` to continue from the last checkpoint after a crash or interruption. No GPU work is lost.

```bash
# Crash at 60%? Resume from where it stopped:
python -m src.generate_instructions --config config.yaml --resume
python -m src.generate_responses --config config.yaml --resume
python -m src.filter_responses --config config.yaml --resume
python -m src.generate_multiturn --config config.yaml --resume
python -m src.pull_public_instructions --config config.yaml --resume
```

**How it works per script:**

| Script | Checkpoint mechanism | Resume behavior |
|---|---|---|
| `generate_instructions` | Raw instructions → `.raw.jsonl`, flushed per chunk. Dedupe at end. | Counts valid lines in `.raw.jsonl`, generates only remaining. |
| `generate_responses` | Candidates appended per chunk. | Sanitizes output (removes partial lines), skips processed count. |
| `filter_responses` | Results appended per chunk. Atomic `.progress` file tracks input index. | Reads `.progress` (corrupt-safe, defaults to 0), continues from there. |
| `generate_multiturn` | Conversations appended per chunk. Atomic `.progress` file. | Same as filter. |
| `pull_public_instructions` | Per-source progress in `.source_progress.json` tracking stream position. | Skips completed sources, resumes partial sources at correct stream offset. |

**Safety features:**
- `fsync` after every chunk flush
- Atomic writes (temp file + rename) for final outputs and progress files
- Partial/corrupt trailing lines in JSONL are detected and removed on resume
- Without `--resume`, scripts start fresh (truncate outputs)

---

## Common Recipes

### Recipe 1: Full Pipeline (Self-Generated Only)

```bash
python -m src.generate_instructions --config config.yaml
python -m src.generate_multiturn --config config.yaml
python -m src.generate_responses --config config.yaml
python -m src.filter_responses --config config.yaml
python -m src.mix_datasets --config config.yaml
```

### Recipe 2: Full Pipeline (Public Only)

```bash
python -m src.pull_public_instructions --config config.yaml
python -m src.generate_responses --config config.yaml \
    --instructions data/public_instructions/instructions.jsonl \
    --output data/public_instructions/candidates.jsonl
python -m src.filter_responses --config config.yaml \
    --candidates data/public_instructions/candidates.jsonl \
    --output data/public_instructions/replay.jsonl
python -m src.mix_datasets --config config.yaml
```

### Recipe 3: Both Tracks + Multi-Turn (maximum diversity)

```bash
# Track A
python -m src.generate_instructions --config config.yaml
python -m src.generate_multiturn --config config.yaml
python -m src.generate_responses --config config.yaml
python -m src.filter_responses --config config.yaml

# Track B
python -m src.pull_public_instructions --config config.yaml
python -m src.generate_responses --config config.yaml \
    --instructions data/public_instructions/instructions.jsonl \
    --output data/public_instructions/candidates.jsonl
python -m src.filter_responses --config config.yaml \
    --candidates data/public_instructions/candidates.jsonl \
    --output data/public_instructions/replay.jsonl

# Mix (auto-loads all three replay sources + domain)
python -m src.mix_datasets --config config.yaml
```

### Recipe 4: Quick Test Run

```bash
python -m src.generate_instructions --config config.yaml --quick
python -m src.generate_responses --config config.yaml
python -m src.filter_responses --config config.yaml
wc -l data/replay/replay.jsonl
head -1 data/replay/replay.jsonl | python -m json.tool
```

---

## Output Format

All replay data uses TRL / Unsloth chat-messages JSONL format.

**Single-turn with system prompt:**

```json
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain photosynthesis"}, {"role": "assistant", "content": "Photosynthesis is..."}]}
```

**Single-turn without system prompt** (empty string sampled from pool):

```json
{"messages": [{"role": "user", "content": "Solve x^2 = 4"}, {"role": "assistant", "content": "x = ±2..."}]}
```

**Multi-turn:**

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "What is recursion?"}, {"role": "assistant", "content": "Recursion is..."}, {"role": "user", "content": "Can you show a Python example?"}, {"role": "assistant", "content": "def factorial(n):..."}]}
```

---

## Expanding the Committee

By default, the same model is both generator and judge (single-model setup).

**Local models via vLLM:**

```yaml
committee:
  generators:
    - {type: vllm_local, model: Qwen/Qwen3-4B-Instruct-2507}
    - {type: vllm_local, model: Qwen/Qwen3-72B-Instruct}
  judges:
    - {type: vllm_local, model: Qwen/Qwen3-4B-Instruct-2507}
    - {type: vllm_local, model: Qwen/Qwen3-72B-Instruct}
```

**External API models** (Steps 2-3 only; multi-turn requires `vllm_local`):

```yaml
committee:
  generators:
    - {type: vllm_local, model: Qwen/Qwen3-4B-Instruct-2507}
    - {type: openai_api, model: gpt-4o}
  judges:
    - {type: vllm_local, model: Qwen/Qwen3-4B-Instruct-2507}
    - {type: openai_api, model: gpt-4o}
```

For `openai_api` backends, set environment variables or add per-entry config:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"  # optional
```

---

## CLI Reference

Every pipeline script accepts `--config` (defaults to `config.yaml`) and `--resume`.

| Script | Extra flags | Description |
|---|---|---|
| `src.generate_instructions` | `--quick` | Generate `quick_n` instructions (testing) |
| `src.generate_instructions` | `--resume` | Resume from last raw checkpoint |
| `src.generate_responses` | `--instructions PATH` | Override input instructions file |
| `src.generate_responses` | `--output PATH` | Override output candidates file |
| `src.generate_responses` | `--resume` | Resume from last checkpoint |
| `src.filter_responses` | `--candidates PATH` | Override input candidates file |
| `src.filter_responses` | `--output PATH` | Override output replay file |
| `src.filter_responses` | `--resume` | Resume from last progress checkpoint |
| `src.generate_multiturn` | `--resume` | Resume from last progress checkpoint |
| `src.pull_public_instructions` | `--output PATH` | Override output instructions file |
| `src.pull_public_instructions` | `--resume` | Resume from per-source stream position |
| `src.pull_public_conversations` | `--output PATH` | Override output conversations file |
| `src.pull_public_conversations` | `--resume` | Resume from per-source stream position |
| `src.mix_datasets` | -- | Reads all paths from config |

**Evaluation scripts:**

| Script | Flags | Description |
|---|---|---|
| `eval.run_eval` | `--config PATH` | Eval config (default: `eval/eval_config.yaml`) |
| `eval.run_eval` | `--model PATH` | HuggingFace model path (overrides config) |
| `eval.run_eval` | `--suite NAME` | Benchmark suite: `quick`, `standard`, `full` |
| `eval.run_eval` | `--tasks LIST` | Comma-separated task list (overrides `--suite`) |
| `eval.run_eval` | `--output-file NAME` | Explicit output filename |
| `eval.run_eval` | `--output-dir DIR` | Output directory (default: `eval/results`) |
| `eval.compare` | `result_a result_b` | Two result JSON paths (positional, supports globs) |
| `eval.compare` | `--output PATH` | Save comparison JSON to file |

---

## Troubleshooting

**Config validation error at startup**

The pipeline validates `config.yaml` with pydantic before any GPU work. Common issues:
- Typo in a key name (e.g. `commitee` instead of `committee`)
- `domain_fraction` outside (0, 1) range
- `min_turns` > `max_turns`
- Empty `generators` or `judges` list

**Low filter pass rate (e.g. 290 out of 43,000)**

Check the filtering summary printed at the end. Common causes:
- High `parse_failures` percentage → judge isn't following the rubric format. Lower `filtering.sampling.temperature` or try a larger judge model.
- All scores below `min_score` → lower `filtering.min_score` to `2.0`
- The sample parse failures show the raw judge output tail — check if the model is outputting `Score: N` at the end.

**Out of GPU memory**

- Reduce `chunk_size` in the relevant step (smaller vLLM batches)
- Reduce `response_generation.sampling.max_tokens`
- Use `tensor_parallel_size: 1` if you only have one GPU

**vLLM not using both GPUs**

- Ensure `tensor_parallel_size: 2` is set in config.yaml
- Check `nvidia-smi` — both GPUs must be visible
- vLLM requires NCCL

**Crash during generation — how to recover**

Just add `--resume` to the same command. All scripts checkpoint progress to disk and skip already-completed work.

**Public dataset download fails**

- Some datasets require `huggingface-cli login` for gated access
- `streaming: true` avoids downloading the full dataset
- The pipeline has a 10M iteration safety cap per source to prevent infinite loops

**Empty instructions from Step 1**

- The pipeline aborts after 5 consecutive zero-yield chunks with a clear error
- Try lowering `instruction_generation.sampling.temperature` to `0.8`
- Check that `stop_tokens` match your model's template
- Verify `min_length` < `max_length`

---

## Logging

All output uses structured logging with timestamps and levels. Logs are written to both stdout and timestamped files under `logs/`:

```
logs/
├── generate_instructions_20260604_065012.log
├── generate_responses_20260604_070115.log
├── filter_responses_20260604_073022.log
└── pull_public_instructions_20260604_080512.log
```

Example output:

```
[2026-06-04 06:50:12] INFO    Logging to logs/generate_instructions_20260604_065012.log
[2026-06-04 06:50:12] INFO    Loading model: Qwen/Qwen3-4B-Instruct-2507 (tensor_parallel_size=1)
[2026-06-04 06:50:15] INFO    Generating 5000 instructions in chunks of 5000...
[2026-06-04 06:51:02] WARNING malformed JSON at data/candidates/candidates.jsonl:4231, skipping
```

Set `log_level` in config.yaml to control verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

---

## Benchmark Evaluation

The `eval/` module evaluates any HuggingFace model on standard open benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) with a vLLM backend, plus custom scripts for LLM-as-judge and safety benchmarks.

### Standard Benchmarks (via lm-eval-harness)

| Category | Benchmark | Metric | Shots |
|----------|-----------|--------|-------|
| Math | GSM8K | exact_match (strict) | 5-shot CoT |
| Math | MATH (Hendrycks) | exact_match / math_verify | 4-shot CoT |
| Code | HumanEval | pass@1 | 0-shot |
| Code | MBPP | pass@1 | 3-shot |
| Instruction Following | IFEval | prompt-level strict acc | 0-shot |
| Knowledge | MMLU | accuracy | 5-shot |
| Knowledge | MMLU-Pro | exact_match | 5-shot |
| Reasoning | ARC-Challenge | acc_norm | 25-shot |
| Reasoning | HellaSwag | acc_norm | 10-shot |
| Truthfulness | TruthfulQA (MC2) | accuracy | 0-shot |
| Safety | ToxiGen | accuracy | 0-shot |

### Custom Benchmarks (separate scripts)

| Benchmark | Script | What It Measures | Judge |
|-----------|--------|-----------------|-------|
| MT-Bench | `eval.mt_bench` | Multi-turn conversation quality (1-10) | Azure OpenAI / GPT-4 |
| AlpacaEval 2.0 | `eval.alpaca_eval_run` | Instruction-following win-rate | Azure OpenAI / GPT-4 |
| LiveCodeBench | `eval.livecodebench_run` | Contamination-free code generation | Test execution |
| DS-1000 | `eval.ds1000_run` | Data science code (7 Python libs) | Unit tests |
| HaluEval | `eval.safety_bench` | Hallucination detection | Self-evaluation |
| JailbreakBench | `eval.safety_bench` | Jailbreak resistance / refusal rate | Pattern matching |

### Usage

```bash
# Standard benchmarks (lm-eval-harness)
python -m eval.run_eval --suite quick           # GSM8K, IFEval, ARC
python -m eval.run_eval --suite full            # All 11 benchmarks
python -m eval.run_eval --model /path/to/model --tasks gsm8k,truthfulqa,toxigen

# Custom benchmarks
python -m eval.mt_bench --model /path/to/model              # MT-Bench
python -m eval.safety_bench --model /path/to/model --bench all  # HaluEval + Jailbreak
python -m eval.alpaca_eval_run --model /path/to/model       # AlpacaEval 2.0

# Compare two models
python -m eval.compare eval/results/base.json eval/results/finetuned.json

# Batch evaluate all models
./run_all_evals.sh
```

### Azure OpenAI Configuration (for LLM-as-judge)

Create a `.env` file (already gitignored):

```bash
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT=gpt4omini
```

### LoRA Adapter Merging

Merge LoRA adapters with the base model to create standalone full models:

```bash
python merge_lora.py
```

This scans `/home/azureuser/nvme_0/replay_lora/*/saved_model/`, merges each adapter with the base model, and saves to `saved_full_model/` beside it.

---

## Experiment Results

We fine-tuned `Qwen/Qwen3-4B-Instruct-2507` on a physics/chemistry/biology task dataset under five data mixing strategies, evaluated with both full fine-tuning and LoRA.

| Model | Description |
|-------|-------------|
| **Base** | Original Qwen3-4B-Instruct-2507 (no fine-tuning) |
| **task_only** | Domain task data only |
| **replay_task** | Self-generated replay + domain task data |
| **public_replay_task** | Public dataset replay + domain task data |
| **replay_public_replay_task** | Both replay sources + domain task data |
| **public_conv_task** | Raw public conversations + domain task data |

### Full Fine-Tuning Results

| Benchmark | Base | task_only | replay_task | public_replay | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------:|
| **GSM8K** (strict) | 73.46 | **81.88** (+8.42) | 69.90 (-3.56) | 64.29 (-9.17) | 77.79 (+4.32) |
| **MMLU-Pro Chemistry** | **63.78** | 59.36 (-4.42) | 63.52 (-0.26) | 55.30 (-8.48) | 51.86 (-11.92) |
| **MMLU-Pro Physics** | **63.66** | 56.81 (-6.85) | 62.05 (-1.61) | 57.27 (-6.39) | 52.35 (-11.31) |
| **HumanEval** | **74.39** | 69.51 (-4.88) | 71.34 (-3.05) | 74.39 (0.00) | 63.41 (-10.98) |
| **IFEval** | **59.15** | 55.64 (-3.51) | 57.12 (-2.03) | 50.83 (-8.32) | 45.47 (-13.68) |
| **TruthfulQA** | **62.60** | 55.71 (-6.89) | 60.61 (-1.99) | 58.49 (-4.12) | 55.91 (-6.69) |
| **MT-Bench** (/10) | 8.43 | 8.24 | 8.38 | 8.38 | 7.92 |
| **Jailbreak Refusal** | 100% | 100% | 100% | 100% | 100% |

### LoRA Fine-Tuning Results

| Benchmark | Base | task_only | replay_task | public_replay | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------:|
| **MMLU-Pro Chemistry** | **63.78** | 56.63 (-7.15) | 63.60 (-0.18) | 54.86 (-8.92) | 48.32 (-15.46) |
| **MMLU-Pro Physics** | **63.66** | 54.81 (-8.85) | 62.43 (-1.23) | 58.74 (-4.92) | 49.65 (-14.01) |
| **HumanEval** | 74.39 | 71.34 (-3.05) | **76.22** (+1.83) | 73.78 (-0.61) | 55.49 (-18.90) |
| **IFEval** | **59.33** | 53.97 (-5.36) | 53.23 (-6.10) | 41.77 (-17.56) | 29.57 (-29.76) |
| **TruthfulQA** | **62.60** | 52.82 (-9.79) | 60.76 (-1.85) | 58.12 (-4.49) | 55.47 (-7.13) |
| **MT-Bench** (/10) | 8.48 | 7.51 | 8.42 | **8.49** | — |
| **Jailbreak Refusal** | 100% | **95%** | 100% | 100% | 100% |

### Key Findings

1. **Self-generated replay preserves science knowledge 7x better (full FT) and 250x better (LoRA)** than no replay.

2. **LoRA without replay is MORE destructive** than full fine-tuning — TruthfulQA drops 9.79%, jailbreak refusal breaks to 95%.

3. **Self-replay with LoRA improves capabilities beyond base** — HumanEval +1.83%, Biology +0.56%.

4. **Public conversations are catastrophic** — IFEval -29.76% (LoRA), HumanEval -18.90%.

5. **Distribution matching > response quality** — simple self-replay beats GPT-4-generated public data.

### Conclusion

Self-generated replay is essential for both full fine-tuning and LoRA. It preserves domain knowledge, prevents safety degradation, and can even improve capabilities beyond the base model.

