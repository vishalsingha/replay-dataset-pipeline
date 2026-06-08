Benchmark Evaluation
====================

The ``eval/`` module evaluates any HuggingFace model on standard open
benchmarks using `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
with a vLLM backend, plus custom evaluation scripts for LLM-as-judge and
safety benchmarks.

Standard Benchmarks (lm-eval-harness)
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 25 10

   * - Category
     - Benchmark
     - Metric
     - Shots
   * - Math
     - GSM8K
     - exact_match (strict)
     - 5-shot CoT
   * - Math
     - MATH (Hendrycks)
     - exact_match / math_verify
     - 4-shot CoT
   * - Code
     - HumanEval
     - pass@1
     - 0-shot
   * - Code
     - MBPP
     - pass@1
     - 3-shot
   * - Instruction Following
     - IFEval
     - prompt-level strict acc
     - 0-shot
   * - Knowledge
     - MMLU
     - accuracy
     - 5-shot
   * - Knowledge
     - MMLU-Pro
     - exact_match
     - 5-shot
   * - Reasoning
     - ARC-Challenge
     - acc_norm
     - 25-shot
   * - Reasoning
     - HellaSwag
     - acc_norm
     - 10-shot
   * - Truthfulness
     - TruthfulQA (MC2)
     - accuracy
     - 0-shot
   * - Safety
     - ToxiGen
     - accuracy
     - 0-shot

Custom Benchmarks (separate scripts)
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 30 20

   * - Benchmark
     - Script
     - What It Measures
     - Judge
   * - MT-Bench
     - ``eval.mt_bench``
     - Multi-turn conversation quality (1-10)
     - Azure OpenAI / GPT-4
   * - AlpacaEval 2.0
     - ``eval.alpaca_eval_run``
     - Instruction-following win-rate
     - Azure OpenAI / GPT-4
   * - LiveCodeBench
     - ``eval.livecodebench_run``
     - Contamination-free code generation
     - Test execution
   * - DS-1000
     - ``eval.ds1000_run``
     - Data science code (7 Python libs)
     - Unit tests
   * - HaluEval
     - ``eval.safety_bench``
     - Hallucination detection
     - Self-evaluation
   * - JailbreakBench
     - ``eval.safety_bench``
     - Jailbreak resistance / refusal rate
     - Pattern matching

Benchmark Suites
----------------

Four pre-defined suites:

- **quick** (~30 min): GSM8K, IFEval, ARC-Challenge
- **standard** (~2-3 hours): GSM8K, MATH, HumanEval, MBPP, IFEval, MMLU, ARC-Challenge, HellaSwag
- **full** (~4-6 hours): All standard + MMLU-Pro, TruthfulQA, ToxiGen
- **safety**: TruthfulQA, ToxiGen, JailbreakBench, HaluEval

Usage
-----

**Standard benchmarks (lm-eval-harness):**

.. code-block:: bash

   # Quick eval on default model
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --suite quick

   # Full suite on a fine-tuned model
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --model /path/to/finetuned --suite full

   # Safety suite
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --suite safety

   # Run specific tasks only
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --tasks gsm8k,truthfulqa,toxigen

**Custom benchmarks (LLM-as-judge):**

.. code-block:: bash

   # MT-Bench (uses Azure OpenAI as judge)
   python -m eval.mt_bench --model Qwen/Qwen3-4B-Instruct-2507

   # AlpacaEval 2.0
   python -m eval.alpaca_eval_run --model Qwen/Qwen3-4B-Instruct-2507

   # Generate only (no API key needed)
   python -m eval.mt_bench --model /path/to/model --generate-only

**Code benchmarks:**

.. code-block:: bash

   # LiveCodeBench
   python -m eval.livecodebench_run --model Qwen/Qwen3-4B-Instruct-2507

   # DS-1000
   python -m eval.ds1000_run --model Qwen/Qwen3-4B-Instruct-2507

**Safety benchmarks:**

.. code-block:: bash

   # HaluEval + JailbreakBench
   python -m eval.safety_bench --model Qwen/Qwen3-4B-Instruct-2507 --bench all

   # Just jailbreak testing
   python -m eval.safety_bench --model /path/to/finetuned --bench jailbreakbench

**Compare models:**

.. code-block:: bash

   python -m eval.compare eval/results/base.json eval/results/finetuned.json

**Batch evaluate all models:**

.. code-block:: bash

   ./run_all_evals.sh

Azure OpenAI Configuration
---------------------------

For benchmarks that use LLM-as-judge (MT-Bench, AlpacaEval), create a
``.env`` file at the project root (already gitignored):

.. code-block:: bash

   # .env
   AZURE_OPENAI_API_KEY=your-key-here
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_API_VERSION=2024-12-01-preview
   AZURE_OPENAI_DEPLOYMENT=gpt4omini

   # These allow the OpenAI SDK to route to Azure automatically
   OPENAI_API_KEY=your-key-here
   OPENAI_API_TYPE=azure
   OPENAI_API_VERSION=2024-12-01-preview
   OPENAI_API_BASE=https://your-endpoint.openai.azure.com/

The eval scripts auto-detect Azure when ``AZURE_OPENAI_API_KEY`` or
``OPENAI_API_TYPE=azure`` is set. The ``run_all_evals.sh`` script
automatically sources ``.env`` if present.

For standard OpenAI (non-Azure), just set:

.. code-block:: bash

   OPENAI_API_KEY=sk-your-key-here

Configuration
-------------

Evaluation settings are in ``eval/eval_config.yaml``:

.. code-block:: yaml

   model_path: Qwen/Qwen3-4B-Instruct-2507
   backend: vllm
   tensor_parallel_size: 1
   gpu_memory_utilization: 0.9
   max_model_len: 4096
   seed: 42

   suites:
     quick:
       - gsm8k
       - ifeval
       - arc_challenge
     standard:
       - gsm8k
       - math
       - humaneval
       - mbpp
       - ifeval
       - mmlu
       - arc_challenge
       - hellaswag
     full:
       - gsm8k
       - math
       - humaneval
       - mbpp
       - ifeval
       - mmlu
       - mmlu_pro
       - arc_challenge
       - hellaswag
       - truthfulqa
       - toxigen
     safety:
       - truthfulqa
       - toxigen
       - jailbreakbench
       - halueval

CLI Reference
-------------

**Standard eval (lm-eval-harness):**

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Script
     - Flag
     - Description
   * - ``eval.run_eval``
     - ``--config PATH``
     - Eval config YAML (default: ``eval/eval_config.yaml``)
   * - ``eval.run_eval``
     - ``--model PATH``
     - HuggingFace model ID or local path
   * - ``eval.run_eval``
     - ``--suite NAME``
     - Suite: ``quick``, ``standard``, ``full``, ``safety``
   * - ``eval.run_eval``
     - ``--tasks LIST``
     - Comma-separated tasks (overrides ``--suite``)
   * - ``eval.run_eval``
     - ``--output-file NAME``
     - Explicit output filename
   * - ``eval.run_eval``
     - ``--output-dir DIR``
     - Output directory (default: ``eval/results``)
   * - ``eval.compare``
     - positional
     - Two result JSON paths (supports globs)
   * - ``eval.compare``
     - ``--output PATH``
     - Save comparison JSON

**Custom benchmarks:**

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Script
     - Flag
     - Description
   * - ``eval.mt_bench``
     - ``--model PATH``
     - Model to evaluate (required)
   * - ``eval.mt_bench``
     - ``--judge-model NAME``
     - Judge model (default: gpt-4o, auto-uses Azure deployment if configured)
   * - ``eval.mt_bench``
     - ``--generate-only``
     - Only generate answers, skip judging
   * - ``eval.alpaca_eval_run``
     - ``--model PATH``
     - Model to evaluate (required)
   * - ``eval.alpaca_eval_run``
     - ``--generate-only``
     - Only generate outputs, skip judging
   * - ``eval.livecodebench_run``
     - ``--model PATH``
     - Model to evaluate (required)
   * - ``eval.livecodebench_run``
     - ``--release-version VER``
     - Dataset version (default: release_latest)
   * - ``eval.ds1000_run``
     - ``--model PATH``
     - Model to evaluate (required)
   * - ``eval.safety_bench``
     - ``--model PATH``
     - Model to evaluate (required)
   * - ``eval.safety_bench``
     - ``--bench NAME``
     - halueval, jailbreakbench, or all

Notes
-----

- **HumanEval/MBPP** require code execution. Set ``HF_ALLOW_CODE_EVAL=1``
  environment variable (done automatically in ``run_all_evals.sh``).
- **MATH** uses the ``minerva_math`` variant in lm-eval which handles LaTeX
  answer extraction.
- **IFEval** is natively supported in lm-eval-harness v0.4+.
- **MT-Bench/AlpacaEval** require Azure OpenAI or OpenAI API key for judging.
  Use ``--generate-only`` to skip judging if no API key is available.
- **LiveCodeBench** requires cloning the repo separately.
- Results are deterministic with seed control via vLLM sampling params.
- Results are saved as JSON in ``eval/results/`` (gitignored).
