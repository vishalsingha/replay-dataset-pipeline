Benchmark Evaluation
====================

The ``eval/`` module evaluates any HuggingFace model on standard open
benchmarks using `lm-evaluation-harness <https://github.com/EleutherAI/lm-evaluation-harness>`_
with a vLLM backend.

Benchmarks
----------

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

Benchmark Suites
----------------

Three pre-defined suites with increasing coverage:

- **quick** (~30 min): GSM8K, IFEval, ARC-Challenge
- **standard** (~2-3 hours): GSM8K, MATH, HumanEval, MBPP, IFEval, MMLU, ARC-Challenge, HellaSwag
- **full** (~4-6 hours): All benchmarks including MMLU-Pro

Usage
-----

Run an evaluation:

.. code-block:: bash

   # Quick eval on default model (from eval/eval_config.yaml)
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --suite quick

   # Standard suite on a fine-tuned model
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --model /path/to/finetuned --suite standard

   # Full suite with explicit output filename
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --model /path/to/model --suite full \
       --output-file my_model.json

   # Run specific tasks only
   CUDA_VISIBLE_DEVICES=0 python -m eval.run_eval --tasks gsm8k,humaneval,ifeval

Compare two models:

.. code-block:: bash

   python -m eval.compare eval/results/base.json eval/results/finetuned.json

   # Save comparison to JSON
   python -m eval.compare eval/results/base.json eval/results/finetuned.json \
       --output eval/results/comparison.json

Batch evaluate all models:

.. code-block:: bash

   ./run_all_evals.sh

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

CLI Reference
-------------

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
     - Suite: ``quick``, ``standard``, ``full``
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

Notes
-----

- **HumanEval/MBPP** require code execution. Set ``HF_ALLOW_CODE_EVAL=1``
  environment variable (done automatically in ``run_all_evals.sh``).
- **MATH** uses the ``minerva_math`` variant in lm-eval which handles LaTeX
  answer extraction.
- **IFEval** is natively supported in lm-eval-harness v0.4+.
- Results are deterministic with seed control via vLLM sampling params.
- Results are saved as JSON in ``eval/results/`` (gitignored).
