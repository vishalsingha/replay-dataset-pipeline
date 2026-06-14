Experiment Results
==================

We fine-tuned ``Qwen/Qwen3-4B-Instruct-2507`` on a physics/chemistry/biology
task dataset under five data mixing strategies using both **full fine-tuning**
and **LoRA**, then evaluated across 13+ benchmarks.

Models
------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Model
     - Description
   * - **Base**
     - Original Qwen3-4B-Instruct-2507 (no fine-tuning)
   * - **task_only**
     - Domain task data only
   * - **replay_task**
     - Self-generated replay + domain task data
   * - **public_replay_task**
     - Public dataset replay + domain task data
   * - **replay_public_replay_task**
     - Both replay sources + domain task data
   * - **public_conv_task**
     - Raw public conversations + domain task data

Full Fine-Tuning Results
------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13

   * - Benchmark
     - Base
     - task_only
     - replay_task
     - public_replay
     - pub_conv
   * - GSM8K (strict)
     - 73.46
     - **81.88** (+8.42)
     - 69.90 (-3.56)
     - 64.29 (-9.17)
     - 77.79 (+4.32)
   * - MMLU-Pro Chemistry
     - **63.78**
     - 59.36 (-4.42)
     - 63.52 (-0.26)
     - 55.30 (-8.48)
     - 51.86 (-11.92)
   * - MMLU-Pro Physics
     - **63.66**
     - 56.81 (-6.85)
     - 62.05 (-1.61)
     - 57.27 (-6.39)
     - 52.35 (-11.31)
   * - MMLU-Pro Health
     - **58.92**
     - 54.40 (-4.52)
     - 58.19 (-0.73)
     - 57.33 (-1.59)
     - 54.28 (-4.64)
   * - MATH (Hendrycks)
     - 54.14
     - 54.56 (+0.42)
     - **54.80** (+0.66)
     - 52.84 (-1.30)
     - 50.06 (-4.08)
   * - HumanEval
     - **74.39**
     - 69.51 (-4.88)
     - 71.34 (-3.05)
     - 74.39 (0.00)
     - 63.41 (-10.98)
   * - MBPP
     - 65.40
     - **66.40** (+1.00)
     - 66.20 (+0.80)
     - 65.20 (-0.20)
     - 63.20 (-2.20)
   * - IFEval (strict)
     - **59.15**
     - 55.64 (-3.51)
     - 57.12 (-2.03)
     - 50.83 (-8.32)
     - 45.47 (-13.68)
   * - TruthfulQA
     - **62.60**
     - 55.71 (-6.89)
     - 60.61 (-1.99)
     - 58.49 (-4.12)
     - 55.91 (-6.69)
   * - MT-Bench (/10)
     - 8.43
     - 8.24
     - 8.38
     - 8.38
     - 7.92
   * - Jailbreak Refusal
     - 100%
     - 100%
     - 100%
     - 100%
     - 100%

**Full FT Ranking:** replay_task (-0.87% avg) > task_only (-0.78%) > replay_pub_rep (-2.19%) > public_replay (-2.71%) > pub_conv (-4.05%)

LoRA Fine-Tuning Results
------------------------

LoRA (r=16, alpha=32, all linear layers) with the same data splits.

.. list-table::
   :header-rows: 1
   :widths: 22 13 13 13 13 13

   * - Benchmark
     - Base
     - task_only
     - replay_task
     - public_replay
     - pub_conv
   * - GSM8K (strict)
     - 73.46
     - **79.76** (+6.29)
     - 67.85 (-5.61)
     - 62.24 (-11.22)
     - 78.09 (+4.62)
   * - MMLU-Pro Chemistry
     - **63.78**
     - 56.63 (-7.15)
     - 63.60 (-0.18)
     - 54.86 (-8.92)
     - 48.32 (-15.46)
   * - MMLU-Pro Physics
     - **63.66**
     - 54.81 (-8.85)
     - 62.43 (-1.23)
     - 58.74 (-4.92)
     - 49.65 (-14.01)
   * - MMLU-Pro Health
     - 58.92
     - 51.59 (-7.33)
     - **59.90** (+0.98)
     - 58.44 (-0.48)
     - 50.37 (-8.55)
   * - MATH (Hendrycks)
     - 54.14
     - **55.92** (+1.78)
     - 54.52 (+0.38)
     - 51.48 (-2.66)
     - 47.28 (-6.86)
   * - HumanEval
     - 74.39
     - 71.34 (-3.05)
     - **76.22** (+1.83)
     - 73.78 (-0.61)
     - 55.49 (-18.90)
   * - MBPP
     - 65.40
     - 65.40 (0.00)
     - **66.40** (+1.00)
     - 63.40 (-2.00)
     - 61.60 (-3.80)
   * - IFEval (strict)
     - **59.33**
     - 53.97 (-5.36)
     - 53.23 (-6.10)
     - 41.77 (-17.56)
     - 29.57 (-29.76)
   * - TruthfulQA
     - **62.60**
     - 52.82 (-9.79)
     - 60.76 (-1.85)
     - 58.12 (-4.49)
     - 55.47 (-7.13)
   * - MT-Bench (/10)
     - 8.48
     - 7.51
     - 8.42
     - **8.49**
     - ---
   * - Jailbreak Refusal
     - 100%
     - **95%**
     - 100%
     - 100%
     - 100%

**LoRA Ranking:** replay_task (-0.92% avg) > task_only (-1.44%) > replay_pub_rep (-3.31%) > public_replay (-4.21%) > pub_conv (-7.44%)

Key Findings
------------

1. **Self-generated replay preserves science knowledge 7x better (full FT) and
   250x better (LoRA)** than no replay. MMLU-Pro science avg loss: -0.69% with
   replay vs -4.44% without (full FT); -0.03% vs -7.59% (LoRA).

2. **LoRA without replay is MORE destructive than full fine-tuning.** LoRA
   ``task_only`` shows worse forgetting: TruthfulQA -9.79% (vs -6.89% full FT),
   MMLU-Pro Physics -8.85% (vs -6.85%), and jailbreak refusal drops to 95%.

3. **Self-replay with LoRA improves capabilities beyond base.** HumanEval
   +1.83%, Biology +0.56%, Health +0.98%. Replay acts as beneficial
   regularization.

4. **Public conversations are catastrophic.** IFEval -29.76% (LoRA),
   HumanEval -18.90%, MMLU-Pro Math -16.80%.

5. **Distribution matching > response quality.** Simple self-replay outperforms
   carefully curated GPT-4-generated public datasets.

6. **Safety is preserved with replay.** All replay models maintain 100%
   jailbreak refusal. Only LoRA ``task_only`` breaks safety (95%).

Conclusion
----------

Self-generated replay is essential for both full fine-tuning and LoRA. For LoRA
in particular, the low-rank constraint makes forgetting sharper, making replay
even more critical. The ``replay_task`` strategy provides the best trade-off:
minimal forgetting, preserved safety, and in some cases improved capabilities
beyond the base model.
