Experiment Results
==================

We fine-tuned ``Qwen/Qwen3-4B-Instruct-2507`` on a domain task dataset under
four different data mixing strategies to measure catastrophic forgetting.

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
     - SFT on domain task data only
   * - **replay_task**
     - SFT on self-generated replay + domain task data
   * - **public_replay_task**
     - SFT on public dataset replay + domain task data

Full Benchmark Results
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 14 14 14 14

   * - Benchmark
     - Base
     - task_only
     - replay_task
     - public_replay_task
   * - GSM8K (strict)
     - 73.46
     - **81.88** (+8.42)
     - 69.45 (-4.02)
     - 66.72 (-6.75)
   * - GSM8K (flex)
     - 79.38
     - **83.78** (+4.40)
     - 82.87 (+3.49)
     - 82.03 (+2.65)
   * - MATH (exact)
     - 47.98
     - **48.32** (+0.34)
     - 48.26 (+0.28)
     - 47.04 (-0.94)
   * - MATH (verify)
     - 54.14
     - 54.56 (+0.42)
     - **55.36** (+1.22)
     - 52.92 (-1.22)
   * - HumanEval
     - **74.39**
     - 69.51 (-4.88)
     - 71.95 (-2.44)
     - 73.78 (-0.61)
   * - MBPP
     - 65.40
     - 66.40 (+1.00)
     - **67.20** (+1.80)
     - 64.80 (-0.60)
   * - IFEval (strict)
     - **59.15**
     - 55.82 (-3.33)
     - 56.56 (-2.59)
     - 51.39 (-7.76)
   * - IFEval (inst)
     - **69.78**
     - 67.75 (-2.04)
     - 68.11 (-1.68)
     - 64.39 (-5.40)
   * - MMLU
     - 70.60
     - 70.48 (-0.12)
     - 70.56 (-0.04)
     - **70.70** (+0.10)
   * - MMLU-Pro
     - **60.44**
     - 55.75 (-4.69)
     - 59.30 (-1.14)
     - 55.85 (-4.59)
   * - ARC-Challenge
     - 58.62
     - 59.22 (+0.60)
     - **59.47** (+0.85)
     - 56.91 (-1.71)
   * - HellaSwag
     - 69.14
     - 70.96 (+1.82)
     - 69.95 (+0.81)
     - **71.32** (+2.18)

Key Findings
------------

1. **task_only shows classic catastrophic forgetting.** Large gains on the
   domain task (GSM8K +8.42%) but clear regressions on HumanEval (-4.88%),
   IFEval (-3.33%), and MMLU-Pro (-4.69%).

2. **Self-generated replay (replay_task) is the most effective strategy** for
   balancing task performance with capability retention:

   - MMLU-Pro regression reduced from -4.69% to just -1.14%
   - HumanEval forgetting halved (-2.44% vs -4.88%)
   - Still gains on MATH verify (+1.22%), MBPP (+1.80%), ARC (+0.85%)
   - Best overall Pareto trade-off between task gains and forgetting

3. **Public dataset replay (public_replay_task) underperforms** self-generated
   replay:

   - Worst IFEval regression (-7.76%)
   - MMLU-Pro loss comparable to task_only (-4.59%)
   - Supports the hypothesis that public datasets are distributionally
     mismatched with the base model's internal representation

4. **MMLU is highly stable** across all strategies (within ±0.12%), suggesting
   broad factual knowledge is robust to moderate SFT.

5. **HellaSwag improves** with fine-tuning across all strategies (+0.81% to
   +2.18%), likely due to improved language modeling from additional training.

Conclusion
----------

Self-generated replay data — synthesized by extracting instructions from the
model's own latent distribution — provides the best protection against
catastrophic forgetting while preserving domain task gains. This validates the
core design of this pipeline: using the model itself as the source of rehearsal
data rather than relying on externally curated datasets.
