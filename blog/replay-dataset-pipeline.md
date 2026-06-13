# Why Your Fine-Tuned LLM Forgets Everything — and How Self-Synthesized Replay Fixes It

Fine-tuning makes a general LLM useful for your domain. But it also destroys general capabilities — coding, math, instruction following, multi-turn conversation. This is **catastrophic forgetting**, and the standard fixes don't work.

In this post, I explain why public data mixing fails, describe a research-backed alternative (self-synthesized replay), and share experimental results from both full fine-tuning and LoRA on a science domain task.

---

## The Problem

During SFT, model parameters shift toward the domain data optimum and away from general knowledge. Luo et al. ([arXiv:2308.08747](https://arxiv.org/abs/2308.08747)) found this intensifies with model scale, with math and factual recall degrading fastest.

Fine-tuning Llama-3-70B on medical data drops MATH scores by 30%+ and degrades MMLU-PRO measurably ([Ding & Wang, 2025](https://arxiv.org/abs/2506.09428)).

---

## Why Public Data Mixing Fails

The standard mitigation is mixing public SFT datasets (OpenOrca, UltraChat, ShareGPT) into training. Ding & Wang (2025) tested eight public datasets mixed with medical data — **every single one degraded general benchmarks**. Some dropped average scores from 39.00 to 35.91.

The reason: **distributional mismatch**. Public datasets reflect the instruction distribution of whatever model generated them (GPT-4, ChatGPT), not your model. Injecting mismatched data pulls the model toward a foreign distribution, making things worse.

---

## The Solution: Self-Generated Replay

Instead of borrowing data, **have the model generate its own replay data**. Two papers establish this:

**SSR** (Huang et al., ACL 2024, [arXiv:2403.01244](https://arxiv.org/abs/2403.01244)): Uses the LLM itself to generate synthetic rehearsal instances. The model's weights encode its training distribution — sampling from it produces inherently well-matched data.

**Ding & Wang (2025)** ([arXiv:2506.09428](https://arxiv.org/abs/2506.09428)): Uses the "pre-query trick" — feed only the chat template's user-turn prefix and let the model continue. The continuation naturally samples from the model's latent instruction distribution. Then generate multiple responses, filter by quality, and mix with domain data at ~83/17 ratio.

Their results on Llama-3-70B + medical data:

| Method | MMLU-PRO | MATH Lvl 5 | Average |
|---|---|---|---|
| Original model | 46.74 | 23.34 | 39.00 |
| Best public dataset (ShareGPT) | 46.14 | 20.24 | 38.00 |
| **Self-reconstructed replay** | **46.73** | **23.29** | **39.21** |

Critically, even the simplest version (one model, one response, no filtering) outperformed most public baselines. **Distribution matching matters more than response quality.**

---

## Our Experiments: Qwen3-4B on Science Domain

We validated this on `Qwen3-4B-Instruct-2507` fine-tuned on physics/chemistry/biology task data, comparing five strategies against the base model across 13+ benchmarks.

| Model | Description |
|-------|------------|
| **task_only** | Domain data only |
| **replay_task** | Self-generated replay + domain |
| **public_replay_task** | Public dataset replay + domain |
| **replay_public_replay_task** | Both replay sources + domain |
| **public_conv_task** | Raw public conversations + domain |

### Experiment 1: Full Fine-Tuning

**University-Level STEM (where forgetting hits hardest):**

| Benchmark | Base | task_only | replay_task | public_replay | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------:|
| MMLU-Pro Chemistry | **63.78** | 59.36 (-4.42) | 63.52 (-0.26) | 55.30 (-8.48) | 51.86 (-11.92) |
| MMLU-Pro Physics | **63.66** | 56.81 (-6.85) | 62.05 (-1.61) | 57.27 (-6.39) | 52.35 (-11.31) |
| MMLU-Pro Health | **58.92** | 54.40 (-4.52) | 58.19 (-0.73) | 57.33 (-1.59) | 54.28 (-4.64) |
| MMLU-Pro Math | **76.61** | 70.91 (-5.70) | 74.46 (-2.15) | 70.02 (-6.59) | 62.32 (-14.29) |

**Other key benchmarks:**

| Benchmark | Base | task_only | replay_task | public_replay | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------:|
| GSM8K (strict) | 73.46 | **81.88** (+8.42) | 69.90 (-3.56) | 64.29 (-9.17) | 77.79 (+4.32) |
| IFEval | **59.15** | 55.64 (-3.51) | 57.12 (-2.03) | 50.83 (-8.32) | 45.47 (-13.68) |
| HumanEval | **74.39** | 69.51 (-4.88) | 71.34 (-3.05) | 74.39 (0.00) | 63.41 (-10.98) |
| TruthfulQA | **62.60** | 55.71 (-6.89) | 60.61 (-1.99) | 58.49 (-4.12) | 55.91 (-6.69) |
| MT-Bench (/10) | 8.43 | 8.24 | 8.38 | 8.38 | 7.92 |
| Jailbreak Refusal | 100% | 100% | 100% | 100% | 100% |

**Ranking:** replay_task (-0.87% avg) > task_only (-0.78%) > replay_pub_rep (-2.19%) > public_replay (-2.71%) > pub_conv (-4.05%)

### Experiment 2: LoRA Fine-Tuning

LoRA (r=16, all linear layers) with the same data splits. Counter-intuitively, **LoRA without replay causes MORE damage than full fine-tuning.**

**University-Level STEM (LoRA):**

| Benchmark | Base | task_only | replay_task | public_replay | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------:|
| MMLU-Pro Chemistry | **63.78** | 56.63 (-7.15) | 63.60 (-0.18) | 54.86 (-8.92) | 48.32 (-15.46) |
| MMLU-Pro Physics | **63.66** | 54.81 (-8.85) | 62.43 (-1.23) | 58.74 (-4.92) | 49.65 (-14.01) |
| MMLU-Pro Health | 58.92 | 51.59 (-7.33) | **59.90** (+0.98) | 58.44 (-0.48) | 50.37 (-8.55) |
| MMLU-Pro Math | **76.61** | 69.58 (-7.03) | 76.31 (-0.30) | 73.28 (-3.33) | 59.81 (-16.80) |

**Other key benchmarks (LoRA):**

| Benchmark | Base | task_only | replay_task | public_replay | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------:|
| IFEval | **59.33** | 53.97 (-5.36) | 53.23 (-6.10) | 41.77 (-17.56) | 29.57 (-29.76) |
| HumanEval | 74.39 | 71.34 (-3.05) | **76.22** (+1.83) | 73.78 (-0.61) | 55.49 (-18.90) |
| TruthfulQA | **62.60** | 52.82 (-9.79) | 60.76 (-1.85) | 58.12 (-4.49) | 55.47 (-7.13) |
| MT-Bench (/10) | 8.48 | 7.51 | 8.42 | **8.49** | — |
| Jailbreak Refusal | 100% | **95%** | 100% | 100% | 100% |

**Ranking:** replay_task (-0.92% avg) > task_only (-1.44%) > replay_pub_rep (-3.31%) > public_replay (-4.21%) > pub_conv (-7.44%)

---

## Key Findings

**1. Self-replay preserves science knowledge 7x better (full FT) and 250x better (LoRA) than no replay.** MMLU-Pro science avg: -0.69% with replay vs -4.44% without (full FT); -0.03% vs -7.59% (LoRA).

**2. Task-only fine-tuning paradoxically hurts its own domain.** Despite training on science, MMLU-Pro Physics drops 6.85% (full FT) / 8.85% (LoRA). The model learns task format but loses deep reasoning.

**3. LoRA without replay is MORE destructive than full fine-tuning.** LoRA's low-rank constraint forces sharper parameter interference. TruthfulQA drops 9.79% (vs 6.89% full FT), and jailbreak refusal breaks to 95%.

**4. Self-replay with LoRA actually improves capabilities.** HumanEval +1.83%, Biology +0.56%, Health +0.98%. The replay acts as beneficial regularization.

**5. Public conversations are catastrophic.** `pub_conv` destroys IFEval (-29.76% LoRA), HumanEval (-18.90%), MMLU-Pro Math (-16.80%). Unfiltered conversations teach verbosity over precision.

**6. Safety remains intact — except LoRA without replay.** The only safety failure across all experiments: LoRA `task_only` drops to 95% jailbreak refusal. Self-replay prevents this.

---

## Practical Takeaways

- **Always use self-generated replay** when fine-tuning, regardless of method (full FT or LoRA)
- **Distribution matching > response quality**: even simple self-replay beats GPT-4-generated public data
- **LoRA needs replay MORE than full fine-tuning** — don't assume parameter efficiency prevents forgetting
- **~83% replay / 17% domain** is a good starting ratio
- **Avoid raw public conversations** as replay — they actively harm instruction following and reasoning

---

## References

1. Ding & Wang (2025). *Improved SFT for LLMs to Mitigate Catastrophic Forgetting.* [arXiv:2506.09428](https://arxiv.org/abs/2506.09428)
2. Huang et al. (2024). *Mitigating Catastrophic Forgetting with Self-Synthesized Rehearsal.* ACL 2024. [arXiv:2403.01244](https://arxiv.org/abs/2403.01244)
3. Luo et al. (2023). *An Empirical Study of Catastrophic Forgetting in LLMs.* [arXiv:2308.08747](https://arxiv.org/abs/2308.08747)
4. Wang et al. (2024). *Continual Learning of LLMs: A Comprehensive Survey.* [arXiv:2404.16789](https://arxiv.org/abs/2404.16789)
5. Wang et al. (2025). *SERS: Self-Evolving Pseudo-Rehearsal for Catastrophic Forgetting.* NeurIPS 2025.

---

*Open-source implementation: [github.com/vishalsingha/replay-dataset-pipeline](https://github.com/vishalsingha/replay-dataset-pipeline)*
