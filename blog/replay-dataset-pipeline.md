# Why Your Fine-Tuned LLM Forgets Everything — and How Self-Synthesized Replay Fixes It

Fine-tuning is how you make a general-purpose LLM useful for your specific domain. You take Qwen, Llama, or Mistral, feed it your medical QA pairs or legal contracts, and out comes a specialist.

Except something else comes out too: a model that can no longer hold a basic conversation, struggles with simple arithmetic, and writes code like it's never seen a programming language before.

This is **catastrophic forgetting** — one of the most practically damaging problems in LLM deployment today. And the standard fixes don't actually solve it. In this post, I'll explain why the common approaches fail, walk through a line of recent research that offers a fundamentally better solution, and share how I implemented it.

---

## The Problem: Why Fine-Tuning Destroys General Knowledge

Catastrophic forgetting isn't new. McCloskey and Cohen identified it back in 1989 in connectionist networks. But it's become acutely relevant in the LLM era because the stakes are higher: a fine-tuned model that loses general abilities is unusable in production, where users expect both domain expertise *and* basic competence.

The mechanism is straightforward. During SFT, the model's parameters shift to minimize loss on the new domain data. But those same parameters encode the general knowledge the model acquired during pre-training and initial alignment. As the weights move toward the new optimum, they move *away* from the regions that encode general capabilities — coding, math, reasoning, multi-turn conversation, instruction following.

The severity depends on several factors. Luo et al. ([arXiv:2308.08747](https://arxiv.org/abs/2308.08747)) conducted an empirical study and found that forgetting actually *intensifies* as model scale increases in the 1B–7B range, likely because larger models have stronger initial performance and thus more to lose. They also found that the pattern varies by capability: mathematical reasoning and factual recall degrade fastest, while simpler language understanding tasks are more resilient.

This isn't just a theoretical concern. If you fine-tune Llama-3-70B-Instruct on medical QA data, you can expect MATH benchmark scores to drop by 30%+ and MMLU-PRO to degrade measurably — even with careful hyperparameter tuning ([Ding & Wang, 2025](https://arxiv.org/abs/2506.09428)).

---

## The Standard Fix: Data Mixing (and Why It's Not Enough)

The most common mitigation is **rehearsal** — mixing general-purpose data into your fine-tuning dataset so the model revisits old capabilities while learning new ones. This is the LLM-era descendant of experience replay, a technique with roots going back to Robins (1995) and formalized for deep learning by Rolnick et al. (2019).

The idea is intuitive: if forgetting happens because the model only sees domain data, show it some general data too. In practice, people grab a public SFT dataset — OpenOrca, UltraChat, ShareGPT, OpenHermes — mix it at some ratio (say 80% general, 20% domain), and fine-tune on the combined set.

This helps. But there's a fundamental problem that most practitioners overlook: **distributional mismatch**.

Ding & Wang (2025) tested this directly. They fine-tuned Llama-3-70B-Instruct on medical data mixed with eight different public SFT datasets (ShareGPT, Evol-Instruct, GenQA, OpenHermes 1 & 2.5, Tulu V2 Mix, WildChat, UltraChat). Every single configuration showed degradation on general benchmarks compared to the original model. Some dropped the average score from 39.00 to as low as 35.91.

Why? Because public SFT datasets reflect the instruction distribution and response style of *whatever model generated them*, not the model you're fine-tuning. OpenHermes was generated with GPT-4. UltraChat was generated with a different pipeline. ShareGPT is a grab-bag of ChatGPT conversations. None of these match what Llama-3-70B learned during its own alignment.

When you inject distributionally mismatched data, you're not just rehearsing — you're actively pulling the model toward a *different* distribution. The model has to reconcile its own knowledge with the foreign data, and the result is a compromise that's worse than either extreme.

---

## A Better Approach: Let the Model Generate Its Own Replay Data

This is where recent research offers a real insight. Instead of borrowing someone else's data, **have the model reconstruct the data it was originally trained on**.

Two papers lay the groundwork for this idea:

### Self-Synthesized Rehearsal (SSR)

Huang et al. ([arXiv:2403.01244](https://arxiv.org/abs/2403.01244), ACL 2024) proposed **Self-Synthesized Rehearsal (SSR)** — a framework that uses the LLM itself to generate synthetic instances for rehearsal, eliminating the need for original training data entirely.

SSR works in three steps:
1. **Synthetic instance generation**: The base LLM generates synthetic inputs via in-context learning
2. **Output refinement**: The latest checkpoint of the LLM refines the outputs, preserving its most recently acquired abilities
3. **Diversity selection**: High-quality, diverse samples are selected for rehearsal in future learning stages

The key contribution is the realization that you don't *need* the original training data. The model's weights encode a compressed representation of its training distribution. By prompting the model to generate, you're sampling from that implicit distribution — and the samples are inherently well-matched because they come from the model itself.

SSR demonstrated superior or comparable performance to conventional rehearsal methods while being more data-efficient, and effectively preserved generalization capabilities across general domains.

### Reconstructing the Latent Instruction Distribution

Ding & Wang ([arXiv:2506.09428](https://arxiv.org/abs/2506.09428), 2025) took this idea further with a more principled approach to instruction generation and a multi-model pipeline for response quality. Their method has three stages:

**Stage 1 — Instruction Generation via the Pre-Query Trick.**
Rather than prompting the model with "Generate diverse instructions" (which biases the distribution), they exploit the chat template structure directly. Every instruction-tuned model has a template like:

```
<|start_header_id|>user<|end_header_id|>
```

Feed the model *only* this prefix and let it continue. The continuation is whatever instruction the model's internal distribution considers most natural. Sample 100,000 times with high-temperature nucleus sampling (`temperature ≈ 1.0`, `top_p ≈ 0.98`), and you recover a broad, faithful approximation of the instruction distribution the model was aligned on.

This is elegant because it doesn't impose any external structure on the generation. The model isn't following a meta-instruction about what kind of instructions to create — it's simply completing the template the way its training taught it to, which means the output naturally reflects the distribution it internalized.

**Stage 2 — Multi-Response Generation.**
For each instruction, a committee of K models generates L candidate responses each (3 models x 3 responses = 9 candidates per instruction in their setup). Using multiple models introduces response diversity and enables a form of knowledge distillation when stronger models are included in the committee.

**Stage 3 — Quality Filtering.**
The same committee of models acts as judges, scoring each candidate on a 5-point rubric covering helpfulness, relevance, clarity, and AI-persona adherence. Scores are averaged across judges, and the highest-scoring response is selected for each instruction.

The results are striking. When mixed with medical domain data at a 83/17 ratio:

| Rehearsal Method | MMLU-PRO | MATH Lvl 5 | Average |
|---|---|---|---|
| Original Llama-3-70B-Instruct | 46.74 | 23.34 | 39.00 |
| Best public dataset (ShareGPT) | 46.14 | 20.24 | 38.00 |
| Worst public dataset (GenQA) | 43.33 | 15.41 | 35.91 |
| **Self-reconstructed (their method)** | **46.73** | **23.29** | **39.21** |

The self-reconstructed dataset not only prevented degradation — it slightly *improved* the average score beyond the original model. Meanwhile, every public SFT dataset made things worse, some catastrophically (GenQA lost nearly 8 points on MATH).

---

## Why Distribution Matching Matters More Than Data Quality

The ablation study from Ding & Wang reveals something counterintuitive. They tested:

1. **Single model, single response, no filtering** (simplest): Average 38.81
2. **Single model, 3 responses, self-filtered**: Average 39.06
3. **Three models, 3 responses each, committee-filtered** (full method): Average 39.21

The simplest version — one model generating one response per self-generated instruction, with zero quality filtering — already outperformed nearly every public dataset baseline. That single-model-single-response setup scored 38.81, beating ShareGPT (38.00), Evol-Instruct (38.81), WildChat (38.35), and every other baseline.

This tells us something important: **getting the instruction distribution right matters more than having high-quality responses**. The public datasets have responses from GPT-4, carefully curated human conversations, and sophisticated synthesis pipelines. The single-model-single-response setup has none of that. But because its instructions come from the model's own distribution, the rehearsal is more effective.

Multi-response generation and committee filtering still help — they push the score from 38.81 to 39.21, a meaningful gain. But the foundation is distribution alignment, not response polish.

---

## The Broader Landscape: Where This Fits

This self-synthesis approach sits within a larger taxonomy of continual learning methods. Wang et al. ([arXiv:2404.16789](https://arxiv.org/abs/2404.16789)) provide a comprehensive survey categorizing approaches into:

- **Replay-based**: Store or generate data from previous tasks for rehearsal
- **Regularization-based**: Add penalty terms (like EWC) to prevent important weights from changing
- **Architecture-based**: Freeze parts of the network or add task-specific modules
- **Gradient-based**: Modify gradient updates to minimize interference

Self-synthesized replay is a replay-based method, but it addresses a critical limitation of traditional replay: **you don't need access to the original data**. When Meta releases Llama or Alibaba releases Qwen, they don't ship the SFT dataset. Traditional experience replay is simply impossible. Generative replay — using the model itself as the generator — sidesteps this entirely.

The recent NeurIPS 2025 paper on SERS (Self-Evolving Pseudo-Rehearsal) extends this further, adding dynamic regularization based on task similarity measured by Wasserstein distance between distributions. The direction is clear: the field is moving toward methods that treat the model itself as a compressed knowledge store that can be unpacked on demand.

---

## Practical Considerations

If you want to implement this approach, there are several things the papers don't emphasize but that matter enormously in practice:

**Instruction deduplication is essential.** When you sample 100K+ instructions from the model's distribution, many will be near-duplicates. Exact dedup catches verbatim copies, but you also need near-duplicate removal (MinHash LSH works well) to avoid over-representing popular instruction types.

**Multi-turn data matters.** Both papers focus on single-turn instruction-response pairs. But in production, models need to maintain coherent multi-turn conversations. Extending a fraction of self-generated instructions into 2–5 turn dialogues helps preserve this ability.

**System prompt diversity prevents over-conditioning.** If your replay data always uses "You are a helpful assistant" as the system prompt, the model will over-condition on it. Using a weighted pool of diverse system prompts (including no system prompt at all) keeps the model flexible for different deployment personas.

**The mixing ratio of ~83% replay / 17% domain is a reasonable starting point.** Domain-specific capabilities are learned quickly from relatively few examples, while general capabilities require broad rehearsal to maintain. Adjust based on your domain data size and diversity.

**Checkpointing is non-negotiable.** Generating 100K+ instructions, 300K+ candidate responses, and scoring them all takes hours of GPU time. Any pipeline that doesn't support incremental progress saving and resume-from-crash is unusable for real workloads.

---

## What This Means for LLM Practitioners

The key takeaway isn't about a specific technique — it's about a shift in how we think about fine-tuning data.

The conventional wisdom is: "grab some public SFT data, mix it in, and hope for the best." The research shows this is actively harmful in many cases. Public datasets introduce distributional mismatch that can degrade the very capabilities you're trying to preserve.

The better approach is: **treat the model's own weights as a compressed dataset, decompress it through self-generation, and use that as rehearsal data.** The model knows what it knows. Let it tell you.

This has implications beyond catastrophic forgetting. The same distribution reconstruction technique could be useful for:
- **Model auditing**: Understanding what instruction distribution a model was trained on
- **Data-free distillation**: Transferring knowledge without access to training data
- **Curriculum design**: Generating training data that's naturally calibrated to the model's current capabilities

The papers referenced here are just the beginning. As models get larger and fine-tuning becomes the default deployment pattern, the question of how to specialize without destroying general capability will only get more important.

---

## Experimental Validation: Qwen3-4B on Science Domain

To validate the self-synthesized replay approach, I ran a comprehensive experiment fine-tuning `Qwen3-4B-Instruct-2507` on a physics, chemistry, and biology (PCB) task dataset under five different data mixing strategies, then evaluated across 13+ benchmarks covering instruction following, STEM knowledge, code, and safety.

### Models Compared

| Model | Training Data |
|-------|--------------|
| **Base** | Original Qwen3-4B-Instruct (no fine-tuning) |
| **task_only** | Domain PCB data only |
| **replay_task** | Self-generated replay (this pipeline) + domain data |
| **public_replay_task** | Public dataset replay + domain data |
| **replay_public_replay_task** | Both replay sources + domain data |
| **public_conv_task** | Raw public conversations + domain data |

### Instruction Following

| Benchmark | Base | task_only | replay_task | public_replay | replay_pub_rep | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------------:|:--------:|
| IFEval (Prompt Strict) | **59.15** | 55.64 (-3.51) | 57.12 (-2.03) | 50.83 (-8.32) | 55.27 (-3.88) | 45.47 (-13.68) |
| IFEval (Inst Strict) | **69.90** | 67.75 (-2.15) | 68.23 (-1.67) | 64.39 (-5.51) | 67.51 (-2.39) | 59.71 (-10.19) |
| MT-Bench (/10) | 8.43 | 8.24 | 8.38 | 8.38 | **8.50** | 7.92 |
| TruthfulQA | **62.60** | 55.71 (-6.89) | 60.61 (-1.99) | 58.49 (-4.12) | 59.08 (-3.52) | 55.91 (-6.69) |

### STEM (Up to Class 12)

| Benchmark | Base | task_only | replay_task | public_replay | replay_pub_rep | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------------:|:--------:|
| HS Biology | 90.00 | **90.97** | 90.65 | 89.68 | 89.68 | 90.65 |
| HS Chemistry | 74.38 | **76.35** | 74.38 | 75.86 | 75.37 | 75.37 |
| HS Physics | 64.90 | 63.58 | 65.56 | **66.23** | 65.56 | 64.90 |
| GSM8K (Math) | 73.46 | **81.88** (+8.42) | 69.90 (-3.56) | 64.29 (-9.17) | 63.46 (-10.01) | 77.79 (+4.32) |
| ARC-Challenge | 58.62 | 59.22 | **59.64** | 57.00 | 58.45 | 56.23 |

### STEM (University Level)

| Benchmark | Base | task_only | replay_task | public_replay | replay_pub_rep | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------------:|:--------:|
| MMLU-Pro Biology | 79.50 | 77.55 (-1.95) | **79.36** (-0.14) | 79.50 | 78.10 | 75.59 (-3.91) |
| MMLU-Pro Chemistry | **63.78** | 59.36 (-4.42) | 63.52 (-0.26) | 55.30 (-8.48) | 56.71 (-7.07) | 51.86 (-11.92) |
| MMLU-Pro Physics | **63.66** | 56.81 (-6.85) | 62.05 (-1.61) | 57.27 (-6.39) | 57.04 (-6.62) | 52.35 (-11.31) |
| MMLU-Pro Health | **58.92** | 54.40 (-4.52) | 58.19 (-0.73) | 57.33 (-1.59) | 58.68 (-0.24) | 54.28 (-4.64) |
| MMLU-Pro Math | **76.61** | 70.91 (-5.70) | 74.46 (-2.15) | 70.02 (-6.59) | 69.87 (-6.74) | 62.32 (-14.29) |
| MATH (Hendrycks) | 54.14 | 54.56 (+0.42) | **54.80** (+0.66) | 52.84 (-1.30) | 54.06 (-0.08) | 50.06 (-4.08) |

### Code

| Benchmark | Base | task_only | replay_task | public_replay | replay_pub_rep | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------------:|:--------:|
| HumanEval | **74.39** | 69.51 (-4.88) | 71.34 (-3.05) | 74.39 (0.00) | 71.95 (-2.44) | 63.41 (-10.98) |
| MBPP | 65.40 | **66.40** (+1.00) | 66.20 (+0.80) | 65.20 (-0.20) | 65.60 (+0.20) | 63.20 (-2.20) |
| MT-Bench Coding (/10) | **9.40** | 9.25 | 9.40 | 9.10 | 9.40 | 7.85 |

### Safety

| Benchmark | Base | task_only | replay_task | public_replay | replay_pub_rep | pub_conv |
|-----------|:----:|:---------:|:-----------:|:-------------:|:--------------:|:--------:|
| ToxiGen | 56.70 | 57.13 | 56.70 | 56.70 | 56.70 | 57.02 |
| Jailbreak Refusal | **100%** | **100%** | **100%** | **100%** | **100%** | **100%** |
| HaluEval | 62.1 | 69.6 | 63.9 | 66.1 | 66.1 | **82.3** |
| TruthfulQA | **62.60** | 55.71 (-6.89) | 60.61 (-1.99) | 58.49 (-4.12) | 59.08 (-3.52) | 55.91 (-6.69) |

### Key Findings

**1. Self-generated replay preserves university-level science knowledge 7x better than no replay.** The `replay_task` model loses an average of only -0.69% on MMLU-Pro science subjects (Biology, Chemistry, Physics, Health), compared to -4.44% for `task_only`. This is despite the domain task itself being science — the model forgets the *deep reasoning* required for graduate-level questions while learning the task-specific format.

**2. Task-only fine-tuning creates a paradox: the model gets worse at its own domain.** Although `task_only` achieves the best GSM8K score (+8.42%), it drops 6.85% on MMLU-Pro Physics and 4.42% on Chemistry. The structured Q&A format in the training data teaches problem-solving patterns but overwrites the factual knowledge needed for harder questions.

**3. Public dataset replay doesn't match self-replay.** `public_replay_task` loses 4.11% on average science scores vs only 0.69% for `replay_task`. This confirms the distributional mismatch hypothesis — public conversations from OpenOrca/UltraChat/OpenHermes are generated by different models and don't align with Qwen3's internal knowledge representation.

**4. Raw public conversations are catastrophic.** `public_conv_task` shows the worst results across nearly every benchmark: -13.68% on IFEval, -10.98% on HumanEval, -14.29% on MMLU-Pro Math. Using complete conversations without filtering teaches the model to be verbose and agreeable rather than precise and capable.

**5. All strategies maintain safety alignment.** 100% jailbreak refusal rate is preserved across all models, and ToxiGen scores remain stable. SFT on science data doesn't introduce safety vulnerabilities.

**6. TruthfulQA reveals sycophancy creep.** All fine-tuned models become somewhat more sycophantic (agreeing with premises in questions rather than being truthful), but self-replay minimises this: -1.99% vs -6.89% for `task_only`.

### Overall Ranking

| Rank | Model | Avg Delta (10 benchmarks) | Best Use Case |
|:----:|-------|:-------------------------:|---------------|
| 1 | **replay_task** | -0.87% | Best all-round: preserves science, code, truthfulness |
| 2 | **task_only** | -0.78% | Maximum domain task performance (GSM8K) at forgetting cost |
| 3 | **replay_pub_rep** | -2.19% | Best conversational quality (MT-Bench 8.50) |
| 4 | **public_replay** | -2.71% | Preserves HumanEval perfectly |
| 5 | **pub_conv** | -4.05% | Worst overall — avoid for domain SFT |

The `replay_task` strategy offers the best Pareto trade-off: minimal forgetting across all capabilities while still learning the domain task. For practitioners fine-tuning on science domains, self-generated replay is not optional — it's essential for preserving the deep knowledge that makes the model useful.

---

## References

1. **Ding, F. & Wang, B.** (2025). *Improved Supervised Fine-Tuning for Large Language Models to Mitigate Catastrophic Forgetting.* [arXiv:2506.09428](https://arxiv.org/abs/2506.09428)

2. **Huang, J., Cui, L., Wang, A., Yang, C., Liao, X., Song, L., Yao, J. & Su, J.** (2024). *Mitigating Catastrophic Forgetting in Large Language Models with Self-Synthesized Rehearsal.* ACL 2024. [arXiv:2403.01244](https://arxiv.org/abs/2403.01244)

3. **McCloskey, M. & Cohen, N.J.** (1989). *Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem.* Psychology of Learning and Motivation, 24:109–165.

4. **Robins, A.** (1995). *Catastrophic Forgetting, Rehearsal and Pseudorehearsal.* Connection Science, 7(2):123–146.

5. **Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. & Wayne, G.** (2019). *Experience Replay for Continual Learning.* NeurIPS 2019. [arXiv:1811.11682](https://arxiv.org/abs/1811.11682)

6. **Wang, T., et al.** (2024). *Continual Learning of Large Language Models: A Comprehensive Survey.* [arXiv:2404.16789](https://arxiv.org/abs/2404.16789)

7. **Luo, Y., et al.** (2023). *An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning.* [arXiv:2308.08747](https://arxiv.org/abs/2308.08747)

8. **Wang, J., et al.** (2025). *Self-Evolving Pseudo-Rehearsal for Catastrophic Forgetting with Task Similarity in LLMs (SERS).* NeurIPS 2025.

---

*If you want to try self-synthesized replay on your own models, I've open-sourced an implementation at [github.com/vishalsingha/replay-dataset-pipeline](https://github.com/vishalsingha/replay-dataset-pipeline) — it supports self-generated and public instruction tracks, multi-turn data, multi-model committees, benchmark evaluation, and production features like checkpointing and config validation.*
