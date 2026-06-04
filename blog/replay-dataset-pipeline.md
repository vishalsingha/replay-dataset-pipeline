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

*If you want to try self-synthesized replay on your own models, I've open-sourced an implementation at [github.com/vishalsingha/replay-dataset-pipeline](https://github.com/vishalsingha/replay-dataset-pipeline) — it supports self-generated and public instruction tracks, multi-turn data, multi-model committees, and production features like checkpointing and config validation.*
