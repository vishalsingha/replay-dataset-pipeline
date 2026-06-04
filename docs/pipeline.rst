Pipeline Overview
=================

The pipeline has two independent instruction sources (Track A and Track B)
that feed into a shared response generation and filtering pipeline.

.. code-block:: text

   Track A (self-generated)              Track B (public datasets)
   ─────────────────────────             ─────────────────────────
   Step 1: generate_instructions         pull_public_instructions
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
   Step 3: filter_responses ◄──── (same script, --candidates flag)
     │                                     │
     └──────────────┬──────────────────────┘
                    ▼
   Step 4: mix_datasets (auto-loads all replay sources)
                    │
                    ▼
               train.jsonl


Step 1: Instruction Generation
------------------------------

Reconstructs the model's latent instruction distribution by feeding only
the user-turn chat template prefix and letting the model continue. The
model's continuation is the synthetic instruction.

Uses high-diversity nucleus sampling (``temperature≈1.0``, ``top_p≈0.98``)
since the objective is to broadly cover the model's instruction distribution.

After generation, instructions are deduplicated via exact match and
MinHash LSH near-duplicate removal.

.. code-block:: bash

   python -m src.generate_instructions --config config.yaml
   python -m src.generate_instructions --config config.yaml --quick  # test mode

Step 1b: Multi-Turn Generation
------------------------------

Takes a configurable fraction (default 30%) of instructions and extends
them into multi-turn conversations (2-5 turns). The model generates both
follow-up user messages (via raw text completion) and assistant responses
(via chat completion).

Requires ``vllm_local`` backend (needs tokenizer + raw generation).

.. code-block:: bash

   python -m src.generate_multiturn --config config.yaml

Step 2: Multi-Response Generation
---------------------------------

For each instruction, a committee of generators produces L (default 3)
candidate responses via nucleus sampling. A system prompt is sampled from
a diverse pool for each instruction to exercise the system-prompt
conditioning channel.

Uses vLLM's ``n`` parameter to generate all L candidates in a single call.

.. code-block:: bash

   python -m src.generate_responses --config config.yaml

Step 3: Filtering
-----------------

A committee of judges scores each candidate on a 5-point rubric
(helpfulness, relevance, clarity, AI-persona).
The best-scoring response is selected per instruction.

A detailed summary is printed at the end including parse failure rate
and sample failure reasons.

.. code-block:: bash

   python -m src.filter_responses --config config.yaml

Step 4: Mixing
--------------

Combines all replay sources (single-turn, public, multi-turn) with
your domain SFT data at a configurable ratio (default 17% domain / 83%
replay). Shuffles and optionally splits into train/val.

.. code-block:: bash

   python -m src.mix_datasets --config config.yaml

Public Instructions Pipeline
----------------------------

Pulls prompts only (discards original responses) from public HuggingFace
datasets. Supports streaming for large datasets. Per-source progress is
checkpointed for resumability.

Supported datasets:

- ``Open-Orca/OpenOrca`` — Instruction following
- ``teknium/OpenHermes-2.5`` — General chat + reasoning
- ``HuggingFaceH4/ultrachat_200k`` — Multi-turn chat
- ``m-a-p/CodeFeedback-Filtered-Instruction`` — Code ability
- ``TIGER-Lab/MathInstruct`` — Math/reasoning

.. code-block:: bash

   python -m src.pull_public_instructions --config config.yaml
