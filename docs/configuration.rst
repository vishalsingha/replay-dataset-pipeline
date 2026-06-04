Configuration
=============

All settings live in ``config.yaml``. The configuration is **validated at
startup** using Pydantic — typos, missing fields, and invalid ranges
produce clear error messages immediately.

Top-Level Settings
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Field
     - Description
     - Default
   * - ``model``
     - HuggingFace model ID
     - ``Qwen/Qwen3-4B-Instruct-2507``
   * - ``tensor_parallel_size``
     - Number of GPUs for vLLM tensor parallelism
     - ``2``
   * - ``seed``
     - Global random seed for reproducibility
     - ``42``
   * - ``log_level``
     - Logging verbosity (DEBUG/INFO/WARNING/ERROR)
     - ``INFO``

Deduplication
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Field
     - Description
     - Default
   * - ``dedupe.minhash_threshold``
     - MinHash LSH similarity threshold (0-1)
     - ``0.7``
   * - ``dedupe.minhash_num_perm``
     - Number of MinHash permutations
     - ``128``

Stop Tokens
-----------

Model-specific stop tokens for instruction and follow-up generation.
Change these when switching to a non-Qwen model.

.. code-block:: yaml

   stop_tokens:
     - "<|im_end|>"
     - "<|endoftext|>"
     - "<|im_start|>assistant"

Instruction Generation (Step 1)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Field
     - Description
     - Default
   * - ``instruction_generation.n``
     - Target raw instructions to write
     - ``20000``
   * - ``instruction_generation.quick_n``
     - N for ``--quick`` mode
     - ``1000``
   * - ``instruction_generation.chunk_size``
     - Instructions per checkpoint flush
     - ``5000``

Response Generation (Step 2)
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Field
     - Description
     - Default
   * - ``response_generation.L``
     - Candidates per generator per instruction
     - ``3``
   * - ``response_generation.chunk_size``
     - Instructions per checkpoint flush
     - ``1000``

Filtering (Step 3)
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Field
     - Description
     - Default
   * - ``filtering.chunk_size``
     - Candidates per checkpoint flush
     - ``1000``
   * - ``filtering.min_score``
     - Quality threshold (1-5 scale)
     - ``3.0``

Committee
---------

.. code-block:: yaml

   committee:
     generators:
       - {type: vllm_local, model: Qwen/Qwen3-4B-Instruct-2507}
     judges:
       - {type: vllm_local, model: Qwen/Qwen3-4B-Instruct-2507}

Supported types: ``vllm_local``, ``openai_api``.

Multi-turn requires the first generator to be ``vllm_local``.

System Prompt Pool
------------------

A weighted pool sampled per instruction during response generation.
Total weight must be > 0.

.. code-block:: yaml

   replay_system_prompts:
     - {prompt: "", weight: 3}
     - {prompt: "You are a helpful assistant.", weight: 2}

Mixing (Step 4)
---------------

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Field
     - Description
     - Default
   * - ``mixing.domain_fraction``
     - Fraction of domain data in final mix
     - ``0.17``
   * - ``mixing.domain_data_path``
     - Path to domain SFT dataset (JSONL)
     - **must set**
   * - ``mixing.val_fraction``
     - Validation split fraction (0 = no split)
     - ``0.0``

Paths
-----

.. code-block:: yaml

   paths:
     instructions: data/instructions/instructions.jsonl
     public_instructions: data/public_instructions/instructions.jsonl
     candidates: data/candidates/candidates.jsonl
     replay: data/replay/replay.jsonl
     public_replay: data/public_instructions/replay.jsonl
     multiturn: data/multiturn/multiturn.jsonl
     train: data/final/train.jsonl
     val: data/final/val.jsonl
