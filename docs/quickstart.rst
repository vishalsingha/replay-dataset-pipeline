Quick Start
===========

Installation
------------

.. code-block:: bash

   cd /path/to/replay
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

Requirements: Python 3.10+, GPU(s) with enough VRAM for Qwen3-4B-Instruct-2507.

Minimal Run
-----------

1. **Generate instructions** from the model's latent distribution:

   .. code-block:: bash

      python -m src.generate_instructions --config config.yaml --quick

2. **Generate candidate responses** (L=3 per instruction):

   .. code-block:: bash

      python -m src.generate_responses --config config.yaml

3. **Filter** using the 5-point rubric judge:

   .. code-block:: bash

      python -m src.filter_responses --config config.yaml

4. **Check the output:**

   .. code-block:: bash

      wc -l data/replay/replay.jsonl
      head -1 data/replay/replay.jsonl | python -m json.tool

5. Set ``mixing.domain_data_path`` in ``config.yaml``, then **mix**:

   .. code-block:: bash

      python -m src.mix_datasets --config config.yaml

Output: ``data/final/train.jsonl`` — ready for TRL/Unsloth SFT training.

Resuming After a Crash
----------------------

Every script supports ``--resume`` to continue from the last checkpoint:

.. code-block:: bash

   python -m src.generate_instructions --config config.yaml --resume
