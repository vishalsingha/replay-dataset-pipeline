Troubleshooting
===============

Config Validation Error
-----------------------

The pipeline validates ``config.yaml`` with Pydantic before any GPU work.
Common causes:

- Typo in a key name (e.g. ``commitee`` instead of ``committee``)
- ``domain_fraction`` outside (0, 1)
- ``min_turns`` > ``max_turns``
- Empty ``generators`` or ``judges`` list

Low Filter Pass Rate
--------------------

Check the filtering summary printed at the end. Common causes:

- **High parse failure %** — Judge isn't outputting ``Score: N``. Lower
  ``filtering.sampling.temperature`` or try a larger judge model.
- **All scores below threshold** — Lower ``filtering.min_score`` to ``2.0``.
- The sample failures show the raw judge output tail.

Out of GPU Memory
-----------------

- Reduce ``chunk_size`` in the relevant step
- Reduce ``response_generation.sampling.max_tokens``
- Use ``tensor_parallel_size: 1`` for single GPU

Crash Recovery
--------------

Add ``--resume`` to any command to continue from last checkpoint:

.. code-block:: bash

   python -m src.generate_responses --config config.yaml --resume

All scripts checkpoint to disk after each chunk. No GPU work is lost.

Public Dataset Download Fails
-----------------------------

- Run ``huggingface-cli login`` for gated datasets
- ``streaming: true`` avoids full downloads
- 10M iteration safety cap prevents infinite loops

Empty Instructions
------------------

The pipeline aborts after 5 consecutive zero-yield chunks. Check:

- ``instruction_generation.sampling.temperature`` (try ``0.8``)
- ``stop_tokens`` match your model's template
- ``min_length`` < ``max_length``
