Replay Dataset Pipeline
=======================

Replay (rehearsal) dataset generation for mitigating catastrophic forgetting
during SFT.

Instead of mixing public SFT datasets (which are distributionally mismatched),
this pipeline **reconstructs the base model's own latent instruction
distribution** and synthesizes high-quality responses from it.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   pipeline
   configuration
   evaluation
   results
   api/index
   troubleshooting
