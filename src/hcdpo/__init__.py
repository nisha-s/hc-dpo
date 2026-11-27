"""HC-DPO: DPO with a hard KL budget enforced by a Lagrange multiplier."""

import os

# This project is PyTorch-only. Without these, transformers imports TensorFlow whenever
# it is installed (it is on Colab), which is slow and breaks if TensorFlow's own
# dependencies are out of step. Must be set before transformers is first imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
