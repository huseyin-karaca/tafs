"""Single entry point for reproducible randomness.

Rules:
    * Never call ``torch.manual_seed`` / ``np.random.seed`` /
      ``random.seed`` directly anywhere else in the codebase.
    * Always route through :func:`seed_everything`, which seeds Python's
      ``random``, NumPy, PyTorch (CPU + CUDA), and exports
      ``PYTHONHASHSEED`` so downstream subprocesses inherit the seed.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> int:
    """Seed every common PRNG. Returns the seed so callers can log it."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
