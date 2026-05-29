"""Seeding utility correctness."""

from __future__ import annotations

import numpy as np
import torch

from src.core.utils.seeding import seed_everything


def test_seed_makes_random_streams_reproducible() -> None:
    seed_everything(123)
    a_np = np.random.rand(5)
    a_torch = torch.randn(5)

    seed_everything(123)
    b_np = np.random.rand(5)
    b_torch = torch.randn(5)

    np.testing.assert_allclose(a_np, b_np)
    torch.testing.assert_close(a_torch, b_torch)
