"""End-to-end smoke test for the TAFS pipeline.

Verifies that the full pipeline works without errors:
    1. Build a tiny synthetic cache via TAFSSyntheticDataModule.
    2. Run the training-free baselines (UniformEnsemble, OracleRouter).
    3. Check that evaluate() returns the expected metric keys.

This test does NOT train the TAFS neural model — that requires the
model to be implemented first. Add a trained-model test here once
TAFS.forward() is implemented.

Run with:
    make smoke
    python -m pytest tests/test_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.core.utils.seeding import seed_everything
from src.tafs.data.datamodule import TAFSSyntheticDataModule
from src.tafs.models.baselines import OracleRouter, RandomRouter, UniformEnsemble


def test_synthetic_datamodule_builds(tmp_path: Path) -> None:
    """Verify that TAFSSyntheticDataModule can build and load its cache."""
    seed_everything(0)
    dm = TAFSSyntheticDataModule(
        cache_dir=str(tmp_path / "cache"),
        n_trajectories=10,
        T=40,
        seed=0,
        batch_size=16,
    )
    dm.setup("fit")
    assert dm._train is not None
    assert dm._val is not None
    assert dm._test is not None

    batch = next(iter(dm.train_dataloader()))
    assert "feature_matrix" in batch
    assert "error_matrix" in batch
    assert "base_preds" in batch
    assert "target" in batch


def test_baseline_evaluate(tmp_path: Path) -> None:
    """Verify that baseline routers evaluate without errors."""
    seed_everything(0)
    dm = TAFSSyntheticDataModule(
        cache_dir=str(tmp_path / "cache"),
        n_trajectories=20,
        T=40,
        seed=0,
        batch_size=16,
    )
    dm.setup("fit")

    num_experts = len(dm.expert_names)

    uniform = UniformEnsemble(num_experts=num_experts)
    metrics = uniform.evaluate(dm)
    assert "selected_error_mean" in metrics
    assert "oracle_error_mean" in metrics

    oracle = OracleRouter(num_experts=num_experts)
    oracle_metrics = oracle.evaluate(dm)
    # Oracle should not be worse than uniform on the same split.
    assert oracle_metrics["selected_error_mean"] <= metrics["selected_error_mean"] + 1e-6

    rnd = RandomRouter(num_experts=num_experts, seed=42)
    rnd_metrics = rnd.evaluate(dm)
    assert rnd_metrics["n_windows"] > 0
