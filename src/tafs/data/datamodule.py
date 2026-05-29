"""TAFS LightningDataModule.

Provides train/val/test splits of (feature_matrix, base_preds, targets,
error_matrix) tensors. Subclasses BaseDataModule for manifest-aware
caching: if the on-disk cache matches the config hash, it loads from
disk; otherwise it rebuilds.

Two concrete subclasses are provided here:
    TAFSSyntheticDataModule   — wraps the synthetic regime-switch generator.
    TAFSRealDataModule        — wraps a real dataset from the feature pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.core.data.base import BaseDataModule, CacheManifest
from src.tafs.data.synthetic import BASE_PREDICTOR_NAMES, SyntheticConfig, generate_dataset


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TAFSDataset(Dataset):
    """In-memory dataset for a pre-built TAFS cache.

    Each item is a dict with:
        feature_matrix: (p,)   float32
        base_preds:     (K,)   float32
        target:         ()     float32
        error_matrix:   (K,)   float32
        regime:         ()     int8   (0 in real-data mode)
    """

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.feature_matrix = torch.from_numpy(arrays["feature_matrix"])
        self.base_preds = torch.from_numpy(arrays["base_preds"])
        self.targets = torch.from_numpy(arrays["targets"])
        self.error_matrix = torch.from_numpy(arrays["error_matrix"])
        self.regime = torch.from_numpy(arrays.get("regime", np.zeros(len(self.targets), dtype=np.int8)))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "feature_matrix": self.feature_matrix[idx],
            "base_preds": self.base_preds[idx],
            "target": self.targets[idx],
            "error_matrix": self.error_matrix[idx],
            "primary_regime": self.regime[idx],
        }


# ---------------------------------------------------------------------------
# Synthetic DataModule
# ---------------------------------------------------------------------------


class TAFSSyntheticDataModule(BaseDataModule):
    """DataModule for the synthetic regime-switch experiment.

    Args:
        cache_dir: path where the cache directory lives (e.g. data/processed/tafs/synthetic).
        n_trajectories: number of synthetic trajectories to generate.
        T: time steps per trajectory.
        regime_change: index where regime switches from A to B.
        noise_std: observation noise standard deviation.
        seed: RNG seed for data generation (separate from the split seed).
        **kwargs: forwarded to BaseDataModule (batch_size, num_workers,
                  train_ratio, val_ratio, split_mode, etc.).
    """

    def __init__(
        self,
        cache_dir: str = "data/processed/tafs/synthetic",
        n_trajectories: int = 500,
        T: int = 400,
        regime_change: int = 200,
        noise_std: float = 0.3,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(cache_dir=cache_dir, seed=seed, **kwargs)
        self._cfg = SyntheticConfig(
            n_trajectories=n_trajectories,
            T=T,
            regime_change=regime_change,
            noise_std=noise_std,
            seed=seed,
        )

    @property
    def expert_names(self) -> list[str]:
        return list(BASE_PREDICTOR_NAMES)

    def config_payload(self) -> dict[str, Any]:
        """Uniquely identifies this cache (used for manifest hash)."""
        from dataclasses import asdict
        return asdict(self._cfg)

    def build_cache(self) -> None:
        """Generate data and write to self.cache_dir."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        arrays = generate_dataset(self._cfg)
        # Save each array as .npy
        for key, arr in arrays.items():
            np.save(self.cache_dir / f"{key}.npy", arr)
        CacheManifest.write(
            self.cache_dir / "manifest.json",
            dataset="synthetic",
            config_payload=self.config_payload(),
            n_windows=len(arrays["targets"]),
        )

    def _make_dataset(self) -> Dataset:
        arrays = {
            key: np.load(self.cache_dir / f"{key}.npy")
            for key in ("feature_matrix", "base_preds", "targets", "error_matrix", "regime")
        }
        return TAFSDataset(arrays)


# ---------------------------------------------------------------------------
# Real-dataset DataModule (stub)
# ---------------------------------------------------------------------------


class TAFSRealDataModule(BaseDataModule):
    """DataModule for a real dataset processed by the feature pipeline.

    The cache at ``cache_dir`` is expected to contain:
        feature_matrix.npy  (N, p)  float32
        base_preds.npy      (N, K)  float32
        targets.npy         (N,)    float32
        error_matrix.npy    (N, K)  float32
        manifest.json

    The cache is written by ``make prepare_data DATASET=<name>``, which
    calls ``src.tafs.data.feature_pipeline`` to build features and run
    the K base predictors.

    Args:
        cache_dir: path to the pre-built cache for this dataset.
        expert_names_list: ordered list of K base predictor names (must
            match the column order in error_matrix.npy).
        **kwargs: forwarded to BaseDataModule.
    """

    def __init__(
        self,
        cache_dir: str,
        expert_names_list: list[str],
        **kwargs: Any,
    ) -> None:
        super().__init__(cache_dir=cache_dir, **kwargs)
        self._expert_names_list = expert_names_list

    @property
    def expert_names(self) -> list[str]:
        return self._expert_names_list

    def config_payload(self) -> dict[str, Any]:
        # TODO: return a dict that uniquely identifies the feature pipeline
        # version + dataset + base predictor config. This is used by the
        # manifest check to detect stale caches.
        raise NotImplementedError

    def build_cache(self) -> None:
        # TODO: call the feature pipeline and base predictor runner.
        # Normally this is invoked via `make prepare_data`, not during training.
        raise NotImplementedError

    def _make_dataset(self) -> Dataset:
        arrays = {
            key: np.load(self.cache_dir / f"{key}.npy")
            for key in ("feature_matrix", "base_preds", "targets", "error_matrix")
        }
        return TAFSDataset(arrays)
