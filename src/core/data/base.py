"""Base LightningDataModule with manifest-aware caching.

Two-stage caching contract:

    Stage 1: raw -> ``data/interim/<dataset>/<split>.parquet``
             (materialised by ``make prepare_data DATASET=<name>``).
    Stage 2: interim -> ``data/processed/tafs/<dataset>/`` cache of
             feature matrices, base predictor outputs, and the per-step
             error matrix.

Each cache directory carries a ``manifest.json`` recording the dataset
name, version hash, feature pipeline version, timestamp, git SHA, and
sample count. The DataModule refuses to load a cache whose manifest
hash does not match the config it was asked to materialise.

Subclasses implement:
    * :meth:`build_cache` — materialise the cache directory.
    * :meth:`_make_dataset` — return a ``torch.utils.data.Dataset``
      that yields ``dict[str, Tensor]`` batches with keys:
      ``error_matrix`` (B, K), ``feature_tokens`` (B, p), ``context`` (B, C).
"""

from __future__ import annotations

from abc import abstractmethod
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, Subset

from src.core.utils import git


def _hash_config(payload: dict[str, Any]) -> str:
    """Stable short hash of a JSON-serialisable config dict."""
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class CacheManifest:
    """Schema for ``manifest.json`` files in every cache directory.

    The router rejects a cache whose recorded ``config_hash`` differs
    from the hash of the config it was asked to load — this catches
    silent staleness when the user edits a feature pipeline or encoder
    version without bumping the cache directory name.
    """

    @staticmethod
    def write(path: Path, *, dataset: str, config_payload: dict[str, Any], n_windows: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": dataset,
            "config_hash": _hash_config(config_payload),
            "config": config_payload,
            "n_windows": n_windows,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git": git.repo_state(),
        }
        path.write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text())

    @staticmethod
    def check(path: Path, config_payload: dict[str, Any]) -> bool:
        if not path.exists():
            return False
        manifest = CacheManifest.load(path)
        return manifest.get("config_hash") == _hash_config(config_payload)

    @staticmethod
    def hash(config_payload: dict[str, Any]) -> str:
        """Public view of the cache-key hash, for remote-cache addressing."""
        return _hash_config(config_payload)


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class BaseDataModule(pl.LightningDataModule):
    """Cache-aware LightningDataModule.

    Subclass responsibility:
        * :meth:`build_cache` — write the cache directory + manifest.
        * :meth:`_make_dataset` — return a torch ``Dataset``.
        * :meth:`expert_names` — list of K expert / base-predictor names.
        * :meth:`config_payload` — dict whose hash identifies the cache.

    The base class provides:
        * setup() that builds the cache lazily, then constructs the
          full dataset and splits it deterministically by seed.
        * ``class_priors`` (oracle argmin distribution on the training
          split, used by ``class_balanced_loss``).
        * reseed_split(seed) for the per-seed split refactor used by
          the Nadeau-Bengio paired t-test.
    """

    def __init__(
        self,
        cache_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        eager_load: bool = True,
        split_mode: str = "random",
        chrono_gap_windows: int = 0,
    ) -> None:
        super().__init__()
        if split_mode not in ("random", "chronological"):
            raise ValueError(f"split_mode must be 'random' or 'chronological', got {split_mode!r}")
        self.cache_dir = Path(cache_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        # Eager loading shares tensors with workers via fork+COW. Workers
        # still help by collating in parallel with the GPU step, so we
        # respect the caller's ``num_workers`` choice regardless.
        self.num_workers = num_workers
        self.eager_load = eager_load
        self.seed = seed
        self.split_mode = split_mode
        self.chrono_gap_windows = int(chrono_gap_windows)

        self._dataset: Dataset | None = None
        self._train: Subset | None = None
        self._val: Subset | None = None
        self._test: Subset | None = None

    # ---- subclass hooks ------------------------------------------------

    @property
    @abstractmethod
    def expert_names(self) -> list[str]:
        """Stable ordered list of K expert names. Used everywhere."""

    @abstractmethod
    def config_payload(self) -> dict[str, Any]:
        """Dict identifying this cache (used to compute the manifest hash)."""

    @abstractmethod
    def build_cache(self) -> None:
        """Materialise ``self.cache_dir`` and write its manifest."""

    @abstractmethod
    def _make_dataset(self) -> Dataset:
        """Return a torch Dataset over the full cache."""

    # ---- lifecycle -----------------------------------------------------

    def prepare_data(self) -> None:
        """Build the cache if absent or stale. Called once per node."""
        manifest_path = self.cache_dir / "manifest.json"
        if CacheManifest.check(manifest_path, self.config_payload()):
            return
        self.build_cache()

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        if self._dataset is not None:
            return
        # In case prepare_data was skipped (single-process notebooks) or the
        # on-disk manifest is stale (e.g. config_payload changed since the
        # last build — common when ``keep_experts`` is added to subset the
        # encoder pool, which shifts the aggregate hash).
        if not CacheManifest.check(self.cache_dir / "manifest.json", self.config_payload()):
            self.build_cache()
        self._dataset = self._make_dataset()
        self.reseed_split(self.seed)

    def reseed_split(self, split_seed: int) -> None:
        """Re-derive the train/val/test partition for the requested seed.

        Two modes:

        * ``split_mode="random"`` (default) — random permutation of window
          indices, seeded by ``split_seed``. Requires non-overlapping
          windows (stride = horizon) to avoid leakage.
        * ``split_mode="chronological"`` — contiguous slice of windows in
          time order, with ``chrono_gap_windows`` dropped between the
          train→val and val→test boundaries to keep every val/test
          window's forecast region disjoint from any training window's
          forecast region. The partition is *independent* of
          ``split_seed`` in this mode.
        """
        if self._dataset is None:
            self.setup()
            return
        n = len(self._dataset)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        if self.split_mode == "chronological":
            gap = self.chrono_gap_windows
            train_end = n_train
            val_start = train_end + gap
            val_end = val_start + n_val
            test_start = val_end + gap
            if test_start >= n:
                raise ValueError(
                    f"chronological split has no test windows: "
                    f"n={n}, n_train={n_train}, gap={gap}, n_val={n_val}"
                )
            train_idx = list(range(0, train_end))
            val_idx = list(range(val_start, val_end))
            test_idx = list(range(test_start, n))
        else:
            rng = np.random.default_rng(int(split_seed))
            idx = rng.permutation(n).tolist()
            train_idx = idx[:n_train]
            val_idx = idx[n_train : n_train + n_val]
            test_idx = idx[n_train + n_val :]
        self._train = Subset(self._dataset, train_idx)
        self._val = Subset(self._dataset, val_idx)
        self._test = Subset(self._dataset, test_idx)
        self.seed = int(split_seed)

    # ---- loaders -------------------------------------------------------

    def _loader(self, subset: Subset, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test, shuffle=False)

    # ---- diagnostics ---------------------------------------------------

    @property
    def class_priors(self) -> list[float]:
        """Oracle argmin-class distribution on the training split.

        Used by ``class_balanced_loss`` to inverse-weight the hard CE loss.
        """
        errs = self.train_error_matrix
        best = errs.argmin(axis=-1)
        k = len(self.expert_names)
        counts = np.bincount(best, minlength=k).astype(float)
        total = counts.sum()
        return (counts / total).tolist() if total > 0 else [1.0 / k] * k

    @property
    def train_error_matrix(self) -> np.ndarray:
        """``(N_train, K)`` error matrix; used by inverse-MASE baselines."""
        self._ensure_setup()
        em = getattr(self._dataset, "error_matrix", None)
        if em is not None:
            if hasattr(em, "numpy"):
                em_np = em.detach().cpu().numpy()
            else:
                em_np = np.asarray(em)
            idx = np.asarray(self._train.indices, dtype=np.int64)  # type: ignore[union-attr]
            return em_np[idx]
        # Fallback path for datasets that don't expose ``error_matrix`` directly.
        rows = [
            np.asarray(self._dataset[i]["error_matrix"])
            for i in self._train.indices  # type: ignore[union-attr]
        ]
        return np.stack(rows, axis=0)

    def _ensure_setup(self) -> None:
        if self._dataset is None:
            self.setup()
