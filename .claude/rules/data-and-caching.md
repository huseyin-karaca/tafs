# Data & Caching

**Two-stage caching** (`BaseDataModule` + `CacheManifest` live in `src/core/data/base.py`; TAFS-specific extraction lives in `src/tafs/data/`):

1. **Raw → interim parquet.** Downloaded / extracted series stored under `data/interim/<dataset>/`. Materialised by `make prepare_data DATASET=<name>` → `python -m src.tafs.data.prepare --dataset <name>`.
2. **Interim → processed cache.** Cache the engineered feature matrix `X ∈ ℝ^{N × p}`, the `N × K` matrix of base-predictor outputs `Ŷ`, and the `N × K` squared-error matrix under `data/processed/tafs/<dataset>/`.

**Manifest pattern.** Every cache directory carries a `manifest.json` recording: dataset name, version hash, feature pipeline version, creation timestamp, git SHA, and sample count. The `DataModule` refuses to load a cache whose manifest hash does not match the requested config.

**LightningDataModule contract** (`src/core/data/base.py:BaseDataModule`; `src/tafs/data/datamodule.py` subclasses it):
- `setup(stage)` materialises splits and exposes `class_priors: list[float]`.
- Batches are `dict[str, Tensor]` with keys: `feature_matrix`, `base_preds`, `target`, `error_matrix`, `primary_regime`.
- `eager_load: bool` controls in-RAM vs. lazy loading.

**Do not download inside training code.** All network calls live under `src/tafs/data/` and run only when `prepare_data` is invoked.
