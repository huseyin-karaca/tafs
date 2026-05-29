# TAFS — Transformer-Attentive Feature Selection

Implementation of the TAFS paper: a feature-axis transformer that learns
context-conditioned combination weights over a pool of frozen base forecasters.
Each engineered feature is treated as a token; a CLS-based attention readout
produces time-varying combination weights under unconstrained / affine / convex
constraints.

The full paper draft is at `reports/tafs-draft.pdf`. The architecture spec is
at `.claude/specs/tafs.md`.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
**Do not use `pip` directly** — always go through `uv` so `pyproject.toml`
stays in sync.

```bash
# 1. Install uv (if you don't have it)
curl -Lss https://astral.sh/uv/install.sh | sh

# 2. Create a virtual environment and install all dependencies
uv venv --python 3.10
source .venv/bin/activate
uv sync

# 3. To add a new package
uv add <package-name>         # adds to pyproject.toml and installs
uv add --dev <package-name>   # dev-only dependency
```

---

## Quick start

```bash
# Verify everything works (builds a tiny synthetic cache, runs baselines)
make smoke

# Run the full synthetic experiment (5 seeds, all methods)
make run CONFIG=tafs/synthetic

# Override any config value inline
python run.py experiment=tafs/synthetic trainer.max_epochs=10

# View results in MLflow UI
make mlflow_ui      # then open http://localhost:5000
```

---

## Project layout

```
run.py                    Single Hydra entry point
Makefile                  Wraps run.py; see `make help`
configs/
  config.yaml             Root config — defaults and top-level overrides
  experiment/tafs/        One YAML per experiment (synthetic, real datasets)
  data/tafs/              One YAML per dataset
  trainer/default.yaml    PyTorch Lightning Trainer defaults
  mlflow/local.yaml       MLflow tracking URI
src/
  core/                   Shared infrastructure (ABCs, data, stats, utils)
  tafs/                   TAFS-specific code
    models/tafs.py        Main model  ← implement here
    models/baselines.py   Training-free baselines (already implemented)
    models/static_transforms.py  Ablation baselines
    models/meta_learners.py      Non-attention ensemble combiners
    data/synthetic.py     Synthetic data generator  ← implement here
    data/datamodule.py    LightningDataModule for TAFS
    data/feature_pipeline.py  Real-dataset feature engineering  ← implement here
    experiments/synthetic.py    Synthetic experiment driver
    experiments/real_world.py   Real-world experiment driver
tests/
  core/                   Tests for shared infrastructure
  tafs/                   Tests for TAFS-specific modules
notebooks/
  getting_started.ipynb   Interactive demo of imports and the data pipeline
reports/
  tafs-draft.pdf          Paper draft (source of truth for architecture)
  tafs/                   LaTeX source for the paper
```

---

## How Hydra configs work

Hydra composes configs from multiple files and lets you override any value
on the CLI. The root config is `configs/config.yaml`. It pulls in:

- `experiment/tafs/<name>.yaml` — which experiment class to run and all its hyperparameters
- `data/tafs/<name>.yaml` — dataset config, referenced by the experiment YAML
- `trainer/default.yaml` — Lightning Trainer settings
- `mlflow/local.yaml` — where to store experiment results

Override anything inline without editing files:

```bash
# Change max epochs only for this run
python run.py experiment=tafs/synthetic trainer.max_epochs=50

# Change batch size defined in the data config
python run.py experiment=tafs/synthetic datamodule_cfg.batch_size=128
```

Every run saves the fully resolved config under `outputs/<date>/<time>/.hydra/`.

---

## Adding a new dataset

1. Create `configs/data/tafs/<dataset>.yaml` pointing at `TAFSRealDataModule`
   with your dataset's cache directory and expert names list.
2. Create `configs/experiment/tafs/<dataset>.yaml` referencing that data config.
3. Implement `build_cache()` in `TAFSRealDataModule` (or subclass it) to run
   the feature pipeline and base predictors on your raw data.
4. Run `make prepare_data DATASET=<dataset>` to build the cache.
5. Run `make run CONFIG=tafs/<dataset>`.

---

## Development workflow

```bash
make test      # run all tests
make lint      # check formatting and style
make format    # auto-fix formatting
make clean     # remove __pycache__, .pytest_cache, outputs/
```

Tests are in `tests/`. The smoke test (`tests/test_smoke.py`) must pass after
any change to `src/`. The forward-pass tests (`tests/tafs/test_tafs_forward.py`)
will pass once `TAFS.forward()` is implemented.
