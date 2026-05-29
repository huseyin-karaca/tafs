# Architecture & Layout

```
run.py                         # single Hydra entry point (~50 lines)
Makefile                       # wraps run.py + data prep + lint + test
configs/
  config.yaml                  # root: defaults list + _self_
  experiment/tafs/<name>.yaml  # one file per reported experiment
  data/tafs/<dataset>.yaml     # one file per dataset
  trainer/default.yaml
  mlflow/local.yaml
  logging/default.yaml
src/
  core/                        # paper-agnostic infrastructure
    experiments/base.py        # BaseExperiment (parent/child/grandchild orchestration)
    models/base.py             # BaseRouter + TrainableLightningRouter + TrainingFreeBaseline
    data/base.py               # BaseDataModule + CacheManifest
    utils/                     # logging, metrics, git, seeding, mlflow_setup, progress
    stats/                     # paired_t (Nadeau-Bengio test)
    reporting/                 # Report, io
  tafs/
    models/                    # tafs.py, static_transforms.py, meta_learners.py, baselines.py
    data/                      # datamodule.py, feature_pipeline.py, synthetic.py
    experiments/               # synthetic.py, real_world.py
tests/
  core/                        # tests for shared infra
  tafs/                        # tests for TAFS-specific modules
```

**Required:**
- `run.py` is ~50 lines: setup logging → set float32 precision → setup MLflow → `instantiate(cfg.experiment).run(cfg.experiment)`.
- Every `core/` subpackage exposes ABCs in `base.py`. `tafs/` subclasses them; it does not invent parallel hierarchies.
- All concrete classes are constructed via `hydra.utils.instantiate(cfg)` — never instantiated by hand in `run.py`.

**Import rules (enforced):**
- `tafs → core` ✓
- `core → tafs` ✗

**New files outside this layout require justification in a comment or PR description.**
