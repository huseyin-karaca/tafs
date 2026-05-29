# MLflow & Reproducibility

**Per-parent-run repro artefacts** (logged by `BaseExperiment` at parent open):
- `repro/git_state.json` — commit SHA, branch, dirty flag.
- `repro/git_dirty.patch` — `git diff HEAD` (only when dirty).
- `repro/resolved_config.yaml` — `OmegaConf.to_yaml(cfg, resolve=True)`.

**Default MLflow tracking URI** is read from `configs/mlflow/local.yaml`; never hard-code it.

**Aggregation:** across-seed mean and SEM (`metric__seed_sem`) logged on the child run. Per-seed metrics live only on grandchild runs.
