## MLflow Run Hierarchy

```
experiment   : tafs
parent run   : <dataset>_<version>    e.g. synth_v1, tng_v1
child run    : <method_name>          e.g. tafs_affine, mlp_ensemble, lgbm
grandchild   : seed_<n>_split_<n>    e.g. seed_0_split_0
```

Per-child aggregation logs `mean(metric)` and `metric__seed_sem` across grandchild runs.

View results:
```bash
make mlflow_ui     # opens mlflow UI at localhost:5000
```
