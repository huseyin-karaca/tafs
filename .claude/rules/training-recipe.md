# Training Recipe

All trainable combiners use the same scaffold (`src/core/models/base.py:TrainableLightningRouter`):

**Optimizer / schedule.** AdamW, `lr=1e-3`, `weight_decay=1e-2`; 200 linear-warmup steps then cosine annealing to 0. Implemented via `LambdaLR` at per-step interval.

**Loss.** Primary: mean squared error on the weighted combination output `ŷ_t = w_t^T ŷ_base_t`. Auxiliary hard/soft CE terms (from `TrainableLightningRouter._compute_loss`) provide a routing supervision signal and are controlled by `primary_weight`, `hard_ce_weight`, `soft_ce_weight`.

**Checkpointing.** Use `tempfile.TemporaryDirectory()` as `dirpath` for `ModelCheckpoint`; monitor `val/L_mase` (or `val/loss`), `mode="min"`, `save_top_k=1`. `EarlyStopping` only when `early_stopping_patience is not None`.

**Lightning logger.** `MLFlowLogger(run_id=mlflow_run_id, tracking_uri=mlflow.get_tracking_uri())`. The `run_id` is the **grandchild** seed run, passed in by `BaseExperiment`.

**Precision.** `cfg.float32_matmul_precision` (`"high"` default) is honoured at the top of `run.py`; do not call `torch.set_float32_matmul_precision` elsewhere.

**Do not** instantiate `pl.Trainer` outside of `fit()`. Trainer config flows in as `trainer_cfg: DictConfig` and is `instantiate`-d after callbacks are stitched in.
