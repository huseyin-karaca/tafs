# Combiner Abstractions

TAFS selects/combines among `K` pre-trained, frozen base predictors. The ABCs live in `src/core/models/base.py`; concrete subclasses live in `src/tafs/models/`.

```
BaseRouter (ABC)                              # src/core/models/base.py
├── TrainableLightningRouter (pl.LightningModule)
│       fit() runs pl.Trainer; predict_proba/select/evaluate defined here
└── TrainingFreeBaseline
        fit() = no-op; select() implemented by subclass
```

**Required contract:**
- `predict_proba(batch) -> torch.Tensor` of shape `(B, K)`.
- `select(batch) -> np.ndarray` of shape `(B,)`. Default = `predict_proba.argmax(-1)`.
- `evaluate(datamodule) -> dict` returns all metrics; trainable subclasses reuse cached `(selected_idx, error_matrix)` buffered in `on_test_start` / `test_step`.

**Batch keys used by TAFS:**
- `feature_matrix`: `(B, p)` — engineered features at each time step.
- `context`: `(B, C)` — context summary s_t (lagged targets + seasonal residuals + optional regime flags).
- `base_preds`: `(B, K)` — frozen base predictor forecasts ŷ_t.
- `error_matrix`: `(B, K)` — per-step squared error per base predictor (used for loss and evaluation).

**TAFS is a combiner, not a hard router.** `predict_proba` returns the continuous weights w_t. `select` (argmax) is only meaningful in the convex regime. `evaluate` reports MSE/MASE of the weighted combination `ŷ_t = w_t^T ŷ_base_t`, not selection accuracy.
