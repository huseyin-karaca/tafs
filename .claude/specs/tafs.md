# Paper 1 — TAFS: Transformer-Attentive Feature Selection for Sequential Regression

**Source of truth:** `reports/tafs-draft.pdf` (DRAFT, MAY 17 2026). This spec is the implementation-facing mirror of that PDF; the coder agent must not diverge from it.

## Core sell
In high-dimensional, data-scarce sequential regression (hundreds of engineered lags / rollings / calendar / exogenous features against a few hundred to a few thousand observations), feature relevance is **time-varying and context-dependent**. TAFS treats each engineered feature as a token and applies a feature-axis transformer conditioned on a **context summary of the recent history**, then reads out **combination weights over a fixed pool of pre-trained, frozen base forecasters** under one of three constraint regimes (unconstrained / affine / convex). The attention mass on each feature token is itself the time-varying importance signal. The pipeline is trained end-to-end on cumulative prediction error.

## Position
- **vs. FASTT.** TAFS replaces FASTT's static gating (diagonal / linear / low-rank / nonlinear) with context-conditioned attention. The four static transforms are recovered as strict special cases when the context token is removed and queries/keys are tied to the identity (Proposition in Section III).
- **vs. Karaca SPL 2026 (BSDT-Q).** SPL uses a fixed linear `Q` per boosting round. Here `Q` becomes a context-dependent attention matrix `Q(s_t)`.
- **vs. Arda 2024.** Arda combines base predictors with time-varying weights but performs no feature selection. TAFS does both jointly.

## Architecture (frozen by the manuscript)

1. **Base predictor pool** (`K = 9`, all frozen and cached offline): LightGBM, XGBoost, CatBoost, RandomForest, Lasso, MLP, LSTM, SARIMAX, Naive. Each produces a one-step prediction `ŷ_t^(k) = f_k(x_t)`; the H-step variant uses native multi-output where available, direct per-horizon heads otherwise (no recursive forecasting).
2. **Feature-axis tokenization.** For feature `j` at step `t`: `t_{t,j} = a_j · x_{t,j} + e_j ∈ ℝ^d`, where `a_j ∈ ℝ^d` is a learnable per-feature scaler and `e_j ∈ ℝ^d` is a learnable feature-identity embedding. Features are partitioned into named families `{lag, rolling, calendar, exogenous, regime}`; each `e_j` is **initialised to a family-specific anchor plus small Gaussian noise**.
3. **Context token.** `c_t = W_c s_t + b_c` where `s_t` stacks (a) `L_y` lags of the target, (b) residuals of `y_t` against a simple seasonal-naive predictor, (c) optional regime-indicator flags (weekday vs. weekend, trading vs. non-trading, etc.).
4. **CLS token.** `c_cls ∈ ℝ^d` (learnable) is prepended as the readout position. Input sequence: `Z_t^(0) = (c_cls, c_t, t_{t,1}, …, t_{t,p}) ∈ ℝ^{(p+2)×d}`.
5. **Feature-axis transformer.** `L = 3` pre-norm encoder layers, `h = 4` heads, hidden width `d = 128`, FFN width `d_ffn = 512`, dropout `0.1`. **No positional encoding** — the feature axis is permutation-invariant and the identity embeddings already carry token identity. Per-CLS-to-feature attention weights `α_{t,j}` (averaged over layers/heads) are exported as the interpretability signal.
6. **Combination head.** From the CLS output `u_t = Z_t^(L)[0, :]`, a two-layer MLP produces `z_t = W_2 GELU(W_1 u_t + b_1) + b_2 ∈ ℝ^K`. Three constraint regimes project `z_t` to `w_t ∈ W`:
   - **Unconstrained:** `w_t = z_t`.
   - **Affine:** `w_t = z_t - (1/K)(1ᵀz_t − 1)1` (orthogonal projection onto `1ᵀw = 1`).
   - **Convex:** `w_t = softmax(z_t / τ)` with `τ = 1`.
   The default reported configuration is **affine**.
7. **Prediction & loss.** `ỹ_t = w_tᵀ ŷ_t`. Train end-to-end on `L = (1/T) Σ ℓ(y_t, ỹ_t)` with `ℓ = squared error` on all real datasets. Huber and pinball are exposed by the framework but **not** exercised in the reported experiments.
8. **Multi-step.** Framework supports H-step natively (weights shared across horizons, loss averaged over `H`). All reported experiments are one-step except where the dataset's native horizon dictates otherwise (see Table I in PDF).

## Training recipe

- AdamW, lr `1e-3`, weight decay `1e-2`, batch size `64`, gradient clipping `1.0`, 200 linear-warmup steps then cosine annealing, early stopping on validation loss with patience `15` epochs.
- 60/20/20 chronological train/val/test split. Standardisation fit on the training window only and applied to val/test.
- 5 random seeds per configuration; mean and parenthesised standard deviation reported.
- Approximate trainable parameter count: ~0.6 M (same order as the MLP and LGBM ensemble combiners, 0.1–0.4 M).
- Inference cost is dominated by the cached base predictions (>90% of wall-clock); the TAFS forward pass adds at most ~10% overhead on the studied datasets.

## Datasets (exactly as in Table I of the PDF)

| Dataset | Length | `p` | `H` | Domain |
|---|---|---|---|---|
| Turkish Natural Gas (TNG) | 1096 | 84 | 1 | energy |
| Istanbul Stock Exchange (ISE) | 536 | 76 | 1 | finance |
| M5 Walmart (subset) | 1941 | 112 | 28 | retail |
| M4 Quarterly (avg.) | 92 | 64 | 8 | economic |
| Weather (10-min) | 52696 | 96 | 96 | meteorology |
| Exchange Rate | 7588 | 72 | 96 | finance |
| ETTm2 | 69680 | 88 | 96 | energy |

Feature pipeline per dataset: lags up to `L_max = 24` for low-frequency, up to `L_max = 168` for high-frequency; rolling means/medians/stds at windows `{3, 7, 14, 28}`; calendar variables (day-of-week, month, hour, holiday flag); available exogenous series lagged by the same horizons.

## Synthetic regime-switch task

`N = 500` trajectories of length `T = 400` from a two-regime ARMA(2,1). First half (regime A): `y_t` depends only on lag-1 and a rolling-7 statistic plus noise. Second half (regime B): `y_t` depends on two exogenous series. The remaining `p − 4 = 28` engineered features are pure nuisance. Change point at `t = 200`, hidden from every model. **MSE is reported in place of MASE** here because the stationary-noise denominator is undefined.

## Baselines (exactly the table in Section IV-B)

- **Individual base predictors (9):** LightGBM, XGBoost, CatBoost, RandomForest, Lasso, MLP, LSTM, SARIMAX, Naive. Each tuned on the same validation window with a fixed search budget.
- **Classical feature selection + regressor (3):** `Pearson + LGBM`, `MutualInformation + LGBM`, `SelectFromModel + LGBM`. (Wrapper methods are not run.)
- **Static learned feature transforms (4):** Diagonal gating, full linear, low-rank with `r = 16`, small nonlinear MLP bottleneck. Each paired with the same base-predictor pool combined under the affine head. These differ from TAFS *only* in that the transform is static.
- **Meta-learners without feature selection (2):** MLP-ensemble (affine), LGBM-ensemble (convex). Same base pool.
- **Conventional ensemble baselines (2):** simple average, equal-weight stacking.
- **Per-step oracle:** at each `t`, the lowest-error base predictor; upper bound, not in competition.

## Reported experiments

1. **Synthetic regime-switch demonstration** (Section IV-D-1). Figure 2 (attention mass over the three feature groups across `t`), Figure 3 (per-step attention heatmap), Table II (MSE table including `TAFS (no context)` and `TAFS (full)`).
2. **Real-data comparison** (Section IV-D-2). Table III (MASE on all 7 datasets, all baselines, three TAFS constraint regimes, oracle), Table IV (sMAPE for representative competitors).
3. **Central architectural ablation** (Table V). Replace the feature-axis transformer with each of the four static transforms keeping head/loss/recipe identical; report aggregate MASE and a "regime-switching subset" MASE (Exchange + ISE + Synthetic) to show that context-dependence is what drives the gain.
4. **Context-conditioning ablation** (Table VI). Strip the context token entirely / lagged-target-only / lagged + calendar / full. Report on TNG, ISE, Exchange, Synthetic.
5. **Constraint-regime spread**: unconstrained / affine / convex on all real datasets; spread ≤ ~1.5% aggregate MASE; affine adopted as default.
6. **Layer / head sweep**: `L ∈ {1,2,3,4}`, `h ∈ {2,4,8}` on the validation split; report Pareto-optimal default.

## MLflow layout (per `experiment-tracking.md`)

- experiment: `tafs`
- parent run: `<dataset>_v1` (one per dataset in Table I) and `synt_v1`
- child run: `<method_name>` (e.g. `tafs_affine`, `lgbm`, `nonlinear_transform`, `mlp_ensemble_affine`)
- grandchild run: `seed_<n>_split_<i>`

## Out of scope

Probabilistic/quantile heads, horizon-conditional weighting, multivariate targets, learned base-predictor pool, and theoretical convergence appendix are not part of this paper.
