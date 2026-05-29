"""Synthetic regime-switch data generator for TAFS (Section IV-D-1).

Goal: a tractable testbed where the optimal feature subset is different
in each regime, so a context-conditioned attention mechanism should
strictly outperform static feature selectors.

Setup
-----
Each trajectory has T = 400 time steps drawn from a two-regime ARMA(2,1).

    Regime A (t = 0 .. 199): y_t depends on lag-1 and rolling-7 mean.
    Regime B (t = 200 .. 399): y_t depends on two exogenous series.

A total of p = 32 features are engineered per step:
    - 4 "signal" features (lag-1, rolling-7, exog-1, exog-2)
    - 28 nuisance features (other lags, rolling stats at irrelevant windows)

The K = 3 base predictors are:
    - LagPredictor     (good in regime A; uses lag features)
    - ExogPredictor    (good in regime B; uses exog features)
    - AveragePredictor (mediocre in both; simple moving average)

The error matrix entry error_matrix[t, k] = (y_t - ŷ_t^k)^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


BASE_PREDICTOR_NAMES: tuple[str, ...] = (
    "lag_predictor",
    "exog_predictor",
    "average_predictor",
)

FEATURE_NAMES: tuple[str, ...] = (
    "lag_1",
    "rolling_7",
    "exog_1",
    "exog_2",
    # 28 nuisance features
    *[f"nuisance_{i}" for i in range(28)],
)

FEATURE_FAMILIES: tuple[str, ...] = (
    "lag",       # lag_1
    "rolling",   # rolling_7
    "exogenous", # exog_1
    "exogenous", # exog_2
    *["lag"] * 14,
    *["rolling"] * 14,
)


@dataclass
class SyntheticConfig:
    """Configuration for the synthetic TAFS dataset."""

    n_trajectories: int = 500
    T: int = 400
    regime_change: int = 200  # index where regime switches from A to B
    noise_std: float = 0.3
    seed: int = 0


def generate_dataset(cfg: SyntheticConfig) -> dict[str, np.ndarray]:
    """Generate the full synthetic dataset.

    Returns a dict with:
        feature_matrix:  (N, p) float32 — one row per time step
        base_preds:      (N, K) float32 — per-step base predictor outputs
        targets:         (N,)   float32 — ground-truth y_t
        error_matrix:    (N, K) float32 — (y_t - ŷ_t^k)^2
        regime:          (N,)   int8    — 0 = regime A, 1 = regime B
        step_index:      (N,)   int64   — time index within its trajectory
    """
    # TODO: implement the two-regime ARMA(2,1) data generating process.
    #
    # Suggested approach:
    #
    #   rng = np.random.default_rng(cfg.seed)
    #   For each trajectory i in range(cfg.n_trajectories):
    #     1. Generate exogenous series exog_1, exog_2  (e.g. AR(1) processes)
    #     2. Generate y_t for t = 0..T-1:
    #        Regime A (t < cfg.regime_change):
    #            y_t = 0.7 * y_{t-1} + 0.3 * rolling_7(y) + noise
    #        Regime B (t >= cfg.regime_change):
    #            y_t = 0.5 * exog_1_t + 0.5 * exog_2_t + noise
    #     3. Build feature vector x_t of shape (p,):
    #        [lag_1, rolling_7, exog_1, exog_2, nuisance_0, ..., nuisance_27]
    #     4. Compute base predictor outputs:
    #        lag_predictor:     ŷ_t = y_{t-1}
    #        exog_predictor:    ŷ_t = 0.5 * exog_1_t + 0.5 * exog_2_t
    #        average_predictor: ŷ_t = rolling_7(y)
    #     5. Compute error_matrix: (y_t - ŷ_t^k)^2
    #
    #   Collect and stack across all steps across all trajectories.
    #   Shuffle rows (use rng) for the random split to work correctly.
    raise NotImplementedError
