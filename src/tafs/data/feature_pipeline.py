"""Feature engineering pipeline for real TAFS datasets.

Given a raw time series (or a DataFrame of multivariate series), this
module produces the engineered feature matrix X in R^{T x p} used by
the TAFS model. Feature families (from Section III-A of the manuscript):

    lag:        y_{t-1}, y_{t-2}, ..., y_{t-L}
    rolling:    rolling mean/median/std at windows {3, 7, 14, 28}
    calendar:   day-of-week, month, hour, holiday flag (dataset-dependent)
    exogenous:  available covariates lagged by the same horizons as lag
    regime:     optional external regime indicator flags

Entry point
-----------
Call ``build_feature_matrix(series, cfg)`` to produce:
    - ``X``:           (T, p) feature matrix
    - ``feature_names: list[str]`` of length p (for the tokeniser)
    - ``families``:    list[str] of length p (family label per feature)

The output is saved to the dataset cache by ``make prepare_data``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeatureConfig:
    """Controls which feature families and hyperparameters are used.

    Args:
        lag_max: maximum lag order (default 24 for low-frequency data).
        rolling_windows: windows (in time steps) for rolling statistics.
        include_calendar: whether to include calendar features.
        include_exogenous: whether to include exogenous covariates.
        target_col: name of the target column in the input DataFrame.
        exog_cols: list of exogenous covariate column names.
    """

    lag_max: int = 24
    rolling_windows: list[int] = field(default_factory=lambda: [3, 7, 14, 28])
    include_calendar: bool = True
    include_exogenous: bool = False
    target_col: str = "y"
    exog_cols: list[str] = field(default_factory=list)


def build_lag_features(series: pd.Series, lag_max: int) -> pd.DataFrame:
    """Build lag features y_{t-1}, ..., y_{t-lag_max}.

    Args:
        series: target time series (DatetimeIndex or integer index).
        lag_max: maximum lag order.
    Returns:
        DataFrame of shape (T, lag_max) with columns lag_1, lag_2, ...
    """
    # TODO: use pd.DataFrame({f"lag_{k}": series.shift(k) for k in range(1, lag_max + 1)})
    raise NotImplementedError


def build_rolling_features(series: pd.Series, windows: list[int]) -> pd.DataFrame:
    """Build rolling mean/median/std features.

    Args:
        series: target time series.
        windows: list of window sizes (in time steps).
    Returns:
        DataFrame with columns rolling_mean_3, rolling_std_3, ..., etc.
    """
    # TODO: for each window w, compute series.rolling(w).mean(), .std()
    raise NotImplementedError


def build_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build calendar features from a DatetimeIndex.

    Features: day_of_week, month, hour (if sub-daily), is_weekend.
    All values are encoded as integers (no one-hot encoding — the
    feature-axis transformer handles the embedding).
    """
    # TODO
    raise NotImplementedError


def build_feature_matrix(
    df: pd.DataFrame,
    cfg: FeatureConfig,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Assemble the full feature matrix X in R^{T x p}.

    Args:
        df: DataFrame with at minimum a column named cfg.target_col.
            If include_calendar=True the index must be a DatetimeIndex.
        cfg: FeatureConfig specifying which features to build.

    Returns:
        X:            (T, p) float32 array (NaN rows from lag/rolling
                      are dropped; the first lag_max rows are excluded).
        feature_names: list of p feature names (e.g. "lag_1", "rolling_mean_7").
        families:     list of p family labels ("lag", "rolling", "calendar",
                      "exogenous") — used by the TAFS tokeniser for anchor
                      initialisation.
    """
    # TODO:
    # 1. Build each feature group using the helpers above.
    # 2. Concatenate into a single DataFrame.
    # 3. Drop NaN rows introduced by lags/rolling (first lag_max rows).
    # 4. Return X as float32 numpy, plus feature_names and families lists.
    raise NotImplementedError
