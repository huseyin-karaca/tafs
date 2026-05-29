"""Meta-learner baselines without feature selection (Section IV-B).

These combiners use the raw feature vector (or a simple summary) to
learn combination weights over the K base predictors, but they do NOT
apply a feature-axis transformer. They establish the upper performance
bound for "ensemble combination without attention-based feature selection."

Classes
-------
MLPEnsembleRouter    — MLP input -> affine-constrained combination weights.
LGBMEnsembleRouter   — LightGBM-based combination (gradient-boosted trees
                       trained on the feature matrix to predict per-step
                       base-predictor errors, then converted to weights).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.core.models.base import TrainableLightningRouter, TrainingFreeBaseline


class MLPEnsembleRouter(TrainableLightningRouter):
    """Two-layer MLP: raw features -> affine combination weights.

    This is the "MLP ensemble (affine)" baseline. It takes the same
    (B, p) feature matrix as TAFS but maps directly to weights without
    the transformer or context token. No feature selection — all p
    features are treated equally.

    Args:
        num_features: p input features.
        num_experts: K base predictors.
        hidden_dim: MLP hidden width.
        constraint: 'unconstrained' | 'affine' | 'convex'.
        **kwargs: forwarded to TrainableLightningRouter.
    """

    def __init__(
        self,
        num_features: int,
        num_experts: int,
        hidden_dim: int = 256,
        constraint: str = "affine",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # TODO: implement nn.Sequential MLP + constraint head.
        # Architecture: Linear(p, hidden) -> GELU -> Linear(hidden, K)
        # then apply the same constraint projection as TAFS CombinationHead.
        raise NotImplementedError

    def forward(self, batch: dict) -> torch.Tensor:
        """Return (B, K) combination weights from batch["feature_matrix"]."""
        # TODO
        raise NotImplementedError


class LGBMEnsembleRouter(TrainingFreeBaseline):
    """LightGBM-based combiner trained on features to predict base errors.

    Training procedure:
        1. For each base predictor k, train a LightGBM regressor on
           (feature_matrix, error_matrix[:, k]) on the training split.
        2. At inference, predict per-predictor errors, then convert to
           softmax weights (convex constraint).

    This is a training-free baseline in the sense that it does not use
    gradient descent, but it does require a fit() call that runs
    LightGBM training.

    Args:
        num_experts: K base predictors.
        constraint: 'convex' (default) — only softmax makes sense here.
        lgbm_kwargs: passed to lightgbm.LGBMRegressor.
    """

    def __init__(
        self,
        num_experts: int,
        constraint: str = "convex",
        lgbm_kwargs: dict | None = None,
    ) -> None:
        self.num_experts = num_experts
        self.constraint = constraint
        self.lgbm_kwargs = lgbm_kwargs or {}
        self._models: list | None = None  # populated by fit()

    def fit(self, datamodule, trainer_cfg=None, mlflow_run_id=None) -> None:  # type: ignore[override]
        """Train one LightGBM regressor per base predictor on the training split.

        Args:
            datamodule: TAFSDataModule with train_dataloader().
        """
        # TODO:
        # 1. Collect feature_matrix and error_matrix from train_dataloader().
        # 2. For k in range(K): fit LGBMRegressor on (features, errors[:, k]).
        # 3. Store fitted models in self._models.
        raise NotImplementedError

    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Return (B, K) combination weights from LightGBM predictions."""
        # TODO:
        # 1. Run each LGBMRegressor on batch["feature_matrix"].numpy().
        # 2. Stack predictions -> (B, K) predicted errors.
        # 3. Convert to weights via softmax(-pred_errors).
        raise NotImplementedError
