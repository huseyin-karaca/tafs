"""Training-free baseline combiners for TAFS (Section IV-B).

These are the reference points that require no gradient-based training.

Classes
-------
UniformEnsemble     — equal weights over all K base predictors (simple average).
OracleRouter        — per-step hindsight best predictor (upper bound, not in competition).
RandomRouter        — random uniform selection per step.
BestSinglePredictor — at inference time, always select the predictor that had
                      the lowest average error on the training split.
"""

from __future__ import annotations

import numpy as np
import torch

from src.core.models.base import TrainingFreeBaseline


class UniformEnsemble(TrainingFreeBaseline):
    """Equal combination weights: w_t = 1/K for all t.

    Corresponds to simple averaging of all K base predictor forecasts.

    Args:
        num_experts: K base predictors.
    """

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts

    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Return (B, K) uniform weights."""
        b = batch["error_matrix"].size(0)
        return torch.full((b, self.num_experts), 1.0 / self.num_experts)


class OracleRouter(TrainingFreeBaseline):
    """Hindsight oracle: per-step argmin over the error matrix.

    Not a real method — acts as a strict upper bound in the results
    tables. Uses ground-truth errors at test time.

    Args:
        num_experts: K base predictors.
    """

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts

    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Return one-hot weights at the oracle (argmin) predictor."""
        best = batch["error_matrix"].argmin(dim=-1)
        probs = torch.zeros_like(batch["error_matrix"])
        probs.scatter_(1, best.unsqueeze(1), 1.0)
        return probs


class RandomRouter(TrainingFreeBaseline):
    """Uniform random selection: assigns full weight to one random predictor.

    Args:
        num_experts: K base predictors.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, num_experts: int, seed: int = 0) -> None:
        self.num_experts = num_experts
        self._rng = np.random.default_rng(seed)

    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Return (B, K) one-hot weights sampled uniformly."""
        b = batch["error_matrix"].size(0)
        chosen = self._rng.integers(0, self.num_experts, size=b)
        probs = torch.zeros(b, self.num_experts)
        probs[torch.arange(b), torch.from_numpy(chosen)] = 1.0
        return probs


class BestSinglePredictor(TrainingFreeBaseline):
    """Always use the single predictor with the lowest mean training error.

    Requires a fit() call to compute per-predictor training means.

    Args:
        num_experts: K base predictors.
    """

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts
        self._best_k: int = 0

    def fit(self, datamodule, trainer_cfg=None, mlflow_run_id=None) -> None:  # type: ignore[override]
        """Select k* = argmin_k mean_train(error_matrix[:, k])."""
        # TODO: use datamodule.train_error_matrix to find self._best_k.
        raise NotImplementedError

    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Return (B, K) one-hot weights always pointing to self._best_k."""
        b = batch["error_matrix"].size(0)
        probs = torch.zeros(b, self.num_experts)
        probs[:, self._best_k] = 1.0
        return probs
