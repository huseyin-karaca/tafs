"""Static feature-transform baselines (Table V — architectural ablation).

Each class replaces the feature-axis transformer in TAFS with a
static (non-context-conditioned) alternative, keeping the same
combination head and training recipe. This isolates the contribution
of context-dependent attention.

Hierarchy:
    StaticTransformRouter (TrainableLightningRouter)
        DiagonalGatingRouter    — per-feature scalar gate (p params)
        FullLinearRouter        — unconstrained p x K linear map
        LowRankRouter           — p x r x K (r << p) factored transform
        NonlinearMLPRouter      — small MLP bottleneck feature transform

All four share the same forward signature and plug into the same
BaseExperiment orchestration as TAFS.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.core.models.base import TrainableLightningRouter


class StaticTransformRouter(TrainableLightningRouter):
    """Base for static-transform ablations.

    Subclasses implement :meth:`transform` which maps the raw feature
    vector (B, p) to a pre-combination representation (B, K), which
    is then passed to the same combination head as TAFS.

    Args:
        num_features: p input features.
        num_experts: K base predictors.
        constraint: 'unconstrained' | 'affine' | 'convex'.
        **kwargs: forwarded to TrainableLightningRouter.
    """

    def __init__(
        self,
        num_features: int,
        num_experts: int,
        constraint: str = "affine",
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # TODO: instantiate the combination head (same as TAFS but input is
        # the static transform output, not a CLS embedding from a transformer).
        raise NotImplementedError

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        """Map (B, p) features to (B, d_out) representation.

        Subclasses implement this. The result is passed to the
        combination head.
        """
        raise NotImplementedError

    def forward(self, batch: dict) -> torch.Tensor:
        """Return (B, K) combination weights."""
        # TODO: apply self.transform, then pass through the combination head.
        raise NotImplementedError


class DiagonalGatingRouter(StaticTransformRouter):
    """Per-feature scalar gate — the simplest static baseline.

    Learns p scalars; each feature is multiplied by its gate value
    before a linear combination to weights. Equivalent to a 1-layer
    linear model with diagonal weight matrix.
    """

    def __init__(self, num_features: int, num_experts: int, **kwargs: Any) -> None:
        super().__init__(num_features, num_experts, **kwargs)
        # TODO: nn.Parameter of shape (num_features,), initialised to ones.
        raise NotImplementedError

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        # TODO: element-wise multiply features by the gate.
        raise NotImplementedError


class FullLinearRouter(StaticTransformRouter):
    """Unconstrained p × K linear map over features.

    The most expressive static baseline: a full weight matrix with no
    factorisation or non-linearity.
    """

    def __init__(self, num_features: int, num_experts: int, **kwargs: Any) -> None:
        super().__init__(num_features, num_experts, **kwargs)
        # TODO: nn.Linear(num_features, num_experts, bias=True).
        raise NotImplementedError

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class LowRankRouter(StaticTransformRouter):
    """Low-rank factored transform: p -> r -> K.

    Captures correlations among features without the full O(p*K) param
    count of FullLinearRouter. Default rank r=16.
    """

    def __init__(
        self,
        num_features: int,
        num_experts: int,
        rank: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_features, num_experts, **kwargs)
        # TODO: two nn.Linear layers: (p -> r) then (r -> K).
        raise NotImplementedError

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError


class NonlinearMLPRouter(StaticTransformRouter):
    """Small MLP bottleneck: p -> hidden -> K with a non-linearity.

    Adds depth/non-linearity to the static transform while remaining
    context-free (no CLS or context token).
    """

    def __init__(
        self,
        num_features: int,
        num_experts: int,
        hidden_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_features, num_experts, **kwargs)
        # TODO: nn.Sequential(Linear(p, hidden), GELU(), Linear(hidden, K)).
        raise NotImplementedError

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        # TODO
        raise NotImplementedError
