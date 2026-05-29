"""TAFS model — Transformer-Attentive Feature Selection.

Architecture overview (Section III of the manuscript):

    1. Feature tokenisation:
       For feature j at step t:
           token_{t,j} = a_j * x_{t,j} + e_j  in R^d
       where a_j is a learnable per-feature scaler and e_j is a
       learnable feature-identity embedding, initialised to a
       family-specific anchor (lag / rolling / calendar / exogenous).

    2. Context token:
       c_t = W_c * s_t + b_c  in R^d
       where s_t stacks L_y lagged targets + seasonal-naive residuals
       + optional regime-indicator flags.

    3. CLS token prepended as the readout position. Input sequence:
       Z_t^0 = (cls, c_t, token_{t,1}, ..., token_{t,p})  shape (p+2, d)

    4. Feature-axis transformer (no positional encoding):
       L=3 pre-norm encoder layers, h=4 heads, d=128, d_ffn=512,
       dropout=0.1. Per-CLS-to-feature attention weights alpha_{t,j}
       (averaged over layers/heads) are exported as the interpretability
       signal.

    5. Combination head:
       From CLS output u_t = Z_t^L[0]:
           z_t = W2 * GELU(W1 * u_t + b1) + b2  in R^K
       Projected to w_t under one of three constraint regimes:
           unconstrained : w_t = z_t
           affine        : w_t = z_t - (1/K)(1^T z_t - 1) * 1   (sum = 1)
           convex        : w_t = softmax(z_t / tau)               (sum = 1, w >= 0)

    6. Prediction: y_hat_t = w_t^T * y_hat_base_t
       Loss: mean squared error on y_hat_t vs y_t (real datasets).

The TAFS class inherits TrainableLightningRouter so it plugs directly
into the BaseExperiment orchestration loop. The key difference from a
hard router: predict_proba returns the continuous weights w_t, and
evaluate reports the MSE/MASE of the weighted combination, not
selection accuracy.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.models.base import TrainableLightningRouter


class FeatureTokeniser(nn.Module):
    """Map raw feature vector x in R^p to a token sequence in R^{p x d}.

    Each feature j gets its own learnable scaler a_j (initialised to 1)
    and identity embedding e_j (initialised to a family anchor + noise).

    Args:
        num_features: number of input features p.
        d_model: embedding dimension d.
        feature_families: optional list of length p where each entry is
            a family name ('lag', 'rolling', 'calendar', 'exogenous',
            'regime'). Used to initialise e_j to family-specific anchors.
            If None all embeddings are initialised identically.
    """

    def __init__(
        self,
        num_features: int,
        d_model: int,
        feature_families: list[str] | None = None,
    ) -> None:
        super().__init__()
        # TODO: implement per-feature scaler a_j and identity embedding e_j
        # Hint: a_j can be nn.Parameter of shape (p,); e_j is nn.Embedding(p, d)
        # or nn.Parameter of shape (p, d). For the family-anchor initialisation
        # see Section III-B of the manuscript.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, p) raw feature matrix at one time step.
        Returns:
            tokens: (B, p, d) feature token sequence.
        """
        # TODO
        raise NotImplementedError


class ContextToken(nn.Module):
    """Project the context summary s_t in R^C to a single token in R^d.

    Args:
        context_dim: dimension of s_t (C). Set to 0 to disable the
            context path entirely (ablation baseline).
        d_model: embedding dimension d.
    """

    def __init__(self, context_dim: int, d_model: int) -> None:
        super().__init__()
        # TODO: implement a linear projection W_c, b_c
        # When context_dim == 0 this module should be a no-op (return None).
        raise NotImplementedError

    def forward(self, context: torch.Tensor | None) -> torch.Tensor | None:
        """
        Args:
            context: (B, C) or None.
        Returns:
            token: (B, 1, d) or None.
        """
        # TODO
        raise NotImplementedError


class CombinationHead(nn.Module):
    """Two-layer MLP head + constraint projection.

    Maps CLS embedding u_t in R^d to combination weights w_t in R^K
    under the requested constraint regime.

    Args:
        d_model: input dimension (CLS output).
        num_experts: K base predictors.
        constraint: 'unconstrained' | 'affine' | 'convex'.
        tau: softmax temperature for the convex constraint (default 1.0).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        constraint: str = "affine",
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if constraint not in ("unconstrained", "affine", "convex"):
            raise ValueError(f"Unknown constraint: {constraint!r}")
        self.constraint = constraint
        self.tau = tau
        # TODO: implement W1, b1, W2, b2 (two-layer MLP: d -> d -> K)
        raise NotImplementedError

    def forward(self, cls_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_output: (B, d) CLS token output from the transformer.
        Returns:
            weights: (B, K) combination weights w_t under the constraint.
            logits:  (B, K) pre-constraint logits z_t.
        """
        # TODO:
        # 1. Compute z_t = W2(GELU(W1 u_t + b1)) + b2
        # 2. Project to w_t:
        #    - unconstrained: w_t = z_t
        #    - affine: w_t = z_t - (1/K)(1^T z_t - 1) * 1  (sum-to-one)
        #    - convex: w_t = softmax(z_t / tau)             (simplex)
        raise NotImplementedError


class TAFS(TrainableLightningRouter):
    """Full TAFS model.

    Assembles FeatureTokeniser + ContextToken + CLS token + Transformer
    encoder + CombinationHead into the end-to-end trainable combiner.

    TAFS inherits TrainableLightningRouter which provides:
        - the composite loss (primary squared error + hard/soft CE aux terms)
        - AdamW + linear warmup + cosine annealing schedule
        - test-time caching of selections and errors
        - fit() / evaluate() tied into pl.Trainer

    The only thing TAFS must implement is forward(), which returns the
    (B, K) weight tensor.

    Args:
        num_features: p engineered input features.
        num_experts: K frozen base predictors.
        context_dim: dimension of the context summary s_t. Set to 0 to
            disable context conditioning (ablation).
        d_model: transformer embedding width (default 128).
        num_heads: attention heads (default 4).
        num_layers: transformer encoder layers (default 3).
        d_ffn: feed-forward inner width (default 512).
        dropout: dropout rate (default 0.1).
        constraint: 'unconstrained' | 'affine' | 'convex' (default 'affine').
        feature_families: optional list of length p mapping each feature
            to its family name for anchor initialisation.
        **kwargs: forwarded to TrainableLightningRouter (learning_rate,
            weight_decay, warmup_steps, early_stopping_patience, etc.).
    """

    def __init__(
        self,
        num_features: int,
        num_experts: int,
        context_dim: int = 16,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        d_ffn: int = 512,
        dropout: float = 0.1,
        constraint: str = "affine",
        feature_families: list[str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        # TODO: instantiate FeatureTokeniser, ContextToken, CLS token,
        # nn.TransformerEncoder, and CombinationHead.
        raise NotImplementedError

    def forward(self, batch: dict) -> torch.Tensor:
        """Return (B, K) combination weights.

        Expected batch keys:
            feature_matrix: (B, p)   raw feature values at this time step.
            context:        (B, C)   context summary s_t (optional if context_dim=0).
            base_preds:     (B, K)   frozen base predictor forecasts.

        The return value is used by TrainableLightningRouter to compute
        the loss and to buffer predictions for evaluate().
        """
        # TODO:
        # 1. Tokenise features:  tokens = self.tokeniser(batch["feature_matrix"])   -> (B, p, d)
        # 2. Build context token: ctx = self.context_token(batch.get("context"))    -> (B, 1, d) or None
        # 3. Prepend CLS token and optionally the context token:
        #    seq = [cls, ctx, tokens]   -> (B, p+2, d)  or (B, p+1, d) without context
        # 4. Run the transformer encoder over the sequence.
        # 5. Extract the CLS output: u_t = out[:, 0, :]
        # 6. Compute weights: w_t, _ = self.head(u_t)
        # 7. Return w_t  (shape B, K)
        raise NotImplementedError

    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Alias for forward used by BaseRouter.evaluate().

        For TAFS, predict_proba returns the continuous weights w_t.
        Hard 'selection' is only meaningful in the convex regime.
        """
        self.eval()
        with torch.no_grad():
            return self(batch)
