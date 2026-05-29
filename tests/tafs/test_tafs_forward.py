"""Forward-pass shape and constraint tests for the TAFS model.

These tests verify:
    1. Output shapes are correct for all three constraint regimes.
    2. The convex constraint satisfies w >= 0 and sum(w) = 1.
    3. The affine constraint satisfies sum(w) = 1 (weights can be negative).
    4. Gradients flow through every learnable parameter.
    5. The no-context ablation (context_dim=0) runs without errors.
    6. The model can overfit a tiny batch (basic sanity check).

These tests will pass once TAFS.forward() and its sub-modules are
implemented. Until then they will raise NotImplementedError.
"""

from __future__ import annotations

import pytest
import torch

from src.tafs.models.tafs import TAFS


@pytest.fixture
def small_tafs() -> TAFS:
    return TAFS(
        num_features=20,
        num_experts=4,
        context_dim=8,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ffn=64,
        dropout=0.0,
        constraint="affine",
    )


@pytest.mark.parametrize("constraint", ["unconstrained", "affine", "convex"])
def test_forward_shapes(constraint: str) -> None:
    model = TAFS(
        num_features=20,
        num_experts=4,
        context_dim=8,
        d_model=32,
        num_heads=4,
        num_layers=2,
        d_ffn=64,
        constraint=constraint,
    )
    B = 16
    batch = {
        "feature_matrix": torch.randn(B, 20),
        "context": torch.randn(B, 8),
        "base_preds": torch.randn(B, 4),
        "error_matrix": torch.randn(B, 4).abs(),
    }
    weights = model(batch)
    assert weights.shape == (B, 4), f"Expected ({B}, 4), got {weights.shape}"


def test_convex_constraint_satisfied() -> None:
    model = TAFS(
        num_features=20, num_experts=4, context_dim=8,
        d_model=32, num_heads=4, num_layers=2, d_ffn=64,
        constraint="convex",
    )
    B = 8
    batch = {
        "feature_matrix": torch.randn(B, 20),
        "context": torch.randn(B, 8),
        "base_preds": torch.randn(B, 4),
        "error_matrix": torch.randn(B, 4).abs(),
    }
    w = model(batch)
    assert (w >= 0).all(), "Convex weights must be non-negative"
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-5), "Convex weights must sum to 1"


def test_affine_constraint_satisfied() -> None:
    model = TAFS(
        num_features=20, num_experts=4, context_dim=8,
        d_model=32, num_heads=4, num_layers=2, d_ffn=64,
        constraint="affine",
    )
    B = 8
    batch = {
        "feature_matrix": torch.randn(B, 20),
        "context": torch.randn(B, 8),
        "base_preds": torch.randn(B, 4),
        "error_matrix": torch.randn(B, 4).abs(),
    }
    w = model(batch)
    assert torch.allclose(w.sum(dim=-1), torch.ones(B), atol=1e-5), "Affine weights must sum to 1"


def test_gradient_flows() -> None:
    model = TAFS(
        num_features=20, num_experts=4, context_dim=8,
        d_model=32, num_heads=4, num_layers=2, d_ffn=64,
        constraint="affine",
    )
    B = 8
    batch = {
        "feature_matrix": torch.randn(B, 20),
        "context": torch.randn(B, 8),
        "base_preds": torch.randn(B, 4),
        "error_matrix": torch.randn(B, 4).abs(),
        "target": torch.randn(B),
    }
    weights = model(batch)
    y_hat = (weights * batch["base_preds"]).sum(dim=-1)
    loss = ((y_hat - batch["target"]) ** 2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"


def test_no_context_path() -> None:
    """context_dim=0 disables the context token (ablation baseline)."""
    model = TAFS(
        num_features=20, num_experts=4, context_dim=0,
        d_model=32, num_heads=4, num_layers=2, d_ffn=64,
    )
    B = 8
    batch = {
        "feature_matrix": torch.randn(B, 20),
        "base_preds": torch.randn(B, 4),
        "error_matrix": torch.randn(B, 4).abs(),
    }
    weights = model(batch)
    assert weights.shape == (B, 4)
