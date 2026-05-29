"""Tests for SelectionMetrics."""

from __future__ import annotations

import numpy as np

from src.core.utils.metrics import SelectionMetrics


def test_selection_metrics_basic() -> None:
    err = np.array(
        [
            [0.1, 0.5, 0.9, 1.2],
            [1.0, 0.2, 0.3, 0.4],
            [0.7, 0.7, 0.7, 0.6],
        ]
    )
    sel = np.array([0, 1, 2])
    out = SelectionMetrics.compute_all(sel, err, ["a", "b", "c", "d"])
    np.testing.assert_allclose(out["selected_error_mean"], (0.1 + 0.2 + 0.7) / 3)
    np.testing.assert_allclose(out["oracle_error_mean"], (0.1 + 0.2 + 0.6) / 3)
    # Picked 0,1,2; argmin per row is 0,1,3 — accuracy = 2/3.
    assert abs(out["selection_accuracy"] - 2 / 3) < 1e-6


def test_oracle_is_lower_bound() -> None:
    rng = np.random.default_rng(0)
    err = rng.uniform(0.1, 2.0, size=(100, 5))
    sel = rng.integers(0, 5, size=100)
    sel_err = SelectionMetrics.selected_error(sel, err)
    ora_err = SelectionMetrics.oracle_error(err)
    assert (sel_err >= ora_err - 1e-12).all()


def test_routing_diagnostics_distributions_and_recall() -> None:
    err = np.array(
        [
            [0.1, 0.5, 0.9],  # oracle = a
            [0.7, 0.2, 0.4],  # oracle = b
            [0.6, 0.5, 0.4],  # oracle = c
            [0.3, 0.4, 0.5],  # oracle = a
        ]
    )
    sel = np.array([0, 0, 2, 0])  # picked a, a, c, a (recall(a)=1, recall(b)=0, recall(c)=1)
    out = SelectionMetrics.compute_all(sel, err, ["a", "b", "c"])
    assert out["oracle_dist_a"] == 0.5
    assert out["oracle_dist_b"] == 0.25
    assert out["oracle_dist_c"] == 0.25
    assert out["selected_dist_a"] == 0.75
    assert out["selected_dist_b"] == 0.0
    assert out["selected_dist_c"] == 0.25
    assert out["per_expert_recall_a"] == 1.0
    assert out["per_expert_recall_b"] == 0.0
    assert out["per_expert_recall_c"] == 1.0
    # Precision: P(oracle=k | selected=k). 3 picks of a, 2 actually oracle=a.
    assert abs(out["per_expert_precision_a"] - 2 / 3) < 1e-9
    assert out["per_expert_precision_c"] == 1.0


def test_best_static_expert_is_min_of_per_expert_mean() -> None:
    err = np.array([[0.4, 1.0, 0.2], [0.6, 1.0, 0.4]])
    sel = np.array([0, 0])
    out = SelectionMetrics.compute_all(sel, err, ["a", "b", "c"])
    # per-expert means = [0.5, 1.0, 0.3]; min = 0.3 (c)
    assert abs(out["best_static_expert_mase"] - 0.3) < 1e-9
