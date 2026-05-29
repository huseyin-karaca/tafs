"""Combiner evaluation metrics for TAFS.

A combiner emits combination weights over K base predictors. The
"error matrix" of shape ``(N, K)`` gives per-step per-predictor
squared error.

This module computes two families of summary metrics:

    * **Primary** — squared error for TAFS. Always present; produced
      by :meth:`compute_all`.
    * **Secondary** — sMAPE, when the cache includes a per-step
      per-predictor sMAPE matrix. Emitted under ``smape_`` keys.

All keys are returned **without** a split prefix; the caller is
expected to prepend ``"test/"`` / ``"val/"`` / etc. before logging.
"""

from __future__ import annotations

import numpy as np


class SelectionMetrics:
    """Static helpers for router evaluation.

    All methods accept:
        selected_idx:  (N,)   integer expert indices chosen by the router
        error_matrix:  (N, K) per-window per-expert primary error
        expert_names:  list[str] of length K (used for naming the
                                              per-expert error means)
        smape_matrix:  (N, K) optional per-window per-expert sMAPE
    """

    @staticmethod
    def selected_error(selected_idx: np.ndarray, error_matrix: np.ndarray) -> np.ndarray:
        """Per-window error of the chosen expert. Shape ``(N,)``."""
        rows = np.arange(error_matrix.shape[0])
        return error_matrix[rows, selected_idx]

    @staticmethod
    def oracle_error(error_matrix: np.ndarray) -> np.ndarray:
        """Per-window error of the (post-hoc) best expert. Shape ``(N,)``."""
        return error_matrix.min(axis=-1)

    @staticmethod
    def selection_accuracy(selected_idx: np.ndarray, error_matrix: np.ndarray) -> float:
        """Fraction of windows where the router picked argmin_k error."""
        return float((selected_idx == error_matrix.argmin(axis=-1)).mean())

    @staticmethod
    def compute_all(
        selected_idx: np.ndarray,
        error_matrix: np.ndarray,
        expert_names: list[str],
        smape_matrix: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Bundle of scalars suitable for MLflow ``log_metrics``.

        Always-present keys (primary error = squared error for TAFS):
            selected_error_mean    selected_mase_mean   (alias)
            oracle_error_mean      oracle_mase_mean     (alias)
            error_gap_mean         mase_gap_mean        (alias)
            selection_accuracy
            expert_<name>_error_mean   per-expert mean primary error
            oracle_dist_<name>          fraction of windows for which
                                        <name> is the oracle expert
            selected_dist_<name>        fraction of windows the router
                                        chose <name>
            per_expert_recall_<name>    P(router picks <name> | oracle
                                        is <name>); diagonal of the
                                        normalised confusion matrix
            per_expert_precision_<name> P(oracle is <name> | router
                                        picked <name>); how reliable
                                        each chosen expert is
            best_static_expert_mase     min over k of per-expert mean
                                        primary error — the "always
                                        pick the same expert" floor we
                                        must beat to claim routing wins
            n_windows

        When ``smape_matrix`` is supplied, the same five summary
        statistics are also emitted with ``smape_`` infix
        (``selected_smape_mean``, ``oracle_smape_mean``,
        ``smape_gap_mean``, plus per-expert ``expert_<name>_smape_mean``).
        """
        sel_err = SelectionMetrics.selected_error(selected_idx, error_matrix)
        ora_err = SelectionMetrics.oracle_error(error_matrix)
        per_expert_mean = error_matrix.mean(axis=0)
        out: dict[str, float] = {
            "selected_error_mean": float(sel_err.mean()),
            "oracle_error_mean": float(ora_err.mean()),
            "error_gap_mean": float((sel_err - ora_err).mean()),
            "selected_mase_mean": float(sel_err.mean()),
            "oracle_mase_mean": float(ora_err.mean()),
            "mase_gap_mean": float((sel_err - ora_err).mean()),
            "selection_accuracy": SelectionMetrics.selection_accuracy(selected_idx, error_matrix),
            "n_windows": int(error_matrix.shape[0]),
        }
        for name, val in zip(expert_names, per_expert_mean):
            out[f"expert_{name}_error_mean"] = float(val)

        # why: routing diagnostics — oracle/router class balance and the
        # per-expert recall/precision answer "is the router stuck on one
        # expert, or is it just unable to tell them apart?"
        diag = SelectionMetrics._routing_diagnostics(selected_idx, error_matrix, expert_names)
        out.update(diag)
        out["best_static_expert_mase"] = float(per_expert_mean.min())
        # Oracle-gap fraction: fraction of the achievable improvement
        # (best_static -> oracle) that the router actually captured.
        # 1.0 = oracle; 0.0 = no better than picking the best single
        # expert; negative = worse than best static. NaN when there is
        # no oracle gap (single-expert dominance) or when the test set
        # is empty.
        best_static = float(per_expert_mean.min())
        oracle_mean = float(ora_err.mean())
        gap = best_static - oracle_mean
        out["oracle_gap_fraction"] = (
            float((best_static - float(sel_err.mean())) / gap) if gap > 1e-12 else float("nan")
        )

        if smape_matrix is not None and smape_matrix.shape == error_matrix.shape:
            sel_sm = SelectionMetrics.selected_error(selected_idx, smape_matrix)
            ora_sm = SelectionMetrics.oracle_error(smape_matrix)
            per_expert_sm = smape_matrix.mean(axis=0)
            out["selected_smape_mean"] = float(sel_sm.mean())
            out["oracle_smape_mean"] = float(ora_sm.mean())
            out["smape_gap_mean"] = float((sel_sm - ora_sm).mean())
            for name, val in zip(expert_names, per_expert_sm):
                out[f"expert_{name}_smape_mean"] = float(val)
            out["best_static_expert_smape"] = float(per_expert_sm.min())
        return out

    @staticmethod
    def _routing_diagnostics(
        selected_idx: np.ndarray,
        error_matrix: np.ndarray,
        expert_names: list[str],
    ) -> dict[str, float]:
        n = int(error_matrix.shape[0])
        k = int(error_matrix.shape[1])
        if n == 0 or k == 0:
            return {}
        oracle_idx = error_matrix.argmin(axis=-1)
        oracle_counts = np.bincount(oracle_idx, minlength=k).astype(float)
        selected_counts = np.bincount(selected_idx, minlength=k).astype(float)
        out: dict[str, float] = {}
        for i, name in enumerate(expert_names):
            out[f"oracle_dist_{name}"] = float(oracle_counts[i] / n)
            out[f"selected_dist_{name}"] = float(selected_counts[i] / n)
            mask_o = oracle_idx == i
            mask_s = selected_idx == i
            recall = float((selected_idx[mask_o] == i).mean()) if mask_o.any() else float("nan")
            precision = float((oracle_idx[mask_s] == i).mean()) if mask_s.any() else float("nan")
            out[f"per_expert_recall_{name}"] = recall
            out[f"per_expert_precision_{name}"] = precision
        return out
