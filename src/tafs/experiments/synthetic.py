"""Synthetic regime-switch experiment for TAFS (Section IV-D-1).

This experiment runs all methods on the synthetic dataset, then
produces two manuscript outputs:

    table_synthetic  — Table II: MSE per method × seed (mean ± std)
    fig_attention    — Figure 2: attention mass per feature family over time

It inherits BaseExperiment which handles the three-level MLflow
hierarchy (parent / child / grandchild), per-seed seeding, and
metric aggregation. See src/core/experiments/base.py.

Usage (via Hydra):
    python run.py experiment=tafs/synthetic
"""

from __future__ import annotations

from src.core.experiments.base import BaseExperiment
from src.core.reporting import Report


class TAFSSyntheticExperiment(BaseExperiment):
    """Synthetic experiment = BaseExperiment + Table II + Figure 2.

    The extra produce_reports() hook is called by on_parent_close()
    while the parent MLflow run is still active, so all report artifacts
    land on the correct run. See BaseExperiment.on_parent_close.
    """

    def produce_reports(self) -> list[Report]:
        """Return the manuscript outputs for this experiment.

        TODO: implement _table_synthetic() and _fig_attention() and
        return them as Report objects.

        A Report wraps a dict of {filename: content} and a name, and
        knows how to log itself to MLflow. See src/core/reporting/report.py.
        """
        # TODO
        return []

    def _table_synthetic(self) -> Report:
        """Build Table II: per-method MSE summary from MLflow grandchild runs.

        Hint: self._predictions is a dict keyed by (method_name, seed)
        mapping to the arrays buffered by BaseRouter.evaluate():
            selected_idx:  (N,)
            probs:         (N, K)
            error_matrix:  (N, K)
            primary_regime:(N,)

        Use these to compute per-method, per-regime MSE and format a
        LaTeX table.
        """
        # TODO
        raise NotImplementedError

    def _fig_attention(self) -> Report:
        """Build Figure 2: attention mass per feature family over time.

        This requires access to per-step attention weights alpha_{t,j}
        from the TAFS model. The model must expose these via its
        last_test_predictions dict (add an 'attention_weights' key in
        TAFS.forward when exporting for evaluation).
        """
        # TODO
        raise NotImplementedError
