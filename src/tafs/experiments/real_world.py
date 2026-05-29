"""Real-world experiment for TAFS (Section IV-D-2).

Runs all methods on one real dataset and produces:

    table_real_world  — Table III: MASE per method × dataset

One instance of this experiment is run per dataset in Table I
(TNG, ISE, M5, M4-Quarterly, Weather, Exchange Rate, ETTm2).

Usage (via Hydra):
    python run.py experiment=tafs/tng     # Turkish Natural Gas
    python run.py experiment=tafs/ise     # Istanbul Stock Exchange
"""

from __future__ import annotations

from src.core.experiments.base import BaseExperiment
from src.core.reporting import Report


class TAFSRealWorldExperiment(BaseExperiment):
    """Real-world experiment = BaseExperiment + Table III row.

    Concrete dataset experiments (e.g. configs/experiment/tafs/tng.yaml)
    point at this class via _target_. The datamodule_cfg in those YAML
    files points to the corresponding data/tafs/<dataset>.yaml config.
    """

    def produce_reports(self) -> list[Report]:
        """Return Table III row for this dataset.

        TODO: implement _table_real_world() and return it.
        """
        # TODO
        return []

    def _table_real_world(self) -> Report:
        """Build a per-dataset summary row for Table III.

        Keys to report: MASE of each method, oracle MASE, error gap.
        """
        # TODO
        raise NotImplementedError
