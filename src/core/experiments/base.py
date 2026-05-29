"""Parent / child / grandchild experiment orchestration.

MLflow run hierarchy (per ``experiment-tracking.md``):

    experiment   -> tafs
    parent run   -> <dataset>_<version>          e.g. synth_v1, tng_v1
    child run    -> <method_name>                e.g. tafs_affine, lgbm
    grandchild   -> seed_<n>_split_<i>           e.g. seed_7_split_0

Per-grandchild logging
----------------------
* All training-time metrics (``train/*``, ``val/*``) go to MLflow via
  ``MLFlowLogger`` attached to the Trainer.
* Test metrics returned by ``model.evaluate()`` are logged with a
  ``test/`` prefix so they don't collide with training keys in the UI.
* For trainable routers, the parameter count is logged as the
  ``n_params`` parameter (Table IV needs this).
* A ``predictions.npz`` artifact (selected_idx, probs, error_matrix,
  primary_regime, test_indices, optional error_matrix_smape) is written
  for every grandchild so cross-method figures (Figure 2) can be
  assembled at parent-run scope.

Per-child aggregation
---------------------
For each metric reported by a grandchild we log mean, SEM, and SD
under ``test/<metric>``, ``test/<metric>__seed_sem``,
``test/<metric>__seed_std``. The manuscript convention is mean (std);
SEM is also retained for the Nadeau-Bengio paired t-test downstream.

Parent-run hook
---------------
Subclasses can override :meth:`on_parent_close` to write artifacts that
require cross-method coordination (Figure 2 for the synthetic
experiment). The hook is invoked while the parent MLflow run is still
active so artifacts go to the correct run.
"""

from __future__ import annotations

from abc import ABC
import json
from pathlib import Path
import tempfile

from hydra.utils import instantiate
from loguru import logger
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from src.core.reporting import Report
from src.core.utils import git
from src.core.utils.seeding import seed_everything


class BaseExperiment(ABC):
    """Generic three-level experiment driver."""

    def __init__(
        self,
        parent_name: str,
        datamodule_cfg: DictConfig,
        methods: DictConfig,
        seeds: list[int],
        trainer_cfg: DictConfig | None = None,
        fixed_data_split: bool = False,
        allow_dirty: bool = False,
        **kwargs,
    ) -> None:
        # YAML anchor scratch fields (keys starting with ``_``) survive
        # the Hydra compose step and end up in our kwargs. Drop them
        # silently; reject any other unrecognised kwarg as a typo.
        leaked = [k for k in kwargs if not k.startswith("_")]
        if leaked:
            raise TypeError(f"unexpected kwargs: {leaked}")
        self.parent_name = parent_name
        self.datamodule_cfg = datamodule_cfg
        self.methods = methods
        self.seeds = list(seeds)
        self.trainer_cfg = trainer_cfg
        self.fixed_data_split = fixed_data_split
        self.allow_dirty = allow_dirty

        # Populated as (method_name, seed) -> dict[str, np.ndarray] for any
        # subclass that wants to do cross-method aggregation in
        # ``on_parent_close`` (e.g. SyntheticExperiment / Figure 2).
        self._predictions: dict[tuple[str, int], dict[str, np.ndarray]] = {}

        if fixed_data_split:
            logger.warning(
                "fixed_data_split=True — Nadeau-Bengio paired t-test "
                "assumptions are violated. Use only for sanity checks."
            )

    # ---- top-level entry point -----------------------------------------

    def run(self, full_cfg: DictConfig) -> None:
        if git.is_dirty() and not self.allow_dirty:
            logger.warning(
                "Working tree is dirty; proceeding (set allow_dirty=False "
                "in production runs). Use `make reproduce` to replay."
            )

        with mlflow.start_run(run_name=self.parent_name) as parent:
            self._log_repro_artifacts(full_cfg, parent.info.run_id)
            for method_name, method_cfg in self.methods.items():
                logger.info(f"--- method: {method_name} ---")
                self._run_method(method_name, method_cfg)
            # Subclass hook for any parent-scope artifacts (Figure 2, etc.).
            self.on_parent_close()

    # ---- subclass hooks ------------------------------------------------

    def produce_reports(self) -> list[Report]:
        """Return manuscript outputs owned by this experiment.

        Override per concrete experiment. The default
        :meth:`on_parent_close` logs each returned ``Report`` as a set
        of MLflow artifacts under ``reports/<name>/`` on the parent run.
        Default: no reports.
        """
        return []

    def on_parent_close(self) -> None:
        """Log every ``Report`` returned by :meth:`produce_reports`.

        Called once, while the parent MLflow run is still active, after
        every method has been run. Subclasses may override to add
        cross-method coordination beyond what ``produce_reports``
        captures, but the preferred extension point is
        ``produce_reports``.
        """
        for report in self.produce_reports():
            report.log_to_mlflow()

    # ---- internals -----------------------------------------------------

    def _log_repro_artifacts(self, full_cfg: DictConfig, run_id: str) -> None:
        """Log resolved config + git state under ``repro/`` on the parent."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "resolved_config.yaml").write_text(
                OmegaConf.to_yaml(full_cfg, resolve=True)
            )
            (tmp_path / "git_state.json").write_text(json.dumps(git.repo_state(), indent=2))
            patch = git.dirty_patch()
            if patch:
                (tmp_path / "git_dirty.patch").write_text(patch)
            mlflow.log_artifacts(str(tmp_path), artifact_path="repro")
        mlflow.set_tag("parent_run_id", run_id)

    def _run_method(self, method_name: str, method_cfg: DictConfig) -> None:
        per_seed_metrics: list[dict[str, float]] = []
        with mlflow.start_run(run_name=method_name, nested=True):
            mlflow.log_params({"method": method_name})
            for seed in self.seeds:
                metrics = self._run_seed(method_name, method_cfg, seed)
                per_seed_metrics.append(metrics)
            self._aggregate(per_seed_metrics)

    def _run_seed(self, method_name: str, method_cfg: DictConfig, seed: int) -> dict[str, float]:
        seed_everything(seed)
        run_name = f"seed_{seed}_split_{seed}"
        with mlflow.start_run(run_name=run_name, nested=True) as grandchild:
            mlflow.log_param("seed", seed)
            mlflow.log_param("method", method_name)

            # Fresh datamodule per seed: cheap if the cache is already on disk.
            datamodule = instantiate(self.datamodule_cfg, _recursive_=False)
            datamodule.setup("fit")
            split_seed = seed if not self.fixed_data_split else self.seeds[0]
            datamodule.reseed_split(split_seed)

            model = instantiate(method_cfg, _recursive_=False)

            # Log parameter count for trainable routers; harmless for
            # training-free baselines (they have no nn.Module parents).
            if isinstance(model, torch.nn.Module):
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                mlflow.log_param("n_params", int(n_params))

            model.fit(
                datamodule=datamodule,
                trainer_cfg=self.trainer_cfg,
                mlflow_run_id=grandchild.info.run_id,
            )
            raw_metrics = model.evaluate(datamodule)

            # Prefix every test metric with ``test/`` so the UI keeps
            # train/val/test partitions visually distinct.
            test_metrics = {f"test/{k}": v for k, v in raw_metrics.items()}
            mlflow.log_metrics(test_metrics)

            # Dump per-window predictions as an MLflow artifact AND stash
            # them for the parent-level hook.
            preds = dict(getattr(model, "last_test_predictions", {}))
            if preds:
                self._predictions[(method_name, seed)] = preds
                self._log_predictions_artifact(preds)

            return test_metrics

    @staticmethod
    def _log_predictions_artifact(preds: dict[str, np.ndarray]) -> None:
        """Save ``predictions.npz`` under the current grandchild run."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.npz"
            np.savez_compressed(path, **preds)
            mlflow.log_artifact(str(path), artifact_path="predictions")

    def _aggregate(self, per_seed: list[dict[str, float]]) -> None:
        """Log ``mean(m)``, ``m__seed_sem``, and ``m__seed_std`` per metric.

        Manuscript tables use mean (std); the Nadeau-Bengio paired t-test
        needs SEM as well, so we keep both.
        """
        if not per_seed:
            return
        keys = sorted(per_seed[0].keys())
        rows = np.array([[d.get(k, np.nan) for k in keys] for d in per_seed], dtype=float)
        means = np.nanmean(rows, axis=0)
        if rows.shape[0] > 1:
            stds = np.nanstd(rows, axis=0, ddof=1)
            sems = stds / np.sqrt(rows.shape[0])
        else:
            stds = np.zeros_like(means)
            sems = np.zeros_like(means)
        agg: dict[str, float] = {}
        for k, mean, sem, std in zip(keys, means, sems, stds):
            if np.isnan(mean):
                continue
            agg[k] = float(mean)
            agg[f"{k}__seed_sem"] = float(sem)
            agg[f"{k}__seed_std"] = float(std)
        mlflow.log_metrics(agg)
