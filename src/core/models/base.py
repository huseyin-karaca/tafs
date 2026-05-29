"""Combiner / router ABCs for TAFS.

Hierarchy
---------
:class:`BaseRouter` (ABC)
    +-- :class:`TrainableLightningRouter` (``pl.LightningModule``)
    |       Implements the composite loss + AdamW + warmup-cosine LR
    |       schedule + test-time caching used by all learned combiners.
    +-- :class:`TrainingFreeBaseline`
            ``fit()`` is a no-op (or a prior-fit on the training error
            matrix); ``predict_proba`` / ``select`` are implemented by
            each concrete subclass.

Subclass contract:
    * ``predict_proba(batch) -> (B, K)`` softmax distribution.
    * ``select(batch) -> (B,)`` defaults to argmax.
    * ``evaluate(datamodule) -> dict`` reuses buffered test-time
      selections / errors so the test loader is traversed only once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import math
import tempfile
from typing import Any

import hydra
import mlflow
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class BaseRouter(ABC):
    """Common interface for trainable and training-free routers.

    After :meth:`evaluate` runs, every subclass exposes the per-window
    test-set predictions on :attr:`last_test_predictions` so the
    experiment driver can dump them as MLflow artifacts and assemble
    cross-method figures (e.g. Figure 2 in the manuscript).
    """

    #: Populated by :meth:`evaluate` with arrays of shape:
    #:    selected_idx:   (N,)        chosen expert per window
    #:    probs:          (N, K)      router softmax (one-hot for hard routers)
    #:    error_matrix:   (N, K)      per-window MASE per expert
    #:    error_matrix_smape: (N, K)  per-window sMAPE per expert (if cached)
    #:    primary_regime: (N,)        regime label (synthetic only; zeros elsewhere)
    #:    test_indices:   (N,)        index into the full datamodule cache
    last_test_predictions: dict[str, np.ndarray] = {}

    def fit(
        self,
        datamodule: pl.LightningDataModule,
        trainer_cfg: DictConfig | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        """Default: no-op. Override in trainable / prior-learning subclasses."""

    @abstractmethod
    def predict_proba(self, batch: dict) -> torch.Tensor:
        """Return ``(B, K)`` probability tensor."""

    def select(self, batch: dict) -> np.ndarray:
        """Default = argmax of :meth:`predict_proba`. Override for stochastic baselines."""
        with torch.no_grad():
            probs = self.predict_proba(batch)
        return probs.argmax(dim=-1).cpu().numpy()

    def evaluate(self, datamodule: pl.LightningDataModule) -> dict:
        """Streaming eval over the test loader.

        Subclasses with a training loop already buffer probabilities and
        the error matrix in ``on_test_start`` / ``test_step`` and override
        this method to read those buffers. For training-free baselines
        we recompute by calling :meth:`predict_proba` on each batch.
        """
        from src.core.utils.metrics import SelectionMetrics

        datamodule.setup("test")
        all_idx: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []
        all_err: list[np.ndarray] = []
        all_smape: list[np.ndarray] = []
        all_regime: list[np.ndarray] = []
        for batch in datamodule.test_dataloader():
            with torch.no_grad():
                probs = self.predict_proba(batch)
            all_probs.append(probs.detach().cpu().numpy())
            all_idx.append(self.select(batch))
            all_err.append(batch["error_matrix"].detach().cpu().numpy())
            if "error_matrix_smape" in batch:
                all_smape.append(batch["error_matrix_smape"].detach().cpu().numpy())
            if "primary_regime" in batch:
                all_regime.append(batch["primary_regime"].detach().cpu().numpy())

        sel = np.concatenate(all_idx, axis=0)
        probs = np.concatenate(all_probs, axis=0)
        err = np.concatenate(all_err, axis=0)
        smape = np.concatenate(all_smape, axis=0) if all_smape else None
        regime = (
            np.concatenate(all_regime, axis=0)
            if all_regime
            else np.zeros(sel.shape[0], dtype=np.int8)
        )

        # Test indices into the underlying dataset (so multiple methods
        # at the same seed can be aligned window-for-window in Figure 2).
        test_indices = np.asarray(getattr(datamodule, "_test").indices, dtype=np.int64)

        self.last_test_predictions = {
            "selected_idx": sel,
            "probs": probs,
            "error_matrix": err,
            "primary_regime": regime,
            "test_indices": test_indices,
        }
        if smape is not None:
            self.last_test_predictions["error_matrix_smape"] = smape
        return SelectionMetrics.compute_all(sel, err, datamodule.expert_names, smape)


# ---------------------------------------------------------------------------
# Trainable router scaffold
# ---------------------------------------------------------------------------


class TrainableLightningRouter(BaseRouter, pl.LightningModule):
    """Abstract trainable combiner (TAFS and its ablation baselines).

    Subclasses implement :meth:`forward` returning a ``(B, K)`` softmax
    distribution. This base class implements:

    * the composite loss ``λ_mase L_mase + λ_hard L_hard + λ_soft L_soft``
      (the speech repo's primary / hard / soft template, renamed for
      forecasting),
    * AdamW + linear-warmup + cosine-annealing schedule per
      ``training-recipe.md``,
    * test-time caching of ``(selected_idx, error_matrix)`` so
      :meth:`evaluate` does not re-traverse the loader.
    """

    def __init__(
        self,
        primary_weight: float = 1.0,
        hard_ce_weight: float = 0.3,
        soft_ce_weight: float = 0.5,
        soft_ce_temperature: float = 1.5,
        label_smoothing: float = 0.1,
        class_balanced_loss: bool = True,
        diversity_weight: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 500,
        early_stopping_patience: int | None = 20,
        *_args: Any,
        **_kwargs: Any,
    ) -> None:
        pl.LightningModule.__init__(self)
        self.primary_weight = primary_weight
        self.hard_ce_weight = hard_ce_weight
        self.soft_ce_weight = soft_ce_weight
        self.soft_ce_temperature = soft_ce_temperature
        self.label_smoothing = label_smoothing
        self.class_balanced_loss = class_balanced_loss
        # why: load-balancing term against MoE-style mode collapse. Adds
        # ``-diversity_weight * H(batch-mean routing prob)`` to the loss,
        # which is equivalent to KL(mean_p || uniform) up to a constant.
        # Default 0.0 keeps backward compatibility for older configs.
        self.diversity_weight = diversity_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.early_stopping_patience = early_stopping_patience
        # Class weights are populated by ``_init_class_weights`` after the
        # datamodule announces its class_priors. Registering as a buffer
        # ensures correct device placement and (de)serialisation.
        self.register_buffer("class_weights", torch.ones(1), persistent=False)
        self._class_weights_ready = False

    # ---- subclass hooks -------------------------------------------------

    @abstractmethod
    def forward(self, batch: dict) -> torch.Tensor:
        """Return post-softmax routing distribution of shape ``(B, K)``."""

    # ---- loss -----------------------------------------------------------

    def _init_class_weights(self, class_priors: list[float]) -> None:
        """Inverse-frequency class weights, normalised to sum to K."""
        k = len(class_priors)
        weights = torch.tensor(
            [1.0 / p if p > 0 else 0.0 for p in class_priors],
            dtype=torch.float32,
        )
        if weights.sum() > 0:
            weights = weights / weights.sum() * k
        self.class_weights = weights.to(self.device)
        self._class_weights_ready = True

    def _compute_loss(
        self,
        probs: torch.Tensor,
        error_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        log_probs = torch.log(probs.clamp_min(1e-8))

        # 1. Weighted error (primary).
        weighted_err = (probs * error_matrix).sum(dim=-1)
        primary_loss = weighted_err.mean()

        # 2. Hard CE with oracle label + (optional) class weights.
        best_idx = error_matrix.argmin(dim=-1)
        weight = (
            self.class_weights
            if (self.class_balanced_loss and self._class_weights_ready)
            else None
        )
        hard_ce = F.cross_entropy(
            log_probs,
            best_idx,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )

        # 3. Soft CE against softmax(-error / τ).
        soft_target = F.softmax(-error_matrix / self.soft_ce_temperature, dim=-1)
        soft_ce = -(soft_target * log_probs).sum(dim=-1).mean()

        # 4. Diversity / load-balancing regulariser.
        mean_p = probs.mean(dim=0).clamp_min(1e-8)
        batch_entropy = -(mean_p * mean_p.log()).sum()

        total = (
            self.primary_weight * primary_loss
            + self.hard_ce_weight * hard_ce
            + self.soft_ce_weight * soft_ce
            - self.diversity_weight * batch_entropy
        )

        # Diagnostic metrics (no gradient impact).
        with torch.no_grad():
            selected = probs.argmax(dim=-1)
            oracle = error_matrix.min(dim=-1).values
            chosen = error_matrix.gather(1, selected.unsqueeze(1)).squeeze(1)
            sel_acc = (selected == best_idx).float().mean()
            metrics = {
                "primary_loss": primary_loss.detach(),
                "hard_ce": hard_ce.detach(),
                "soft_ce": soft_ce.detach(),
                "batch_entropy": batch_entropy.detach(),
                "total_loss": total.detach(),
                "selected_error": chosen.mean(),
                "oracle_error": oracle.mean(),
                "error_gap": (chosen - oracle).mean(),
                "selection_accuracy": sel_acc,
            }
        return total, metrics

    # ---- Lightning hooks ------------------------------------------------

    def _log_metrics(self, split: str, metrics: dict[str, torch.Tensor], batch_size: int) -> None:
        for k, v in metrics.items():
            self.log(
                f"{split}/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

    def training_step(self, batch: dict, batch_idx: int = 0) -> torch.Tensor:  # noqa: ARG002
        probs = self(batch)
        loss, metrics = self._compute_loss(probs, batch["error_matrix"])
        self._log_metrics("train", metrics, batch["error_matrix"].size(0))
        return loss

    def validation_step(self, batch: dict, batch_idx: int = 0) -> torch.Tensor:  # noqa: ARG002
        probs = self(batch)
        loss, metrics = self._compute_loss(probs, batch["error_matrix"])
        self.log(
            "val/L_mase",
            metrics["primary_loss"],
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["error_matrix"].size(0),
        )
        self._log_metrics("val", metrics, batch["error_matrix"].size(0))
        return loss

    def on_test_start(self) -> None:
        # Buffers for streaming evaluation; populated by ``test_step``.
        # We retain the full ``(B, K)`` probability tensors so the
        # experiment driver can dump per-window predictions as artifacts
        # (Figure 2 in the manuscript) without re-running inference.
        self._test_selections: list[np.ndarray] = []
        self._test_probs: list[np.ndarray] = []
        self._test_errors: list[np.ndarray] = []
        self._test_smape: list[np.ndarray] = []
        self._test_regimes: list[np.ndarray] = []

    def test_step(self, batch: dict, batch_idx: int = 0) -> None:  # noqa: ARG002
        probs = self(batch)
        self._test_selections.append(probs.argmax(dim=-1).detach().cpu().numpy())
        self._test_probs.append(probs.detach().cpu().numpy())
        self._test_errors.append(batch["error_matrix"].detach().cpu().numpy())
        if "error_matrix_smape" in batch:
            self._test_smape.append(batch["error_matrix_smape"].detach().cpu().numpy())
        if "primary_regime" in batch:
            self._test_regimes.append(batch["primary_regime"].detach().cpu().numpy())

    def configure_optimizers(self):  # type: ignore[override]
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Linear warmup then cosine decay to 0 over the total step budget.
        total_steps = max(int(self.trainer.estimated_stepping_batches), self.warmup_steps + 1)
        warmup = self.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }

    # ---- routing API ---------------------------------------------------

    def predict_proba(self, batch: dict) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            return self(batch)

    # ---- fit + evaluate ------------------------------------------------

    def fit(
        self,
        datamodule: pl.LightningDataModule,
        trainer_cfg: DictConfig | None = None,
        mlflow_run_id: str | None = None,
    ) -> None:
        datamodule.setup("fit")
        if self.class_balanced_loss:
            self._init_class_weights(datamodule.class_priors)

        from src.core.utils.progress import EpochSummaryLogger

        callbacks: list[pl.Callback] = [EpochSummaryLogger()]
        # Use a temp dir for ModelCheckpoint so we don't pollute the repo.
        self._ckpt_tmpdir = tempfile.TemporaryDirectory()
        ckpt = ModelCheckpoint(
            dirpath=self._ckpt_tmpdir.name,
            monitor="val/L_mase",
            mode="min",
            save_top_k=1,
        )
        callbacks.append(ckpt)
        if self.early_stopping_patience is not None:
            callbacks.append(
                EarlyStopping(
                    monitor="val/L_mase",
                    mode="min",
                    patience=self.early_stopping_patience,
                )
            )

        loggers = []
        if mlflow_run_id is not None:
            loggers.append(
                MLFlowLogger(
                    run_id=mlflow_run_id,
                    tracking_uri=mlflow.get_tracking_uri(),
                    log_model=False,
                )
            )

        trainer_cfg = trainer_cfg or DictConfig({})
        trainer: pl.Trainer = hydra.utils.instantiate(
            trainer_cfg,
            callbacks=callbacks,
            logger=loggers or False,
            _convert_="partial",
        )
        trainer.fit(self, datamodule=datamodule)
        # Restore the best-val checkpoint for downstream evaluation.
        if ckpt.best_model_path:
            self._trainer = trainer
            ckpt_state = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)
            self.load_state_dict(ckpt_state["state_dict"])

    def evaluate(self, datamodule: pl.LightningDataModule) -> dict:
        """Run ``trainer.test`` once and read the cached buffers."""
        from src.core.utils.metrics import SelectionMetrics

        datamodule.setup("test")
        trainer = pl.Trainer(
            logger=False,
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            accelerator="auto",
            devices=1,
        )
        trainer.test(self, datamodule=datamodule, verbose=False)
        sel = np.concatenate(self._test_selections, axis=0)
        probs = np.concatenate(self._test_probs, axis=0)
        err = np.concatenate(self._test_errors, axis=0)
        smape = np.concatenate(self._test_smape, axis=0) if self._test_smape else None
        regime = (
            np.concatenate(self._test_regimes, axis=0)
            if self._test_regimes
            else np.zeros(sel.shape[0], dtype=np.int8)
        )
        test_indices = np.asarray(getattr(datamodule, "_test").indices, dtype=np.int64)
        self.last_test_predictions = {
            "selected_idx": sel,
            "probs": probs,
            "error_matrix": err,
            "primary_regime": regime,
            "test_indices": test_indices,
        }
        if smape is not None:
            self.last_test_predictions["error_matrix_smape"] = smape
        return SelectionMetrics.compute_all(sel, err, datamodule.expert_names, smape)


# ---------------------------------------------------------------------------
# Training-free baselines
# ---------------------------------------------------------------------------


class TrainingFreeBaseline(BaseRouter):
    """Baselines that do not require gradient descent.

    Concrete examples (in ``src/tafs/models/baselines.py``):
        * RandomRouter — uniform per-window choice.
        * WeightedRandomRouter — sample from the inverse-MASE prior fit on train.
        * OracleRouter — argmin over the per-window error matrix (upper bound).
        * MeanEnsemble / WeightedEnsemble — return fixed weights (no selection).
    """

    def fit(
        self,
        datamodule: pl.LightningDataModule,
        trainer_cfg: DictConfig | None = None,  # noqa: ARG002
        mlflow_run_id: str | None = None,  # noqa: ARG002
    ) -> None:
        """Default no-op; prior-fitting baselines override."""

    @abstractmethod
    def predict_proba(self, batch: dict) -> torch.Tensor: ...
