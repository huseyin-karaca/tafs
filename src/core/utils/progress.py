"""Lightweight progress / per-epoch logging utility.

The Lightning default progress bar (`tqdm`) reflows the terminal on each
step which makes the loguru-formatted output unreadable. For our use
case (many small fits in sequence under one Hydra job) we'd rather have
*one log line per epoch* showing the training/validation metrics, so the
captured loguru output stays linear and grep-able.

:class:`EpochSummaryLogger` is a tiny Lightning callback that emits a
single loguru-formatted line at the end of every validation epoch,
printing the keys named in :attr:`metric_keys` (defaults cover the
TAFS training loop).
"""

from __future__ import annotations

from loguru import logger
import pytorch_lightning as pl
import torch


class EpochSummaryLogger(pl.Callback):
    """Emit one loguru line per epoch with selected metrics."""

    def __init__(
        self,
        metric_keys: tuple[str, ...] = (
            "train/total_loss",
            "val/L_mase",
            "val/selection_accuracy",
            "val/error_gap",
        ),
    ) -> None:
        super().__init__()
        self.metric_keys = metric_keys

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:  # noqa: ARG002
        metrics = trainer.callback_metrics
        if not metrics:
            return
        parts: list[str] = []
        for key in self.metric_keys:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, torch.Tensor):
                    val = val.item()
                parts.append(f"{key}={val:.4f}")
        if not parts:
            return
        logger.info(f"epoch {trainer.current_epoch:>3d} | " + " | ".join(parts))
