"""MLflow tracking-URI / experiment bootstrap.

Wires the tracking URI and creates/selects the MLflow experiment.
Called once at startup from run.py before any experiment code runs.
"""

from __future__ import annotations

import os

from loguru import logger
import mlflow
from omegaconf import DictConfig


def _resolve_tracking_uri(cfg_uri: str | None) -> str:
    """Pick the tracking URI in priority: config > env > local ``mlruns/``."""
    if cfg_uri:
        return cfg_uri
    return os.environ.get("MLFLOW_TRACKING_URI", "mlruns")


def setup_mlflow(cfg: DictConfig) -> str:
    """Configure mlflow and return the resolved tracking URI.

    ``cfg`` must expose ``tracking_uri`` (optional) and ``experiment_name``
    (required — typically resolved from the Hydra defaults choice).
    """
    uri = _resolve_tracking_uri(cfg.get("tracking_uri"))
    mlflow.set_tracking_uri(uri)
    exp_name = cfg.experiment_name
    mlflow.set_experiment(exp_name)
    logger.info(f"MLflow tracking_uri={uri} experiment={exp_name}")
    return uri
