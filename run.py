"""Single Hydra entry point for TAFS experiments.

Usage:
    python run.py                                    # default = tafs/smoke
    python run.py experiment=tafs/synthetic          # full synthetic experiment
    python run.py experiment=tafs/synthetic trainer.max_epochs=5

Any config key can be overridden on the command line (Hydra syntax).
See configs/ for the full config tree and experiment/*.yaml for examples.

All concrete classes are constructed via ``hydra.utils.instantiate``
inside BaseExperiment.run(); this file only wires logging, float32
precision, and MLflow.
"""

from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from src.core.utils.logging import setup_unified_logging
from src.core.utils.mlflow_setup import setup_mlflow


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    logging_kwargs = {"level": cfg.logging.level}
    if "fmt" in cfg.logging:
        logging_kwargs["fmt"] = cfg.logging.fmt
    setup_unified_logging(**logging_kwargs)

    precision = cfg.get("float32_matmul_precision", "high")
    torch.set_float32_matmul_precision(precision)

    mlflow_cfg = OmegaConf.to_container(cfg.mlflow, resolve=True)
    if not mlflow_cfg.get("experiment_name"):
        mlflow_cfg["experiment_name"] = HydraConfig.get().runtime.choices.experiment
    setup_mlflow(OmegaConf.create(mlflow_cfg))

    experiment = instantiate(cfg.experiment, _recursive_=False)
    experiment.run(cfg.experiment)


if __name__ == "__main__":
    main()
