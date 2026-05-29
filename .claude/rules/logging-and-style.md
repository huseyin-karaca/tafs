# Logging, Style & CLI

**Logging.** `loguru` only, configured once at startup via `src.core.utils.logging.setup_unified_logging(level=..., fmt=...)`. Do not use stdlib `logging` directly in new code.

**CLI surface.** All experiments go through Hydra:
```bash
python run.py experiment=tafs/synthetic                    # full experiment
python run.py experiment=tafs/synthetic trainer.max_epochs=2   # override anything
make run CONFIG=tafs/synthetic OVERRIDES="key=val"        # Makefile wrapper
make prepare_data DATASET=synthetic                        # build a data cache
make smoke                                                 # quick end-to-end check
make lint / make format                                    # ruff (line length 100)
```

**Self-documenting Makefile.** `## comment` lines above each target are printed by `make help`.

**Smoke test.** `tests/test_smoke.py` runs a tiny end-to-end pipeline. Every PR that touches `src/` must pass `make smoke`.

**Precision flag.** `cfg.float32_matmul_precision` (`"high"` default) is set at the top of `run.py`; do not call `torch.set_float32_matmul_precision` elsewhere.

**No prints.** Replace stray `print` with `logger.info` / `logger.debug`.
