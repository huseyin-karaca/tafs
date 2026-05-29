PROJECT_NAME = tafs
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python


## Install Python dependencies (uses uv — do not use pip directly)
.PHONY: requirements
requirements:
	uv sync


## Lint using ruff
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Delete compiled Python files and caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache outputs

## Run pytest
.PHONY: test
test:
	python -m pytest tests/

## Set up a new virtual environment with uv
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Activate with: source .venv/bin/activate"


###############################################################################
# Experiment commands
###############################################################################

CONFIG ?=
OVERRIDES ?=

## Run an experiment: make run CONFIG=tafs/synthetic OVERRIDES="trainer.max_epochs=5"
.PHONY: run
run:
	$(PYTHON_INTERPRETER) run.py experiment=$(CONFIG) $(OVERRIDES)

## Smoke test: 1 epoch, baseline-only, verifies the full pipeline
.PHONY: smoke
smoke:
	$(PYTHON_INTERPRETER) run.py experiment=tafs/smoke

## Prepare a dataset cache: make prepare_data DATASET=synthetic
## This materialises data/processed/tafs/<DATASET>/ from raw series.
## Run this once before training on real data.
.PHONY: prepare_data
prepare_data:
	$(PYTHON_INTERPRETER) -m src.tafs.data.prepare --dataset $(DATASET)

## Open the MLflow UI to inspect results
.PHONY: mlflow_ui
mlflow_ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db


###############################################################################
# Self-documenting help
###############################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
