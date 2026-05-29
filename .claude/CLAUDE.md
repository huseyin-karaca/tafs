# TAFS — Transformer-Attentive Feature Selection

This repository implements TAFS: a feature-axis transformer that learns
context-conditioned combination weights over a pool of frozen base forecasters.

- Full specification: `.claude/specs/tafs.md`
- Paper draft (source of truth for content): `reports/tafs-draft.pdf`
- Code-pattern rules: `.claude/rules/` — read the relevant ones before generating code.

## Layout

```
src/core/          paper-agnostic infrastructure (ABCs, data, utils, stats)
src/tafs/          TAFS-specific code (models, data, experiments)
tests/core/        tests for shared infrastructure
tests/tafs/        tests for TAFS-specific modules
configs/experiment/tafs/   one YAML per reported experiment
configs/data/tafs/         one YAML per dataset
```

**Import rule:** `tafs → core` is allowed. `core → tafs` is not.

## to save tokens, do not read .venv folder content.
