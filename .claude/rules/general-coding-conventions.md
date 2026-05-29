## Code Conventions

- Modular, Hydra-managed, object-oriented, reproducible codebase.
- Every `src/core/` subpackage exposes ABCs in `base.py`. `src/tafs/` subclasses them — never reinvent the hierarchy.
- Hydra DictConfig flows through everything. Construct via `hydra.utils.instantiate(_recursive_=False)`.
- All randomness via `src.core.utils.seeding.seed_everything(seed)`. No bare `torch.manual_seed`, no bare `np.random.seed`.
- Tests mirror the source layout under `tests/{core,tafs}/`. One test file per non-trivial module.
- Type hints required on public APIs. Internal helpers can be loose.
- Format: ruff. Line length 100.
- Package management via `uv`. Do not use `pip` directly.

## What NOT to do

- Do NOT install packages without using `uv add` to update `pyproject.toml`.
- Do NOT create files outside the layout in `architecture-layout.md`.
- Do NOT import `tafs → core` in reverse (`core → tafs`).
- Do NOT skip writing tests. One module = at least one test file.
