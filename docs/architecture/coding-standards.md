# Coding Standards

## Language and Style

- Use Python 3.11 syntax.
- Use Ruff for formatting and linting.
- Use mypy-compatible type hints in public functions and classes.
- Prefer small, composable modules over large notebooks or scripts.
- Keep docstrings concise and present on public classes and functions.

## Architectural Rules

- Raw data is immutable.
- Domain rules belong in `domain/`, not inside ad hoc scripts.
- Every pipeline command must emit structured logs and a run manifest.
- Ranking output must validate that published top-20 IDs are from `1..39` only.
- Optional normalized-track experiments must be explicit and separately configured.

## Testing Rules

- Add or update unit tests for every non-trivial function.
- Integration tests must cover the CLI path for each core pipeline command.
- Regression tests must protect:
  - validated 25- and 35-total days
  - no-zero forecast enforcement
  - deterministic ranking output
  - champion promotion gates

## Documentation Rules

- Keep `docs/prd.md` and `docs/architecture.md` authoritative.
- When architecture changes invalidate a story, update the docs before coding.
- Generated artifacts must reference `run_id`, `dataset_id`, and `feature_set_id`.
