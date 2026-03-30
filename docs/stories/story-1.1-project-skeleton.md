# Story 1.1 — Create the Executable Project Skeleton

**Epic:** 1 — Foundation, Dataset Contract, and Canary Forecast Flow
**Status:** Ready for implementation
**Priority:** P0 — all other stories depend on this
**Estimated complexity:** Medium

## User Story

As an implementation agent,
I want a runnable project skeleton with documented commands and directories,
so that all future work lands in a consistent structure.

## Context

The repository currently contains only BMAD scaffolding, one raw dataset, reference
PDFs, and a senior representative playbook. There is no Python project, no tests, no
build config, and no CLI. This story creates the foundational scaffold that every
subsequent story builds on.

## PRD Traceability

- NFR1: Python 3.11
- NFR2: No UI, batch-oriented
- NFR3: Runnable from documented commands
- NFR4: Clarity, modularity, inspectability
- NFR14: Incremental BMAD story implementation

## Architecture Traceability

- Source tree layout (architecture.md lines 572-626)
- Tech stack (tech-stack.md)
- Coding standards (coding-standards.md)
- Package name: `c5_forecasting`
- CLI framework: Typer
- Config framework: pydantic
- Logging: structlog

## Dependencies

- None (this is the first story)

## Blocked Pending PO Decision

- **CORRECTION-1:** The existing `doc/` directory must be renamed to `docs/` to match
  all governing documents. This story should execute the rename as its first task. If
  the PO prefers to keep `doc/`, all docs must be updated instead. **Decision needed
  before implementation begins.**

## Acceptance Criteria

### AC-1: Python project exists with pyproject.toml

- [ ] `pyproject.toml` exists at repo root using Poetry as build backend
- [ ] Python version constraint is `>=3.11,<3.12`
- [ ] Package name is `c5-forecasting` with import name `c5_forecasting`
- [ ] `src/` layout is used: source lives under `src/c5_forecasting/`
- [ ] Dev dependencies include: pytest, pytest-cov, ruff, mypy, structlog
- [ ] Runtime dependencies include at minimum: pandas, numpy, typer, pydantic, pyarrow, structlog

### AC-2: Directory structure matches architecture

- [ ] All canonical directories exist per architecture source tree:
  - `docs/` (with existing PRD/architecture docs moved from `doc/`)
  - `configs/datasets/`, `configs/features/`, `configs/models/`, `configs/runs/`
  - `data/raw/`, `data/interim/`, `data/processed/`, `data/features/`, `data/forecasts/`
  - `artifacts/manifests/`, `artifacts/runs/`, `artifacts/models/`, `artifacts/metrics/`, `artifacts/plots/`, `artifacts/champion/`
  - `src/c5_forecasting/` with subdirs: `cli/`, `config/`, `domain/`, `data/`, `features/`, `models/`, `ranking/`, `evaluation/`, `registry/`, `pipelines/`
  - `tests/unit/`, `tests/integration/`, `tests/regression/`
  - `docker/`
- [ ] Each Python package directory has an `__init__.py`
- [ ] `data/raw/c5_aggregated_matrix.csv` remains in place and unchanged

### AC-3: .gitignore is comprehensive

- [ ] `.gitignore` excludes: `__pycache__`, `.mypy_cache`, `.ruff_cache`, `*.egg-info`, `dist/`, `.venv/`, `*.pyc`, `.env`
- [ ] `.gitignore` excludes generated artifacts: `artifacts/`, `data/interim/`, `data/processed/`, `data/features/`, `data/forecasts/`
- [ ] `.gitignore` does NOT exclude `data/raw/` (raw data is tracked)
- [ ] `.gitignore` does NOT exclude `configs/` (config files are tracked)

### AC-4: Domain constants module exists

- [ ] `src/c5_forecasting/domain/constants.py` defines:
  - `VALID_PART_IDS: frozenset[int]` = `frozenset(range(1, 40))`
  - `MIN_PART_ID: int` = `1`
  - `MAX_PART_ID: int` = `39`
  - `TOP_K: int` = `20`
  - `PART_COLUMNS: list[str]` = `["P_1", "P_2", ..., "P_39"]`
  - `ZERO_IS_ABSENCE: str` = docstring/comment explaining 0 is not a valid part ID
- [ ] Unit test verifies `0 not in VALID_PART_IDS`
- [ ] Unit test verifies `len(VALID_PART_IDS) == 39`

### AC-5: CLI entry point works

- [ ] Running `python -m c5_forecasting --help` prints available commands
- [ ] A `health-check` or `version` subcommand returns successfully with no errors
- [ ] CLI uses Typer
- [ ] CLI configures structlog on startup

### AC-6: Linting and testing pass

- [ ] `ruff check src/ tests/` passes with zero errors
- [ ] `ruff format --check src/ tests/` passes
- [ ] `mypy src/` passes (may use `--ignore-missing-imports` initially)
- [ ] `pytest tests/` discovers and runs at least the domain constants tests
- [ ] All checks can be run via documented commands in README.md

### AC-7: README.md documents setup

- [ ] README.md exists at repo root
- [ ] Documents: project purpose (1 paragraph), prerequisites (Python 3.11, Poetry), setup commands, how to run tests, how to run linting, how to run the CLI
- [ ] Explicitly states: "MVP scope excludes any UI or web application"

### AC-8: .env.example exists

- [ ] `.env.example` exists with placeholder entries for any environment overrides
- [ ] At minimum contains `C5_LOG_LEVEL=INFO` and `C5_DATA_DIR=data/`

## Test Expectations

| Test | Type | Description |
|---|---|---|
| `test_valid_part_ids_excludes_zero` | unit | `0 not in VALID_PART_IDS` |
| `test_valid_part_ids_count` | unit | `len(VALID_PART_IDS) == 39` |
| `test_part_columns_match_ids` | unit | each column name maps to an ID in `VALID_PART_IDS` |
| `test_cli_help` | integration | CLI `--help` exits 0 |
| `test_cli_health_check` | integration | health-check subcommand exits 0 |

## Implementation Notes

1. Use `poetry init` or write `pyproject.toml` directly. Use `src/` layout.
2. All `__init__.py` files can be empty initially.
3. The CLI can start with just `app = typer.Typer()` and one health-check command.
4. Do not implement ingestion, validation, or any pipeline logic — that belongs to Story 1.2+.
5. The raw CSV must not be modified, moved, or reformatted.

## Definition of Done

- [ ] All acceptance criteria pass
- [ ] All tests pass
- [ ] Linting and type checking pass
- [ ] README documents all setup/run commands
- [ ] The PO can clone the repo, run `poetry install`, and execute `pytest` and the CLI health check successfully
