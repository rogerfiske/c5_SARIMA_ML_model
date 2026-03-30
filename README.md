# c5_forecasting

Next-event ranking forecast platform for daily part-demand prediction. The system
ingests a daily aggregated matrix of part usage counts (parts 1-39), builds features,
trains and evaluates multiple candidate model families via rolling backtests, and
produces a ranked top-20 next-day part list. **MVP scope excludes any UI or web
application.**

## Prerequisites

- Python 3.11 (tested with 3.11.9)
- [Poetry](https://python-poetry.org/) 2.x

## Setup

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

## Usage

```bash
# Show available commands
python -m c5_forecasting --help

# Verify installation
python -m c5_forecasting health-check

# Or use the Poetry-installed script
forecasting --help
forecasting health-check
```

## Development

```bash
# Run tests
poetry run pytest

# Run linting
poetry run ruff check src/ tests/

# Check formatting
poetry run ruff format --check src/ tests/

# Run type checks
poetry run mypy src/
```

## Project Structure

See [docs/architecture/source-tree.md](docs/architecture/source-tree.md) for the
full layout. Key directories:

- `src/c5_forecasting/` - Main package source
- `tests/` - Unit, integration, and regression tests
- `data/raw/` - Immutable source data
- `configs/` - Human-edited configuration files
- `artifacts/` - Generated run outputs (not tracked in git)
- `docs/` - PRD, architecture, stories, and reference materials
