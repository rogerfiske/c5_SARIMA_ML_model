# Story 2.1 — Rolling-Origin Backtesting Engine

**Epic:** 2 — Model Evaluation & Selection
**Status:** Complete
**Branch:** main

## Goal

Build a rolling-origin backtesting engine that evaluates forecasting methods
over sequential cutoff dates, using only historical data up to each cutoff —
no future leakage.

## Requirements

1. Accept a dataset and a ranking method (scoring function protocol).
2. Generate rolling evaluation windows across time with configurable
   `min_train_rows`, `step`, and optional `max_windows` cap.
3. For each cutoff: train on historical data up to cutoff, produce top-20
   forecast, compare to realized next event.
4. Store outputs in auditable format (JSON, CSV, Markdown).
5. Add a CLI command (`backtest`) with variant/step/min-train/max-windows options.
6. Prove chronological correctness and no future leakage via planted-signal test.
7. Emit machine-readable results for downstream metric computation (Story 2.2).
8. Preserve provenance (run ID, timestamps, fingerprints, config).

## Design Decisions

- **Expanding window** (anchored at row 0): frequency baseline benefits from
  more data; matches production use case.
- **Positional indexing** (`df.iloc`): more robust than date filtering; avoids
  edge cases with gaps/duplicates.
- **max_windows keeps most recent**: when capped, tail of window list is kept
  (recent performance more relevant).
- **Defensive sort**: `run_backtest` sorts df by date rather than requiring
  caller to sort.
- **No parallelization**: frequency baseline is cheap (~seconds for 6000+
  folds); keeps it simple and deterministic.
- **ScoringFunction Protocol**: `(pd.DataFrame) -> list[PartScore]` — satisfied
  by `compute_frequency_scores` without modification.

## New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/evaluation/backtest.py` | Core engine: dataclasses + window generation + run_backtest |
| `src/c5_forecasting/evaluation/artifacts.py` | Artifact writers (JSON, CSV, Markdown) |
| `tests/unit/test_backtest.py` | 22 unit tests |
| `tests/unit/test_backtest_artifacts.py` | 9 unit tests |
| `tests/integration/test_backtest.py` | 9 integration tests (real data + CLI) |

## Modified Files

| File | Change |
|------|--------|
| `src/c5_forecasting/evaluation/__init__.py` | Added module docstring |
| `src/c5_forecasting/cli/main.py` | Added `backtest` CLI command |

## Key Dataclasses

- **BacktestConfig** — min_train_rows (365), step (1), max_windows, k (20), model_name
- **BacktestFold** — per-fold result with predicted ranking, actual parts, hit count
- **BacktestSummary** — aggregate statistics (mean/min/max hit count, date ranges)
- **BacktestProvenance** — run_id, timestamp, model, variant, fingerprints, config
- **BacktestResult** — provenance + folds + summary + artifact paths

## CLI Usage

```bash
poetry run python -m c5_forecasting backtest \
  [--variant raw|curated] \
  [--min-train-rows 365] \
  [--step 1] \
  [--max-windows N]
```

## Test Coverage

- **Unit (31 tests):** Config defaults, window generation (step, cap, bounds),
  extract_actual_parts, leakage prevention (planted future-only signal),
  monotonic fold growth, hit_count correctness, determinism, provenance,
  JSON serializability, no-zero-in-any-fold, artifact schemas.
- **Integration (9 tests):** Real dataset backtest completes, no leakage on
  real data, all forecasts valid, plausible hit counts, artifacts written,
  JSON has folds, deterministic, CLI exits 0, CLI prints summary.

## Real-Data Baseline Results (step=100)

| Metric | Value |
|--------|-------|
| Total folds | 61 |
| Mean hit count | 11.33 |
| Min hit count | 8 |
| Max hit count | 15 |
| Cutoff range | 2009-09-07 to 2026-02-10 |

## Acceptance Criteria

- [x] All quality gates pass (ruff check, ruff format, mypy, pytest)
- [x] No future leakage (planted-signal test + real-data temporal assertions)
- [x] CLI command works end-to-end on real data
- [x] Three artifact formats produced (JSON, CSV, Markdown)
- [x] Provenance fully populated
- [x] 40 new tests (31 unit + 9 integration)
