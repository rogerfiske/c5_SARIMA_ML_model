# Story 2.3 — Baseline Ladder

## Summary

Implements a baseline ladder: 4 transparent baseline models under a common
evaluation interface, all run through the same backtest engine + metrics
pipeline (Stories 2.1-2.2), with comparison artifacts ranking them by
primary metrics.

## Baselines

| Model | Formula | Interpretation |
|-------|---------|----------------|
| `frequency_baseline` (existing) | rate = rows where P_i>0 / total rows | Global historical occurrence rate |
| `recency_weighted` (new) | weighted rate with decay=0.995 (~138-day half-life) | Recent evidence weighted more |
| `rolling_window` (new) | rate over last 365 rows only | Only last year matters |
| `uniform_baseline` (new) | score = 20/39 for all parts | Uninformative floor |

All satisfy `ScoringFunction` protocol: `(pd.DataFrame) -> list[PartScore]`.
All return exactly 39 scores in [0, 1].

## New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/models/recency_weighted.py` | Recency-weighted scoring |
| `src/c5_forecasting/models/rolling_window.py` | Rolling-window scoring |
| `src/c5_forecasting/models/uniform.py` | Uniform baseline scoring |
| `src/c5_forecasting/models/registry.py` | Name→function registry |
| `src/c5_forecasting/evaluation/ladder.py` | Ladder runner + comparison writers |
| `tests/unit/test_recency_weighted.py` | 10 unit tests |
| `tests/unit/test_rolling_window.py` | 10 unit tests |
| `tests/unit/test_uniform.py` | 7 unit tests |
| `tests/unit/test_model_registry.py` | 8 unit tests |
| `tests/unit/test_ladder.py` | 12 unit tests |
| `tests/integration/test_ladder.py` | 8 integration tests |

## Modified Files

| File | Change |
|------|--------|
| `src/c5_forecasting/cli/main.py` | Added `--model` option to `backtest`; new `ladder` command |

## CLI Commands

```bash
# Run a single model
python -m c5_forecasting backtest --model recency_weighted --step 100

# Run the full baseline ladder
python -m c5_forecasting ladder --step 100
```

## Comparison Artifacts

- `comparison_results.json` — full machine-readable comparison
- `comparison_summary.csv` — one row per model with all metric means
- `comparison_summary.md` — ranked table with interpretation notes

## Real-Data Results (step=2000, 4 folds)

| Rank | Model | nDCG@20 | WR@20 | Brier |
|------|-------|---------|-------|-------|
| 1 | frequency_baseline | 0.5707 | 0.6000 | 0.2456 |
| 2 | uniform_baseline | 0.5438 | 0.5583 | 0.2488 |
| 3 | recency_weighted | 0.4855 | 0.5250 | 0.2456 |
| 4 | rolling_window | 0.4628 | 0.5333 | 0.2469 |

Best model: `frequency_baseline`

## Quality Gates

- `ruff check src/ tests/` — PASS
- `ruff format --check src/ tests/` — PASS
- `mypy src/` — PASS (0 errors)
- `pytest tests/ -v` — 313 passed

## Test Count

- New tests: 55 (47 unit + 8 integration)
- Previous total: 258
- New total: **313**
