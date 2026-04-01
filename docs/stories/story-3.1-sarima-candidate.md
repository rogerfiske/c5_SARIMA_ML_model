# Story 3.1 — SARIMA/SARIMAX Candidate Family

## Summary

Adds the first non-baseline challenger model: per-part ARIMA(1,1,1) via
statsmodels SARIMAX. Each of the 39 part columns is modelled independently,
forecasts are min-max normalised to [0, 1], and the top-20 ranked part IDs
are produced through the existing ScoringFunction protocol and backtest
framework.

## 1. Design and Approach

- 39 independent ARIMA(1,1,1) models, one per P_1..P_39 count series
- Min-max normalisation across all 39 raw 1-step-ahead forecasts
- Fallback cascade: ARIMA(1,1,1) → ARIMA(0,1,0) → last observed value → 0.0
- All negative forecasts clamped to 0 before normalisation
- Uniform fallback score (TOP_K/39) when all forecasts are identical
- Conforms to ScoringFunction protocol: `(pd.DataFrame) -> list[PartScore]`

## 2. SARIMA vs SARIMAX

Uses `statsmodels.tsa.statespace.SARIMAX` class internally but with no
exogenous variables (pure univariate ARIMA). No seasonal component in v1 —
ARIMA(1,1,1) not SARIMA(p,d,q)(P,D,Q,s) — daily data has no obvious
short-period seasonality. The model is registered as `"sarima"` in the
model registry.

## 3. Input Series

Per-part historical count series: `train_df[col].to_numpy(dtype=float, na_value=0.0)`
for each column P_1 through P_39. NaN values are replaced with 0 before fitting.

## 4. Leakage Enforcement

The backtest engine (Story 2.1) provides the training slice — only data up to
and including the cutoff date. The SARIMA model receives only this slice and
makes a 1-step-ahead forecast. No future data is accessible.

## 5. CLI Commands

```bash
# Run all 5 models (including sarima) through ladder + comparison
python -m c5_forecasting compare --step 2000

# Run just sarima through backtest
python -m c5_forecasting backtest --model sarima --step 2000
```

## 6. Artifact Paths

- Ladder results: `artifacts/backtests/ladder/`
- Comparison reports: `artifacts/comparisons/latest/`
- Per-model backtest: `artifacts/backtests/{model_name}/`

## 7. Real-Data Results (step=2000, 4 folds, no existing champion)

| Rank | Model | nDCG@20 | WR@20 | Brier | Verdict |
|------|-------|---------|-------|-------|---------|
| 1 | frequency_baseline | 0.5707 | 0.6000 | 0.2456 | no_champion |
| 2 | uniform_baseline | 0.5438 | 0.5583 | 0.2488 | blocked_below_delta |
| 3 | sarima | 0.5250 | 0.5750 | 0.2808 | blocked_below_delta |
| 4 | recency_weighted | 0.4855 | 0.5250 | 0.2456 | blocked_below_delta |
| 5 | rolling_window | 0.4628 | 0.5333 | 0.2469 | blocked_below_delta |

SARIMA ranks 3rd of 5 models. It beats recency_weighted and rolling_window
but falls short of frequency_baseline and uniform_baseline on nDCG@20.
However, its WR@20 (0.5750) is competitive — second only to frequency_baseline.
The higher Brier score (0.2808) indicates less calibrated probability estimates.

## 8. Champion-Candidate Status

SARIMA is **not** the champion candidate in this run. The frequency_baseline
remains the best model. SARIMA enters the ladder as a challenger for future
tuning (order search, seasonal components, exogenous variables).

## 9. New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/models/sarima.py` | SARIMA scoring function + helpers |
| `tests/unit/test_sarima.py` | 12 unit tests |
| `tests/integration/test_sarima_backtest.py` | 7 integration tests |

## 10. Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Added `statsmodels = "^0.14"` |
| `src/c5_forecasting/models/registry.py` | Registered `"sarima"` |
| `tests/unit/test_model_registry.py` | Updated count 4 → 5, added "sarima" to expected set |
| `tests/unit/test_ladder.py` | Updated count 4 → 5, added "sarima" to expected set |
| `tests/integration/test_ladder.py` | Updated count 4 → 5 |

## Quality Gates

- `ruff check src/ tests/` — PASS
- `ruff format --check src/ tests/` — PASS
- `mypy src/` — PASS (0 errors)
- `pytest tests/ -v` — 366 passed

## Test Count

- New tests: 19 (12 unit + 7 integration)
- Previous total: 347
- New total: **366**
