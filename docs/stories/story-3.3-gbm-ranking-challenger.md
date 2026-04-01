# Story 3.3 — ML Ranking Challenger (Gradient-Boosted Trees)

## Summary

Adds the first ML-based challenger model: per-part
HistGradientBoostingRegressor (scikit-learn) with 6 engineered features.
Each of the 39 part columns is modelled independently using lag, rolling-mean,
and calendar features. Forecasts are min-max normalised to [0, 1] and the
top-20 ranked part IDs are produced through the existing ScoringFunction
protocol and backtest framework.

## 1. ML Design

39 independent gradient-boosted tree regressors, one per P_1..P_39 count
series. Each regressor predicts the next day's count given today's features.
Training split: X = features[0..n-2], y = counts[1..n-1]. Prediction uses
the last row's features to forecast 1-step-ahead. This per-part architecture
follows the established pattern from SARIMA and NegBinom GLM.

## 2. Model Family and Feature Set

**Algorithm**: `sklearn.ensemble.HistGradientBoostingRegressor` — the
fastest tree-based regressor in scikit-learn with native NaN handling and
deterministic output via `random_state=42`.

**Hyperparameters**: max_iter=100, max_depth=4, min_samples_leaf=20.

**Features per part (6 total)**:

| Feature | Description | Leakage-safe? |
|---------|-------------|---------------|
| lag_1 | Yesterday's count | Yes — shift(1) |
| lag_7 | Same weekday last week | Yes — shift(7) |
| lag_14 | Same weekday 2 weeks ago | Yes — shift(14) |
| rolling_mean_7 | 7-day rolling mean | Yes — shift(1) then rolling |
| rolling_mean_30 | 30-day rolling mean | Yes — shift(1) then rolling |
| day_of_week | Calendar day 0–6 | Yes — calendar, not data |

## 3. Input Series and Engineered Features

Per-part historical count series: `df[col].to_numpy(dtype=float, na_value=0.0)`
for each column P_1 through P_39. NaN values are replaced with 0 before
feature construction. Early rows with unavailable lags/windows are filled
with 0. The `date` column provides day-of-week as a calendar feature.

Rolling means are computed with `shift(1).rolling(window, min_periods=1).mean()`
to ensure the current row's count is excluded, preventing same-row leakage.

## 4. Leakage Enforcement

Three layers of leakage prevention:
1. **Backtest engine** (Story 2.1): provides only the training slice up to
   the cutoff date. No future data is accessible.
2. **Feature construction**: All features use `shift()` to look backward only.
   Rolling means use `shift(1)` before the rolling window.
3. **Training split**: X = features[:-1], y = series[1:] ensures the model
   predicts the next day's count from today's (and earlier) data only.

## 5. CLI Commands

```bash
# Run all 7 models (including gbm_ranking) through ladder + comparison
python -m c5_forecasting compare --step 2000

# Run just gbm_ranking through backtest
python -m c5_forecasting backtest --model gbm_ranking --step 2000
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
| 3 | negbinom_glm | 0.5356 | 0.5667 | 0.2744 | blocked_below_delta |
| 4 | **gbm_ranking** | **0.5260** | **0.5333** | **0.2951** | blocked_below_delta |
| 5 | sarima | 0.5250 | 0.5750 | 0.2808 | blocked_below_delta |
| 6 | recency_weighted | 0.4855 | 0.5250 | 0.2456 | blocked_below_delta |
| 7 | rolling_window | 0.4628 | 0.5333 | 0.2469 | blocked_below_delta |

GBM ranking places 4th of 7 models. It edges out SARIMA on nDCG@20
(0.5260 vs 0.5250) and is substantially faster (~3.5s/fold vs ~12s/fold
for SARIMA). Its higher Brier score (0.2951) indicates less calibrated
probability estimates, as expected for a regression model producing sharper
point predictions. With only 4 evaluation folds (step=2000), results may
shift with finer-grained evaluation.

## 8. Champion-Candidate Status

GBM ranking is **not** the champion candidate. The frequency_baseline
remains the best model. GBM ranking enters the ladder as an ML challenger
for future improvement (more features, hyperparameter tuning, XGBoost
upgrade, cross-part features).

## 9. Fallback Usage

In the real-data smoke test, all 39 parts converged with the primary
HistGradientBoostingRegressor — **zero fallback invocations**. The GBM
is extremely robust with these data characteristics. The fallback cascade
(series mean → 0.0) exists as safety for very short series or convergence
failure edge cases.

## 10. New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/models/gbm_ranking.py` | GBM scoring function + feature engineering + helpers |
| `tests/unit/test_gbm_ranking.py` | 14 unit tests |
| `tests/integration/test_gbm_backtest.py` | 7 integration tests |

## 11. Modified Files

| File | Change |
|------|--------|
| `pyproject.toml` | Added `scikit-learn = "^1.6"` |
| `src/c5_forecasting/models/registry.py` | Registered `"gbm_ranking"` (6→7 models) |
| `tests/unit/test_model_registry.py` | Updated count 6→7, added "gbm_ranking" to expected set |
| `tests/unit/test_ladder.py` | Updated count 6→7, added "gbm_ranking" to expected set |
| `tests/integration/test_ladder.py` | Updated count 6→7 |

## Quality Gates

- `ruff check src/ tests/` — PASS
- `ruff format --check src/ tests/` — PASS
- `mypy src/` — PASS (0 errors)
- `pytest tests/ -v` — 408 passed

## Test Count

- New tests: 21 (14 unit + 7 integration)
- Previous total: 387
- New total: **408**
