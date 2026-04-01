# Story 3.2 — Count-Aware Candidate Family (Negative Binomial GLM)

## Summary

Adds a count-aware challenger model: per-part Negative Binomial GLM via
statsmodels GLM with a NegativeBinomial family. Each of the 39 part columns
is modelled independently using autoregressive lag features (lag-1 and lag-7),
forecasts are min-max normalised to [0, 1], and the top-20 ranked part IDs
are produced through the existing ScoringFunction protocol and backtest
framework.

## 1. Design and Approach

- 39 independent Negative Binomial GLMs, one per P_1..P_39 count series
- Autoregressive features: lag-1 (yesterday's count) + lag-7 (same weekday)
- Min-max normalisation across all 39 raw 1-step-ahead forecasts
- Fallback cascade: NegBinom(lag1+lag7) → NegBinom(lag1 only) → series mean → 0.0
- All negative forecasts clamped to 0 before normalisation
- Uniform fallback score (TOP_K/39) when all forecasts are identical
- Conforms to ScoringFunction protocol: `(pd.DataFrame) -> list[PartScore]`

## 2. Statistical Family — Why Negative Binomial?

Part counts are low-valued integers (mostly 0–4), zero-heavy, and
overdispersed (variance > mean). A Poisson GLM assumes mean = variance,
which underestimates uncertainty for these data. The Negative Binomial
family adds an overdispersion parameter α that allows variance to exceed
the mean, producing better-calibrated count predictions.

Zero-inflated variants were considered but not needed: the fallback cascade
handles parts with mostly-zero series, and NegBinom already accommodates
excess zeros through overdispersion.

## 3. Input Series

Per-part historical count series: `df[col].to_numpy(dtype=float, na_value=0.0)`
for each column P_1 through P_39. NaN values are replaced with 0 before
feature construction. Lag features are filled with 0 for early rows where
the lag is unavailable.

## 4. Leakage Enforcement

The backtest engine (Story 2.1) provides the training slice — only data up to
and including the cutoff date. The NegBinom GLM receives only this slice and
makes a 1-step-ahead forecast. No future data is accessible.

## 5. CLI Commands

```bash
# Run all 6 models (including negbinom_glm) through ladder + comparison
python -m c5_forecasting compare --step 2000

# Run just negbinom_glm through backtest
python -m c5_forecasting backtest --model negbinom_glm --step 2000
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
| 4 | sarima | 0.5250 | 0.5750 | 0.2808 | blocked_below_delta |
| 5 | recency_weighted | 0.4855 | 0.5250 | 0.2456 | blocked_below_delta |
| 6 | rolling_window | 0.4628 | 0.5333 | 0.2469 | blocked_below_delta |

NegBinom GLM ranks 3rd of 6 models. It beats SARIMA on nDCG@20 (0.5356 vs
0.5250) and is substantially faster (~1s total vs ~49s for SARIMA across all
folds). Its WR@20 (0.5667) is competitive. The higher Brier score (0.2744)
indicates less calibrated probability estimates compared to simpler baselines,
which is expected for a model that produces sharper predictions.

## 8. Champion-Candidate Status

NegBinom GLM is **not** the champion candidate in this run. The
frequency_baseline remains the best model. NegBinom GLM enters the ladder
as a challenger for future improvement (additional features, regularization,
or alternative link functions).

## 9. Fallback Usage

In the real-data smoke test, all 39 parts converged with the primary
NegBinom GLM (lag1+lag7) — no fallback paths were triggered. The fallback
cascade exists as safety for:
- Very short series (< 2 rows): returns series value or 0.0
- Convergence failure with lag7: retries with lag1 only
- Complete convergence failure: falls back to series mean

## 10. New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/models/negbinom_glm.py` | NegBinom GLM scoring function + helpers |
| `tests/unit/test_negbinom_glm.py` | 14 unit tests |
| `tests/integration/test_negbinom_backtest.py` | 7 integration tests |

## 11. Modified Files

| File | Change |
|------|--------|
| `src/c5_forecasting/models/registry.py` | Registered `"negbinom_glm"` (5→6 models) |
| `tests/unit/test_model_registry.py` | Updated count 5→6, added "negbinom_glm" to expected set |
| `tests/unit/test_ladder.py` | Updated count 5→6, added "negbinom_glm" to expected set |
| `tests/integration/test_ladder.py` | Updated count 5→6 |

## Quality Gates

- `ruff check src/ tests/` — PASS
- `ruff format --check src/ tests/` — PASS
- `mypy src/` — PASS (0 errors)
- `pytest tests/ -v` — 387 passed

## Test Count

- New tests: 21 (14 unit + 7 integration)
- Previous total: 366
- New total: **387**
