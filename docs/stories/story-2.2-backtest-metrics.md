# Story 2.2 — Ranking and Calibration Metrics

**Epic:** 2 — Model Evaluation & Selection
**Status:** Complete
**Branch:** main

## Goal

Implement count-weighted ranking metrics, binary overlap metrics, and
calibration metrics that consume backtest results and produce structured
reports distinguishing primary vs secondary metrics.

## Requirements (PRD acceptance criteria)

1. The evaluation layer computes **nDCG@20** and **Weighted Recall@20** (primary).
2. The evaluation layer computes binary **Precision@20** and **Recall@20** (secondary).
3. Probability-capable models expose **Brier score** (calibration metric).
4. Metric definitions are documented and tested.
5. A comparison report clearly distinguishes primary and secondary metrics.

## Metric Definitions

### Primary

| Metric | Formula | Notes |
|--------|---------|-------|
| nDCG@20 | DCG / IDCG; DCG = sum(rel(pred_i) / log2(i+1)) | rel = actual count; IDCG from top-20 ideal; 0 if IDCG=0 |
| Weighted Recall@20 | sum(count in pred AND actual) / sum(all active counts) | Count-weighted, not binary |
| Brier Score | (1/39) sum((score - indicator)^2) | Lower is better; requires full 39-score vector |

### Secondary

| Metric | Formula |
|--------|---------|
| Precision@20 | hits / 20 |
| Recall@20 | hits / actual_count |
| Jaccard@20 | intersection / union |

## Design Decisions

- **Pure functions** for each metric: independently testable, composable.
- **Per-fold then aggregate**: `FoldMetrics` per fold, `MetricSummary` across folds.
- **Backward compat via defaults**: new `BacktestFold` fields have defaults; all
  210 prior tests continue passing unchanged.
- **No numpy**: pure Python math (only `math.log2`).
- **Primary/secondary labeling** in report writer and `MetricSummary.to_dict()`.

## New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/evaluation/metrics.py` | Pure metric functions + orchestrators |
| `src/c5_forecasting/evaluation/metric_report.py` | JSON + Markdown metric report writers |
| `tests/unit/test_metrics.py` | 40 unit tests |
| `tests/integration/test_metrics.py` | 8 integration tests |

## Modified Files

| File | Change |
|------|--------|
| `src/c5_forecasting/evaluation/backtest.py` | Added `actual_part_counts`, `all_scores` to BacktestFold; extended `extract_actual_parts()` |
| `src/c5_forecasting/evaluation/artifacts.py` | Metrics columns in CSV; primary/secondary section in Markdown |
| `src/c5_forecasting/cli/main.py` | Backtest command now computes and displays metrics |

## CLI Usage

```bash
poetry run python -m c5_forecasting backtest --step 2000
```

Now outputs:
- nDCG@20 mean, Weighted Recall@20 mean, Brier mean in console
- `metric_report.json` and `metric_report.md` in artifacts directory

## Real-Data Baseline Results (step=2000)

| Metric | Value |
|--------|-------|
| nDCG@20 mean | 0.5707 |
| Weighted Recall@20 mean | 0.6000 |
| Brier Score mean | 0.2456 |
| Total folds | 4 |

## Acceptance Criteria

- [x] nDCG@20 and Weighted Recall@20 computed
- [x] Binary Precision@20 and Recall@20 computed
- [x] Brier score computed for probability-capable models
- [x] Metric definitions documented and tested
- [x] Comparison report distinguishes primary and secondary metrics
- [x] All quality gates pass (ruff check, ruff format, mypy, pytest)
- [x] 48 new tests (40 unit + 8 integration)
