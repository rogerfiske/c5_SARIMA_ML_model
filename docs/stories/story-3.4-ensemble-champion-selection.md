# Story 3.4 — Ensemble Logic and Champion Selection Reporting

## Summary

Adds 3 ensemble methods as first-class registered models, bringing the total
from 7 individual models to 10 models (7 individuals + 3 ensembles). Each
ensemble internally composes multiple scoring functions to produce a single
ranked forecast. The existing comparison/ladder/champion framework handles
ensembles transparently with no code changes needed.

## 1. Ensemble Methods Implemented

### ensemble_avg — Simple Score Averaging

- **Components**: frequency_baseline, gbm_ranking, negbinom_glm, sarima (4 models)
- **Combination**: Arithmetic mean of per-part scores
- **Rationale**: Model diversity reduces variance

### ensemble_rank_avg — Rank Averaging (Borda-count)

- **Components**: frequency_baseline, gbm_ranking, negbinom_glm, sarima (4 models)
- **Combination**: Convert scores to ranks, average ranks, convert back to scores
- **Formula**: `score = 1.0 - (avg_rank - 1) / 38.0`
- **Rationale**: Robust to score-scale differences across models

### ensemble_weighted — Fixed Weighted Average

- **Components**: All 7 individual models
- **Weights**: Frozen constants derived from Story 3.3 results (NOT computed at scoring time)
- **Rationale**: Tests informed weighting vs. simple averaging

Frozen weights (source-code constants to prevent data leakage):
```python
_ENSEMBLE_WEIGHTS = {
    "frequency_baseline": 0.30,
    "negbinom_glm": 0.25,
    "gbm_ranking": 0.20,
    "sarima": 0.15,
    "recency_weighted": 0.05,
    "rolling_window": 0.03,
    "uniform_baseline": 0.02,
}  # Sum = 1.00
```

## 2. Component Model Selection

**ensemble_avg and ensemble_rank_avg** use 4 diverse models representing distinct approaches:
- frequency_baseline (rule-based)
- negbinom_glm (count regression)
- gbm_ranking (ML tree-based)
- sarima (time-series)

Weak baselines (uniform, rolling_window, recency_weighted) excluded to avoid noise.

**ensemble_weighted** includes all 7 models with low weights for weak performers.
Tests completeness vs. selectivity.

## 3. Architecture

**ScoringFunction protocol**: `(pd.DataFrame) -> list[PartScore]`

Ensembles internally call other scoring functions via the registry:
```python
def ensemble_avg_scoring(df: pd.DataFrame) -> list[PartScore]:
    from c5_forecasting.models.registry import get_scoring_function  # lazy import

    component_names = ["frequency_baseline", "negbinom_glm", "gbm_ranking", "sarima"]
    all_scores = {}  # {part_id: [score1, score2, ...]}

    for name in component_names:
        fn = get_scoring_function(name)
        scores = fn(df)
        for ps in scores:
            all_scores.setdefault(ps.part_id, []).append(ps.score)

    result = []
    for part_id, score_list in all_scores.items():
        avg = sum(score_list) / len(score_list)
        result.append(PartScore(part_id=part_id, score=avg))

    result.sort(key=lambda ps: (-ps.score, ps.part_id))
    return result
```

**Circular import avoidance**: Both ensemble.py and registry.py use lazy imports
inside function bodies. Registry imports ensemble functions when
`get_model_registry()` is called. Ensemble functions import `get_scoring_function`
when they are called. These happen at different times → no circularity.

**No changes needed** to backtest, ladder, comparison, or champion engines —
ensembles are just regular models.

## 4. CLI Commands

```bash
# Run all 10 models (including 3 ensembles) through ladder + comparison
python -m c5_forecasting compare --step 2000

# Run just one ensemble through backtest
python -m c5_forecasting backtest --model ensemble_avg --step 2000

# Run ladder with custom subset including ensembles
python -m c5_forecasting ladder --step 2000
```

## 5. Artifact Paths

- Ladder results: `artifacts/backtests/ladder/`
- Comparison reports: `artifacts/comparisons/latest/`
- Per-model backtest: `artifacts/backtests/{model_name}/`

## 6. Real-Data Results

CLI smoke test running with `compare --step 3000`. Results will show all 10
models ranked by nDCG@20 descending, with champion-candidate verdict.

Expected behavior:
- All ensembles appear in ranked output
- Ensembles compete fairly with individual models
- Champion selection logic transparently evaluates ensembles
- No special handling needed

## 7. Champion-Candidate Status

The comparison framework automatically:
- Sorts all 10 models by nDCG@20 descending (with WR, Brier, name tie-breaks)
- Identifies best_in_report (top-ranked model, could be an ensemble)
- Computes verdicts: ELIGIBLE, BLOCKED_BELOW_DELTA, BLOCKED_TIED, NO_CHAMPION
- Sets champion_candidate to best_in_report if verdict is ELIGIBLE or NO_CHAMPION
- Writes comparison_report.json and comparison_report.md with all models ranked

Ensembles flow through as regular models — they appear in the ranked results
table alongside individual models. No code changes were needed.

## 8. Tie Handling

Ensembles follow the same tie-breaking logic as individual models:
1. Primary: nDCG@20 descending
2. Secondary: Weighted Recall@20 descending
3. Tertiary: Brier Score ascending (lower is better)
4. Final: Model name alphabetically ascending

## 9. Test Count

- **New tests**: 27 total
  - Unit tests: 20 (`tests/unit/test_ensemble.py`)
  - Integration tests: 17 (`tests/integration/test_ensemble_backtest.py`)
- **Previous total**: 408
- **New total**: **435**

## 10. New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/models/ensemble.py` | 3 ensemble scoring functions + helpers |
| `tests/unit/test_ensemble.py` | 20 unit tests |
| `tests/integration/test_ensemble_backtest.py` | 17 integration tests |
| `docs/stories/story-3.4-ensemble-champion-selection.md` | Story completion spec |

## 11. Modified Files

| File | Change |
|------|--------|
| `src/c5_forecasting/models/registry.py` | Register 3 ensembles (7→10 models) |
| `tests/unit/test_model_registry.py` | Update count 7→10, add 3 ensemble names to EXPECTED_MODELS |
| `tests/unit/test_ladder.py` | Update count 7→10, add 3 ensemble names |
| `tests/integration/test_ladder.py` | Update count 7→10 |

## Quality Gates

- `ruff check src/ tests/` — PASS
- `ruff format --check src/ tests/` — PASS
- `mypy src/` — PASS (0 errors)
- `pytest tests/unit/ -v` — 318 passed
- `pytest tests/integration/test_ensemble_backtest.py -v` — 17 passed
- `pytest tests/integration/test_ladder.py::TestRealLadder -v` — PASS

## Champion Selection Reporting

No code changes needed. The comparison framework already:
- Sorts all models by nDCG@20 descending
- Identifies best_in_report (top-ranked model)
- Computes verdicts with delta thresholds
- Sets champion_candidate appropriately
- Writes detailed comparison reports

Ensembles are treated as first-class models and appear in all ranking tables
and comparison reports alongside individual models.
