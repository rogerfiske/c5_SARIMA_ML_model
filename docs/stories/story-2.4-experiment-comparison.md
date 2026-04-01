# Story 2.4 — Experiment Comparison and Champion Gating

## Summary

Implements structured experiment comparison and champion-candidate gating on
top of the existing backtest/metric/ladder infrastructure (Stories 2.1–2.3).
Introduces configurable minimum-delta thresholds, a three-tier model status
(best-in-report / champion-candidate / actual champion), and explicit PO
approval before any champion promotion.

## Three-Tier Model Status

| Status | Meaning | Set By |
|--------|---------|--------|
| `best_in_report` | Highest nDCG@20 in current comparison | `compare_to_champion()` |
| `champion_candidate` | best_in_report AND beats champion by >= min_delta | `compare_to_champion()` |
| `actual_champion` | Promoted after PO approval; persisted to `champion.json` | `promote --confirm` CLI |

## New Files

| File | Purpose |
|------|---------|
| `src/c5_forecasting/evaluation/champion.py` | ChampionRecord dataclass, load/save/promote |
| `src/c5_forecasting/evaluation/comparison.py` | ComparisonConfig, CandidateVerdict, compare_to_champion(), report writers |
| `tests/unit/test_champion.py` | 11 unit tests |
| `tests/unit/test_comparison.py` | 15 unit tests |
| `tests/integration/test_champion_flow.py` | 8 integration tests |

## Modified Files

| File | Change |
|------|--------|
| `src/c5_forecasting/cli/main.py` | Added `compare`, `promote`, `champion` commands |

## CLI Commands

```bash
# Run ladder + compare against current champion
python -m c5_forecasting compare --step 2000

# Dry-run promotion (shows what would happen)
python -m c5_forecasting promote

# Actually promote with explicit approval
python -m c5_forecasting promote --confirm

# Show current champion
python -m c5_forecasting champion
```

## Comparison Artifacts

- `comparison_report.json` — full machine-readable comparison with verdicts
- `comparison_report.md` — human-readable ranked table with delta analysis

## Verdict Logic

| Condition | Verdict |
|-----------|---------|
| No existing champion | `no_champion` (best_in_report becomes candidate) |
| nDCG delta >= min_ndcg_delta (default 0.01) | `eligible` |
| 0 < nDCG delta < min_ndcg_delta | `blocked_below_delta` |
| nDCG delta == 0.0 | `blocked_tied` |

Only the best_in_report model can be champion_candidate. Non-best models
always receive a BLOCKED verdict regardless of their delta.

## Deterministic Tie-Breaking

Sort key: `(-nDCG, -WR@20, +Brier, model_name)` — 4-tuple total ordering.

## Real-Data Results (step=2000, 4 folds, no existing champion)

| Rank | Model | nDCG@20 | WR@20 | Brier | Verdict |
|------|-------|---------|-------|-------|---------|
| 1 | frequency_baseline | 0.5707 | 0.6000 | 0.2456 | no_champion |
| 2 | uniform_baseline | 0.5438 | 0.5583 | 0.2488 | blocked_below_delta |
| 3 | recency_weighted | 0.4855 | 0.5250 | 0.2456 | blocked_below_delta |
| 4 | rolling_window | 0.4628 | 0.5333 | 0.2469 | blocked_below_delta |

Champion candidate: `frequency_baseline` (first-ever promotion)

## Quality Gates

- `ruff check src/ tests/` — PASS
- `ruff format --check src/ tests/` — PASS
- `mypy src/` — PASS (0 errors)
- `pytest tests/ -v` — 347 passed

## Test Count

- New tests: 34 (26 unit + 8 integration)
- Previous total: 313
- New total: **347**
