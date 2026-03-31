# Story 1.5 — Deliver a Canary Next-Day Top-20 Forecast Flow

**Epic:** 1 — Foundation, Dataset Contract, and Canary Forecast Flow
**PRD refs:** FR7-FR10, FR21, FR24, FR28, FR29
**Status:** Complete
**Depends on:** Story 1.4

## Objective

Deliver the first end-to-end forecast pipeline: score all 39 parts using a
frequency baseline model, rank with deterministic tie-breaking, validate the
output, and generate CSV, JSON, and Markdown artifacts with full provenance.

## Acceptance Criteria

- [x] Frequency baseline scores all 39 parts by historical occurrence rate
- [x] Ranking module sorts scores, applies deterministic tie-breaking, selects top 20
- [x] Hard validation: asserts 0 not in output, all IDs in 1..39, exactly 20
- [x] CSV artifact: `rank, part_id, score` columns, 20 rows
- [x] JSON artifact: machine-readable with `provenance` and `rankings` keys
- [x] Markdown artifact: human-readable summary table
- [x] Provenance: dataset fingerprint, config fingerprint, timestamp, run ID
- [x] CLI command `forecast-next-day` runs end-to-end
- [x] All quality gates pass (ruff, mypy, pytest)

## Deliverables

### Source modules

| File | Purpose |
|---|---|
| `src/c5_forecasting/models/baseline.py` | Frequency baseline scoring model |
| `src/c5_forecasting/ranking/ranker.py` | Ranking with deterministic tie-breaking + validation |
| `src/c5_forecasting/pipelines/forecast.py` | Pipeline orchestration + artifact generation |
| `src/c5_forecasting/cli/main.py` | Added `forecast-next-day` command |

### Documentation

| File | Purpose |
|---|---|
| `docs/stories/story-1.5-canary-forecast.md` | This story spec |

### Tests

| File | Tests | Type |
|---|---|---|
| `tests/unit/test_baseline.py` | 8 | Unit |
| `tests/unit/test_ranker.py` | 15 | Unit |
| `tests/unit/test_forecast_pipeline.py` | 10 | Unit |
| `tests/integration/test_forecast.py` | 12 | Integration |

## Baseline Ranking Method

The **frequency baseline** (`frequency_baseline`) scores each part P_i as the
fraction of historical rows where P_i > 0:

```
score(P_i) = count(rows where P_i > 0) / total_rows
```

This produces a score in [0, 1] for each of the 39 parts. Parts that appear
more frequently across the dataset receive higher scores.

## Deterministic Tie-Breaking

When two parts have identical scores, the tie is broken by **lower part ID
first**. The sort key is `(-score, part_id)`, which guarantees:
- Higher scores rank first
- Among equal scores, smaller part IDs rank first
- Same input always produces the same output

## Output Validation

`validate_forecast()` enforces these hard constraints (raises
`ForecastValidationError` on violation):

1. Exactly K (20) ranked entries
2. All part IDs in VALID_PART_IDS (1..39)
3. No 0 in output
4. No duplicate part IDs

## CLI Command

```bash
poetry run python -m c5_forecasting forecast-next-day [--variant raw|curated]
```

Default variant is `raw`. The command:
1. Reads the dataset manifest for fingerprints
2. Loads the working dataset Parquet
3. Runs the full pipeline (score → rank → validate → write artifacts)
4. Prints a summary with run ID, model, top 5, and artifact paths

## Artifact Output Paths

- `artifacts/runs/latest/forecast.csv`
- `artifacts/runs/latest/forecast.json`
- `artifacts/runs/latest/forecast.md`

## Forecast Schema per Artifact

### CSV (`forecast.csv`)

```
rank,part_id,score
1,23,0.5733
2,2,0.571117
...
20,9,0.560044
```

### JSON (`forecast.json`)

```json
{
  "provenance": {
    "run_id": "<uuid>",
    "run_timestamp": "20260331T...",
    "model_name": "frequency_baseline",
    "dataset_variant": "raw",
    "dataset_fingerprint": "<sha256>",
    "source_fingerprint": "<sha256>",
    "config_fingerprint": "<sha256-prefix>",
    "k": 20,
    "dataset_row_count": 6412,
    "dataset_date_min": "2008-09-08",
    "dataset_date_max": "2026-03-29"
  },
  "rankings": [
    {"rank": 1, "part_id": 23, "score": 0.5733},
    ...
  ],
  "artifacts": ["...forecast.csv", "...forecast.json", "...forecast.md"]
}
```

### Markdown (`forecast.md`)

Human-readable summary with provenance header and a ranked table:

```
| Rank | Part ID | Score |
|------|---------|-------|
| 1    | 23      | 0.573300 |
...
```

## Sample Top-20 Output (Real Dataset)

| Rank | Part ID | Score |
|------|---------|-------|
| 1 | 23 | 0.573300 |
| 2 | 2 | 0.571117 |
| 3 | 5 | 0.570805 |
| 4 | 11 | 0.570805 |
| 5 | 24 | 0.570649 |
| 6 | 15 | 0.569869 |
| 7 | 38 | 0.569713 |
| 8 | 8 | 0.568465 |
| 9 | 10 | 0.567998 |
| 10 | 13 | 0.566906 |
| 11 | 34 | 0.566594 |
| 12 | 29 | 0.565814 |
| 13 | 17 | 0.565658 |
| 14 | 19 | 0.565658 |
| 15 | 4 | 0.563475 |
| 16 | 22 | 0.563007 |
| 17 | 30 | 0.562227 |
| 18 | 1 | 0.561759 |
| 19 | 21 | 0.560200 |
| 20 | 9 | 0.560044 |

## Test Results

- **164 tests** total (45 new for Story 1.5)
- All passing
- Quality gates: ruff check, ruff format, mypy, pytest — all green
