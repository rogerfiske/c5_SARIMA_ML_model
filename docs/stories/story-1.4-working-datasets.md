# Story 1.4 — Build Working Datasets and Curated Experiment Variant

**Epic:** 1 — Foundation, Dataset Contract, and Canary Forecast Flow
**PRD refs:** FR3, FR6, FR18
**Status:** Complete
**Depends on:** Story 1.3

## Objective

Build versioned working datasets (Parquet) from the validated, annotated raw
CSV. Support two variants — **raw** (default) and **curated** — with explicit,
auditable transform steps and machine-readable manifests for provenance.

## Acceptance Criteria

- [x] Raw variant preserves all 6412 rows (no drops)
- [x] Curated variant applies explicit transforms, documented in manifest
- [x] Reviewed exception rows preserved in both variants
- [x] Manifests are valid JSON with: variant name, source SHA-256, output
      SHA-256, row count, column count, date range, build timestamp, transforms
- [x] Config-driven variant selection (C5_DATASET_VARIANT, default: raw)
- [x] CLI command `build-dataset` runs end-to-end for both variants
- [x] All quality gates pass (ruff, mypy, pytest)

## Deliverables

### Source modules

| File | Purpose |
|---|---|
| `src/c5_forecasting/data/dataset_builder.py` | Raw/curated builders, manifest, Parquet I/O |
| `src/c5_forecasting/cli/main.py` | Added `build-dataset` command |
| `src/c5_forecasting/config/settings.py` | Added `dataset_variant` setting |

### Documentation

| File | Purpose |
|---|---|
| `docs/stories/story-1.4-working-datasets.md` | This story spec |

### Tests

| File | Tests | Type |
|---|---|---|
| `tests/unit/test_dataset_builder.py` | 18 | Unit |
| `tests/integration/test_dataset_builder.py` | 16 | Integration |

## Dataset Variants

### Raw variant (default)

- **Output:** `data/processed/raw_v1.parquet`
- **Row count:** 6412 (identical to source)
- **Transform steps:**
  1. `schema_validation`
  2. `part_column_coercion_str_to_Int64`
  3. `event_annotation`
- **Columns:** 40 original + 5 annotation = 45 total

### Curated variant

- **Output:** `data/processed/curated_v1.parquet`
- **Row count:** 6412 (0 unreviewed exceptions in current data, so same as raw)
- **Transform steps:**
  1. `schema_validation`
  2. `part_column_coercion_str_to_Int64`
  3. `event_annotation`
  4. `exclude_unreviewed_exceptions(dropped=0)`
  5. `sort_by_date`
- **Columns:** 45 (same as raw)
- **Key difference:** Future unreviewed exception rows would be excluded;
  reviewed exception rows are always preserved.

## Manifest Schema

```json
{
  "variant_name": "raw",
  "source_path": "data/raw/c5_aggregated_matrix.csv",
  "source_sha256": "0333d82f881674d5a832b19050c898b918300960f39ccfe0d44707d5feb7c78a",
  "output_path": "data/processed/raw_v1.parquet",
  "output_sha256": "<64-char hex>",
  "row_count": 6412,
  "column_count": 45,
  "date_min": "2008-09-08",
  "date_max": "2026-03-29",
  "build_timestamp": "20260331T...",
  "transform_steps": ["schema_validation", "part_column_coercion_str_to_Int64", "event_annotation"]
}
```

## Default Variant

The default downstream dataset variant is **raw**, controlled by:
- CLI: `--variant raw` (default if omitted)
- Environment: `C5_DATASET_VARIANT=raw`
- Config: `AppSettings.dataset_variant = "raw"`

## Test Results

- **119 tests** total (37 new for Story 1.4)
- All passing
- Quality gates: ruff check, ruff format, mypy, pytest — all green
