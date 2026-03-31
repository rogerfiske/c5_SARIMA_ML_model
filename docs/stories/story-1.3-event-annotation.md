# Story 1.3 — Implement Event Annotation and Anomaly Policy

**Epic:** 1 — Foundation, Dataset Contract, and Canary Forecast Flow
**PRD refs:** FR4, FR5, FR22, FR27
**Status:** Complete
**Depends on:** Story 1.2

## Objective

Enrich the validated raw dataset with domain-context columns that classify each
row by its daily part-usage total. PO-reviewed exception dates receive explicit
labels; unknown non-30 totals are soft-flagged as unreviewed exceptions. Raw
count values are never modified.

## Acceptance Criteria

- [x] Event annotation config (`configs/datasets/event_annotations.yaml`)
      encodes all 9 PO-reviewed exception dates
- [x] Annotation module adds 5 enrichment columns: `row_total`, `total_class`,
      `is_exception_day`, `domain_event_label`, `quality_flags`
- [x] Standard rows (total == 30) classified as `standard_output`
- [x] Reviewed exception dates classified as `reviewed_exception` with label
- [x] Unknown non-30 totals classified as `unreviewed_exception` with warning
- [x] Raw count values are never modified by annotation
- [x] Anomaly policy documented in `docs/anomaly-policy.md`
- [x] CLI command `annotate-dataset` runs end-to-end
- [x] All quality gates pass (ruff, mypy, pytest)

## Deliverables

### Config

| File | Purpose |
|---|---|
| `configs/datasets/event_annotations.yaml` | 9 reviewed exception dates with labels and categories |

### Source modules

| File | Purpose |
|---|---|
| `src/c5_forecasting/data/annotation.py` | Config loader, annotation enrichment, result dataclass |
| `src/c5_forecasting/cli/main.py` | Added `annotate-dataset` command |

### Documentation

| File | Purpose |
|---|---|
| `docs/anomaly-policy.md` | Classification scheme, enrichment columns, governance |
| `docs/stories/story-1.3-event-annotation.md` | This story spec |

### Tests

| File | Tests | Type |
|---|---|---|
| `tests/unit/test_annotation.py` | 18 | Unit |
| `tests/integration/test_annotation.py` | 11 | Integration |

## Reviewed Exception Dates

| Date | Total | Label | Category |
|---|---|---|---|
| 2008-12-25 | 20 | Christmas — reduced output | reduced_output |
| 2009-12-25 | 20 | Christmas — reduced output | reduced_output |
| 2010-12-25 | 20 | Christmas — reduced output | reduced_output |
| 2011-07-03 | 25 | Reduced output | reduced_output |
| 2011-08-28 | 25 | Reduced output | reduced_output |
| 2011-12-25 | 25 | Christmas — reduced output | reduced_output |
| 2012-05-15 | 35 | Additional output | additional_output |
| 2012-11-29 | 35 | Additional output | additional_output |
| 2012-12-25 | 25 | Christmas — reduced output | reduced_output |

## Enrichment Columns Added

| Column | Type | Description |
|---|---|---|
| `row_total` | int | Sum of P_1..P_39 |
| `total_class` | str | `standard_output` / `reviewed_exception` / `unreviewed_exception` |
| `is_exception_day` | bool | True if total != 30 |
| `domain_event_label` | str | Human-readable label from config |
| `quality_flags` | str | Pipe-delimited flags (e.g. `reviewed:reduced_output`) |

## Real Dataset Results

- Total rows: **6412**
- Standard days: **6403**
- Reviewed exceptions: **9**
- Unreviewed exceptions: **0**

## Test Results

- **82 tests** total (29 new for Story 1.3)
- All passing
- Quality gates: ruff check, ruff format, mypy, pytest — all green

## Dependencies Added

- `pyyaml ^6.0` (runtime)
- `types-PyYAML ^6.0` (dev, mypy stubs)
