# Epic 1 — Foundation, Dataset Contract, and Canary Forecast Flow

## Goal

Establish the repository, dataset contract, event-annotation policy, and a fully
runnable canary next-day ranking pipeline so the team has a trustworthy foundation
and an executable vertical slice from raw data to top-20 output.

## Story Sequence

| Story | Title | Status | Depends On |
|---|---|---|---|
| 1.1 | Create the executable project skeleton | Ready | None |
| 1.2 | Implement raw ingestion and schema validation | Blocked on 1.1 | 1.1 |
| 1.3 | Implement event annotation and anomaly policy | Blocked on 1.2 | 1.2 |
| 1.4 | Build working datasets and curated experiment variant | Blocked on 1.3 | 1.3 |
| 1.5 | Deliver a canary next-day top-20 forecast flow | Blocked on 1.4 | 1.4 |

## Epic Completion Criteria

- The project is installable and runnable from documented commands
- The raw dataset passes schema validation with zero data-loss
- All confirmed exception dates (25-total and 35-total) are annotated, not discarded
- A validated working dataset with fingerprints exists
- A canary forecast produces a valid top-20 ranked part list (IDs 1..39 only, no 0)
- CSV, JSON, and Markdown forecast artifacts are generated
- All unit and integration tests pass

## PO Decisions Required Before Epic Starts

1. **Directory naming:** rename `doc/` to `docs/` or update all docs to use `doc/`?
2. **Git init:** should this story also `git init` the repo and create the first commit?

## Story Summaries

### Story 1.2 — Implement raw ingestion and schema validation

**PRD:** FR1, FR2, FR3, FR27
**Key deliverables:**
- CSV loader with explicit M/D/YYYY date parsing
- Schema validator: 40 columns (date + P_1..P_39), non-negative integers, unique dates
- Date continuity check and gap report
- Source hash (SHA-256 of raw CSV) persisted as manifest
- Structured validation report (JSON)
- Fail-fast on schema-breaking issues

**Test expectations:**
- Valid file passes cleanly
- Missing column raises error
- Duplicate date raises error
- Negative count raises error
- Non-integer value raises error
- Validation report contains expected fields

### Story 1.3 — Implement event annotation and anomaly policy

**PRD:** FR4, FR5, FR22, FR27
**Key deliverables:**
- Event annotation config (`configs/datasets/event_annotations.yaml`)
- Annotation enrichment: `total_class`, `is_exception_day`, `domain_event_label`, `quality_flags`
- All 9 confirmed exception dates encoded
- Soft-flag behavior: exceptions are warnings, not errors
- Anomaly policy documented in `docs/anomaly-policy.md`

**Test expectations:**
- Each confirmed exception date is annotated correctly
- Standard-total rows are classified as `standard_output`
- Unknown non-30 totals are flagged as `unreviewed_exception`
- Annotation does not modify raw count values

### Story 1.4 — Build working datasets and curated experiment variant

**PRD:** FR3, FR6, FR18
**Key deliverables:**
- Dataset builder produces `data/processed/raw_v1.parquet` with manifest
- Optional curated builder produces `data/processed/curated_v1.parquet`
- Dataset manifest (JSON) with: variant name, source hash, date range, row count, build timestamp, transform steps
- SHA-256 fingerprints for both variants
- Config-driven variant selection

**Test expectations:**
- Raw variant has same row count as source
- Curated variant documents all transformations
- Manifests are valid JSON with required fields
- Default path uses raw variant

### Story 1.5 — Deliver a canary next-day top-20 forecast flow

**PRD:** FR7-FR10, FR21, FR24, FR28, FR29
**Key deliverables:**
- Frequency baseline model: scores all 39 parts by historical occurrence rate
- Ranking module: sorts scores, applies deterministic tie-breaking, selects top 20
- Hard validation: asserts 0 not in output, all IDs in 1..39
- Artifact generation: CSV (ranked table), JSON (machine-readable), Markdown (summary)
- Provenance: dataset fingerprint, config fingerprint, timestamp, run ID
- CLI command: `forecasting forecast-next-day`

**Test expectations:**
- Output contains exactly 20 part IDs
- All output IDs are in 1..39
- 0 never appears in output
- Ties are broken deterministically (same input = same output)
- All three artifact formats are generated
- Provenance fields are populated
