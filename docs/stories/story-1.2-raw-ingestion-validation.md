# Story 1.2 — Raw Ingestion and Schema Validation

**Epic:** 1 — Foundation, Dataset Contract, and Canary Forecast Flow
**Status:** COMPLETE
**Priority:** P0
**Depends on:** Story 1.1

## User Story

As a research analyst,
I want reliable ingestion and validation of the historical matrix,
so that all downstream experiments begin from a verified dataset.

## PRD Traceability

- FR1: Ingest daily historical dataset with schema `date, P_1..P_39`
- FR2: Validate column presence, data types, duplicate dates, date continuity, non-negative integers
- FR3: Preserve an immutable raw dataset copy with fingerprinted reference
- FR27: Fail fast on schema-breaking data issues

## Deliverables

| Deliverable | Location |
|---|---|
| Raw CSV loader with M/D/YYYY date parsing | `src/c5_forecasting/data/loader.py` |
| Schema validation module | `src/c5_forecasting/data/validation.py` |
| JSON validation report writer | `src/c5_forecasting/data/report.py` |
| CLI `validate-raw` command | `src/c5_forecasting/cli/main.py` |
| Unit tests (loader) | `tests/unit/test_loader.py` |
| Unit tests (validation) | `tests/unit/test_validation.py` |
| Unit tests (report) | `tests/unit/test_report.py` |
| Integration tests (real CSV + CLI) | `tests/integration/test_validate_raw.py` |

## Acceptance Criteria — Verified

- [x] Raw ingestion works against the real CSV
- [x] Schema validation passes on the source dataset
- [x] Explicit parsing of M/D/YYYY is covered by tests
- [x] Duplicate dates are detected
- [x] Missing columns are detected
- [x] Negative values are detected
- [x] Non-integer values are detected
- [x] Date continuity is checked and reported
- [x] Source SHA-256 fingerprint is generated and persisted
- [x] Validation report emitted in machine-readable JSON
- [x] CLI command exits successfully on the valid dataset
- [x] Lint, type checks, and tests all pass (53 tests)

## Verified Dataset Facts

| Fact | Value |
|---|---|
| Row count | 6412 |
| Date range | 2008-09-08 to 2026-03-29 |
| Missing calendar dates | 0 |
| Duplicate dates | 0 |
| Column contract | date + P_1..P_39 (40 columns) |
| Distinct row totals | [20, 25, 30, 35] |
| Row total mean | 29.9938 |
| Source SHA-256 | `0333d82f881674d5a832b19050c898b918300960f39ccfe0d44707d5feb7c78a` |
| Validation artifacts | `artifacts/manifests/validation_report_<timestamp>.json` |
