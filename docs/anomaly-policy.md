# Anomaly Policy — Event Annotation and Exception Handling

## Purpose

This document defines how the c5_forecasting platform identifies, classifies,
and handles daily part-usage totals that deviate from the standard operating
output of **30 parts per day**.

## Core Principles

1. **Raw data is immutable.** Annotation enriches rows with domain context but
   never modifies the original count values.
2. **Exception dates are valid data.** Days with non-standard totals (20, 25,
   35) represent real operating conditions — not data defects.
3. **Reviewed exceptions are PO-approved.** Each confirmed exception date has
   been reviewed by the Product Owner and assigned a label and category.
4. **Unknown exceptions are soft-flagged.** Any future non-30 total that does
   not appear in the reviewed list is classified as an `unreviewed_exception`
   and logged as a warning — never silently dropped or normalized.

## Classification Scheme

Every row in the annotated dataset receives a `total_class` label:

| `total_class`          | Condition                                     | Action              |
|------------------------|-----------------------------------------------|---------------------|
| `standard_output`      | Row total == 30                               | Normal processing   |
| `reviewed_exception`   | Row total != 30 AND date is in reviewed list  | Annotated with label|
| `unreviewed_exception` | Row total != 30 AND date NOT in reviewed list | Soft-flagged, warned|

## Enrichment Columns

The annotation module adds five columns to each row:

| Column               | Type    | Description                                        |
|----------------------|---------|----------------------------------------------------|
| `row_total`          | int     | Sum of P_1 through P_39 for the row                |
| `total_class`        | str     | One of: `standard_output`, `reviewed_exception`, `unreviewed_exception` |
| `is_exception_day`   | bool    | `True` if row total differs from standard (30)     |
| `domain_event_label` | str     | Human-readable label from config (empty if standard)|
| `quality_flags`      | str     | Pipe-delimited flags (e.g. `reviewed:reduced_output`)|

## Reviewed Exception Dates

The following 9 dates have been confirmed by the PO (see
`configs/datasets/event_annotations.yaml`):

| Date       | Total | Label                        | Category           |
|------------|-------|------------------------------|--------------------|
| 2008-12-25 | 20    | Christmas — reduced output   | reduced_output     |
| 2009-12-25 | 20    | Christmas — reduced output   | reduced_output     |
| 2010-12-25 | 20    | Christmas — reduced output   | reduced_output     |
| 2011-07-03 | 25    | Reduced output               | reduced_output     |
| 2011-08-28 | 25    | Reduced output               | reduced_output     |
| 2011-12-25 | 25    | Christmas — reduced output   | reduced_output     |
| 2012-05-15 | 35    | Additional output            | additional_output  |
| 2012-11-29 | 35    | Additional output            | additional_output  |
| 2012-12-25 | 25    | Christmas — reduced output   | reduced_output     |

## Downstream Impact

- **Working datasets (Story 1.4):** The annotation columns travel with the
  data into `data/processed/` Parquet files, allowing downstream models and
  analyses to filter or weight exception days appropriately.
- **Model training:** Models may choose to include or exclude exception days
  based on the `total_class` or `is_exception_day` flag.
- **Reporting:** Exception counts and labels appear in validation and
  annotation reports for human review.

## Governance

- Only the Product Owner may add dates to the reviewed-exception list.
- New exception dates should be added to `configs/datasets/event_annotations.yaml`
  with a label and category.
- The `unreviewed_exception` soft-flag ensures no non-standard day is silently
  ignored until it has been reviewed and classified.
