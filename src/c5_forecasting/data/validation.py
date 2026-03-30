"""Schema validation and data-quality checks for the raw aggregated matrix.

This module implements fail-fast validation for schema-breaking issues and
informational checks for domain-level data quality. It does NOT implement
Story 1.3 event annotation or anomaly policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import structlog

from c5_forecasting.data.loader import coerce_part_columns, compute_source_hash, load_raw_csv
from c5_forecasting.domain.constants import DATE_COLUMN, EXPECTED_COLUMNS, PART_COLUMNS

logger = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    """Structured result of a raw dataset validation run."""

    source_path: str
    source_sha256: str
    row_count: int
    column_count: int
    date_min: str
    date_max: str
    expected_columns: list[str]
    actual_columns: list[str]
    # Fail-fast errors (any of these = invalid dataset)
    errors: list[str] = field(default_factory=list)
    # Informational warnings (dataset is usable but noteworthy)
    warnings: list[str] = field(default_factory=list)
    # Date continuity
    missing_dates: list[str] = field(default_factory=list)
    missing_date_count: int = 0
    # Duplicate dates
    duplicate_dates: list[str] = field(default_factory=list)
    duplicate_date_count: int = 0
    # Row total summary
    row_total_min: int = 0
    row_total_max: int = 0
    row_total_mean: float = 0.0
    distinct_row_totals: list[int] = field(default_factory=list)
    # Overall verdict
    is_valid: bool = False

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary suitable for JSON output."""
        return {
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "expected_columns": self.expected_columns,
            "actual_columns": self.actual_columns,
            "errors": self.errors,
            "warnings": self.warnings,
            "missing_dates": self.missing_dates,
            "missing_date_count": self.missing_date_count,
            "duplicate_dates": self.duplicate_dates,
            "duplicate_date_count": self.duplicate_date_count,
            "row_total_min": self.row_total_min,
            "row_total_max": self.row_total_max,
            "row_total_mean": round(self.row_total_mean, 4),
            "distinct_row_totals": self.distinct_row_totals,
            "is_valid": self.is_valid,
        }


def validate_raw_dataset(path: Path) -> ValidationResult:
    """Run all validation checks against the raw CSV.

    This is the main entry point for Story 1.2 validation. It performs:
    1. Column presence and ordering check
    2. Null detection
    3. Integer type and non-negative value checks
    4. Date uniqueness check
    5. Date monotonicity and continuity check
    6. Row-total summary (informational, not fail-fast)
    7. Source SHA-256 fingerprint

    Args:
        path: Path to the raw CSV file.

    Returns:
        A :class:`ValidationResult` with errors, warnings, and metadata.
    """
    source_hash = compute_source_hash(path)
    df = load_raw_csv(path)

    result = ValidationResult(
        source_path=str(path),
        source_sha256=source_hash,
        row_count=len(df),
        column_count=len(df.columns),
        date_min=str(df[DATE_COLUMN].min().date()) if len(df) > 0 else "",
        date_max=str(df[DATE_COLUMN].max().date()) if len(df) > 0 else "",
        expected_columns=list(EXPECTED_COLUMNS),
        actual_columns=list(df.columns),
    )

    # --- 1. Column check ---
    _check_columns(df, result)

    # --- 2. Null detection ---
    _check_nulls(df, result)

    # If columns are wrong, further checks are unreliable.
    if result.errors:
        logger.error("validation_failed_early", errors=result.errors)
        return result

    # --- 3. Integer type and non-negative check ---
    df_typed = coerce_part_columns(df)
    _check_integer_values(df, df_typed, result)

    # --- 4. Duplicate dates ---
    _check_duplicate_dates(df, result)

    # --- 5. Date continuity ---
    _check_date_continuity(df, result)

    # --- 6. Row totals (informational) ---
    if not result.errors:
        _compute_row_totals(df_typed, result)

    # --- Verdict ---
    result.is_valid = len(result.errors) == 0

    if result.is_valid:
        logger.info("validation_passed", rows=result.row_count, warnings=len(result.warnings))
    else:
        logger.error("validation_failed", errors=result.errors)

    return result


def _check_columns(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check that all expected columns are present."""
    expected_set = set(EXPECTED_COLUMNS)
    actual_set = set(df.columns)

    missing = expected_set - actual_set
    extra = actual_set - expected_set

    if missing:
        result.errors.append(f"Missing columns: {sorted(missing)}")
    if extra:
        result.warnings.append(f"Extra columns (ignored): {sorted(extra)}")

    # Check ordering matches expected
    actual_expected = [c for c in df.columns if c in expected_set]
    if actual_expected != EXPECTED_COLUMNS:
        result.warnings.append("Column order does not match expected order")


def _check_nulls(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check for null/missing values in any column."""
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if len(cols_with_nulls) > 0:
        for col, count in cols_with_nulls.items():
            result.errors.append(f"Column {col!r} has {count} null value(s)")


def _check_integer_values(
    df_raw: pd.DataFrame,
    df_typed: pd.DataFrame,
    result: ValidationResult,
) -> None:
    """Check that all part columns contain non-negative integers."""
    for col in PART_COLUMNS:
        if col not in df_raw.columns:
            continue

        # Check for values that could not be coerced to integer
        coerced_nulls = df_typed[col].isna() & df_raw[col].notna()
        if coerced_nulls.any():
            bad_count = int(coerced_nulls.sum())
            bad_rows = df_raw.index[coerced_nulls].tolist()[:5]
            result.errors.append(
                f"Column {col!r} has {bad_count} non-integer value(s) at rows: {bad_rows}"
            )

        # Check for negative values
        negatives = df_typed[col].dropna() < 0
        if negatives.any():
            neg_count = int(negatives.sum())
            result.errors.append(f"Column {col!r} has {neg_count} negative value(s)")


def _check_duplicate_dates(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check for duplicate dates in the dataset."""
    date_counts = df[DATE_COLUMN].value_counts()
    duplicates = date_counts[date_counts > 1]
    if len(duplicates) > 0:
        dup_dates = sorted(str(d.date()) for d in duplicates.index)
        result.duplicate_dates = dup_dates
        result.duplicate_date_count = len(dup_dates)
        result.errors.append(f"Found {len(dup_dates)} duplicate date(s): {dup_dates}")


def _check_date_continuity(df: pd.DataFrame, result: ValidationResult) -> None:
    """Check for gaps in the daily date sequence.

    The check creates a complete date range from min to max and identifies
    any calendar dates that are absent from the dataset.
    """
    if len(df) == 0:
        return

    dates = df[DATE_COLUMN].sort_values()
    date_min = dates.min()
    date_max = dates.max()

    full_range = pd.date_range(start=date_min, end=date_max, freq="D")
    actual_dates = set(dates.dt.normalize())
    expected_dates = set(full_range.normalize())

    missing = sorted(expected_dates - actual_dates)
    result.missing_dates = [str(d.date()) for d in missing]
    result.missing_date_count = len(missing)

    if missing:
        result.warnings.append(f"Found {len(missing)} missing calendar date(s)")
        logger.warning("date_gaps_found", count=len(missing), first_5=result.missing_dates[:5])
    else:
        logger.info("date_continuity_ok", span_days=len(full_range), actual_rows=len(dates))

    # Check monotonicity
    if not dates.is_monotonic_increasing:
        result.warnings.append("Dates are not in monotonically increasing order")


def _compute_row_totals(df: pd.DataFrame, result: ValidationResult) -> None:
    """Compute row-total summary statistics (informational, not fail-fast)."""
    part_cols = [c for c in PART_COLUMNS if c in df.columns]
    totals = df[part_cols].sum(axis=1)
    result.row_total_min = int(totals.min())
    result.row_total_max = int(totals.max())
    result.row_total_mean = float(totals.mean())
    result.distinct_row_totals = sorted(int(t) for t in totals.unique())
