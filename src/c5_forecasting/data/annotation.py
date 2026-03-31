"""Event annotation and anomaly-policy enrichment for the raw dataset.

This module implements Story 1.3: it reads PO-reviewed exception dates from
a YAML config and enriches a validated DataFrame with domain-context columns.

Key invariants:
- Raw count values are NEVER modified.
- Exception dates are valid operating-condition rows, not data defects.
- Unknown non-30 totals are soft-flagged as ``unreviewed_exception``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
import yaml

from c5_forecasting.domain.constants import DATE_COLUMN, PART_COLUMNS, STANDARD_DAILY_TOTAL

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Annotation column names
# ---------------------------------------------------------------------------
COL_ROW_TOTAL = "row_total"
COL_TOTAL_CLASS = "total_class"
COL_IS_EXCEPTION = "is_exception_day"
COL_EVENT_LABEL = "domain_event_label"
COL_QUALITY_FLAGS = "quality_flags"

ANNOTATION_COLUMNS = [
    COL_ROW_TOTAL,
    COL_TOTAL_CLASS,
    COL_IS_EXCEPTION,
    COL_EVENT_LABEL,
    COL_QUALITY_FLAGS,
]

# ---------------------------------------------------------------------------
# Total-class labels
# ---------------------------------------------------------------------------
CLASS_STANDARD = "standard_output"
CLASS_REVIEWED = "reviewed_exception"
CLASS_UNREVIEWED = "unreviewed_exception"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
@dataclass
class ExceptionEntry:
    """A single PO-reviewed exception date."""

    date: str
    total: int
    label: str
    category: str


@dataclass
class EventAnnotationConfig:
    """Parsed event-annotation configuration."""

    standard_daily_total: int = STANDARD_DAILY_TOTAL
    reviewed_exceptions: list[ExceptionEntry] = field(default_factory=list)

    @property
    def exception_date_set(self) -> set[str]:
        """Set of YYYY-MM-DD date strings for fast lookup."""
        return {e.date for e in self.reviewed_exceptions}

    def get_exception(self, date_str: str) -> ExceptionEntry | None:
        """Lookup a reviewed exception by date string."""
        for entry in self.reviewed_exceptions:
            if entry.date == date_str:
                return entry
        return None


def load_annotation_config(path: Path) -> EventAnnotationConfig:
    """Load and parse the event annotations YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Parsed :class:`EventAnnotationConfig`.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config file is malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotation config not found: {path}")

    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Annotation config must be a YAML mapping, got {type(raw).__name__}")

    standard_total = raw.get("standard_daily_total", STANDARD_DAILY_TOTAL)
    exceptions_raw = raw.get("reviewed_exceptions", [])

    entries: list[ExceptionEntry] = []
    for i, item in enumerate(exceptions_raw):
        required = {"date", "total", "label", "category"}
        missing = required - set(item.keys())
        if missing:
            raise ValueError(
                f"reviewed_exceptions[{i}] missing required fields: {sorted(missing)}"
            )
        entries.append(
            ExceptionEntry(
                date=str(item["date"]),
                total=int(item["total"]),
                label=str(item["label"]),
                category=str(item["category"]),
            )
        )

    logger.info(
        "annotation_config_loaded",
        path=str(path),
        standard_total=standard_total,
        reviewed_count=len(entries),
    )

    return EventAnnotationConfig(
        standard_daily_total=standard_total,
        reviewed_exceptions=entries,
    )


# ---------------------------------------------------------------------------
# Annotation enrichment
# ---------------------------------------------------------------------------
@dataclass
class AnnotationResult:
    """Structured result of the annotation enrichment run."""

    row_count: int
    standard_count: int
    reviewed_exception_count: int
    unreviewed_exception_count: int
    annotation_columns_added: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "row_count": self.row_count,
            "standard_count": self.standard_count,
            "reviewed_exception_count": self.reviewed_exception_count,
            "unreviewed_exception_count": self.unreviewed_exception_count,
            "annotation_columns_added": self.annotation_columns_added,
            "warnings": self.warnings,
        }


def annotate_dataset(
    df: pd.DataFrame,
    config: EventAnnotationConfig,
) -> tuple[pd.DataFrame, AnnotationResult]:
    """Enrich a validated DataFrame with event-annotation columns.

    This function adds five columns to the DataFrame:
    - ``row_total``: sum of part columns for each row
    - ``total_class``: one of ``standard_output``, ``reviewed_exception``,
      ``unreviewed_exception``
    - ``is_exception_day``: boolean flag for any non-standard total
    - ``domain_event_label``: human-readable label from config (empty for standard)
    - ``quality_flags``: pipe-delimited quality flags

    Raw count values are NEVER modified.

    Args:
        df: Validated DataFrame with date and P_1..P_39 columns.
        config: Parsed event-annotation configuration.

    Returns:
        Tuple of (enriched DataFrame copy, AnnotationResult summary).
    """
    df_out = df.copy()

    # Compute row totals from part columns present in the DataFrame.
    # Part columns may still be string-typed (from load_raw_csv), so coerce
    # to numeric for summation.  This does NOT modify the original part columns
    # in df_out — we only use the numeric values for the row_total computation.
    part_cols = [c for c in PART_COLUMNS if c in df_out.columns]
    numeric_parts = df_out[part_cols].apply(pd.to_numeric, errors="coerce")
    df_out[COL_ROW_TOTAL] = numeric_parts.sum(axis=1).astype(int)

    # Normalize dates for lookup
    date_strings = df_out[DATE_COLUMN].dt.strftime("%Y-%m-%d")

    # Build exception lookup dict for O(1) access
    exception_map: dict[str, ExceptionEntry] = {e.date: e for e in config.reviewed_exceptions}

    # Classify each row
    total_classes: list[str] = []
    is_exception: list[bool] = []
    event_labels: list[str] = []
    quality_flags_list: list[str] = []

    standard_count = 0
    reviewed_count = 0
    unreviewed_count = 0
    warnings: list[str] = []

    for idx, (date_str, row_total) in enumerate(
        zip(date_strings, df_out[COL_ROW_TOTAL], strict=True)
    ):
        flags: list[str] = []

        if row_total == config.standard_daily_total:
            # Standard operating day
            total_classes.append(CLASS_STANDARD)
            is_exception.append(False)
            event_labels.append("")
            standard_count += 1
        elif date_str in exception_map:
            # PO-reviewed exception
            entry = exception_map[date_str]
            total_classes.append(CLASS_REVIEWED)
            is_exception.append(True)
            event_labels.append(entry.label)
            flags.append(f"reviewed:{entry.category}")
            reviewed_count += 1
        else:
            # Unknown non-standard total — soft-flag as unreviewed
            total_classes.append(CLASS_UNREVIEWED)
            is_exception.append(True)
            event_labels.append("")
            flags.append("unreviewed_exception")
            unreviewed_count += 1
            warnings.append(f"Row {idx}: date={date_str} total={row_total} — unreviewed exception")

        quality_flags_list.append("|".join(flags))

    df_out[COL_TOTAL_CLASS] = total_classes
    df_out[COL_IS_EXCEPTION] = is_exception
    df_out[COL_EVENT_LABEL] = event_labels
    df_out[COL_QUALITY_FLAGS] = quality_flags_list

    if unreviewed_count > 0:
        logger.warning(
            "unreviewed_exceptions_found",
            count=unreviewed_count,
            first_5=warnings[:5],
        )

    logger.info(
        "annotation_complete",
        total_rows=len(df_out),
        standard=standard_count,
        reviewed=reviewed_count,
        unreviewed=unreviewed_count,
    )

    result = AnnotationResult(
        row_count=len(df_out),
        standard_count=standard_count,
        reviewed_exception_count=reviewed_count,
        unreviewed_exception_count=unreviewed_count,
        annotation_columns_added=list(ANNOTATION_COLUMNS),
        warnings=warnings,
    )

    return df_out, result
