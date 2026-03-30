"""Domain constants for the c5_forecasting platform.

Hard business rules:
- Valid part identifiers for ranking are 1 through 39 only.
- 0 is a historical count value meaning 'part not used that day'.
  It must NEVER appear as a predicted next-event part identifier.
- The forecast output selects the top 20 ranked valid part IDs.
"""

from __future__ import annotations

MIN_PART_ID: int = 1
"""Smallest valid part identifier."""

MAX_PART_ID: int = 39
"""Largest valid part identifier."""

VALID_PART_IDS: frozenset[int] = frozenset(range(MIN_PART_ID, MAX_PART_ID + 1))
"""The complete set of part IDs that may appear in a forecast ranking."""

TOP_K: int = 20
"""Number of top-ranked part IDs in the published forecast package."""

PART_COLUMNS: list[str] = [f"P_{i}" for i in range(MIN_PART_ID, MAX_PART_ID + 1)]
"""Expected column names for part counts in the source dataset: P_1 .. P_39."""

DATE_COLUMN: str = "date"
"""Name of the date column in the source dataset."""

EXPECTED_COLUMNS: list[str] = [DATE_COLUMN] + PART_COLUMNS
"""Full expected column list for the raw CSV: date, P_1, P_2, ..., P_39."""

RAW_CSV_DATE_FORMAT: str = "%m/%d/%Y"
"""Date format used in the raw source CSV (M/D/YYYY, e.g. 9/8/2008)."""

STANDARD_DAILY_TOTAL: int = 30
"""The most common daily part-usage total under normal operating conditions."""
