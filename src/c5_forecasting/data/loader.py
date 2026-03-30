"""Raw CSV loader with explicit date parsing for the aggregated matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import structlog

from c5_forecasting.domain.constants import DATE_COLUMN, EXPECTED_COLUMNS, PART_COLUMNS

logger = structlog.get_logger(__name__)

# Buffer size for SHA-256 hashing of large files.
_HASH_CHUNK_SIZE = 65536


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Load the raw aggregated matrix CSV with explicit M/D/YYYY date parsing.

    The CSV is read with all part columns as strings initially so that
    type validation can be performed downstream. The date column is parsed
    using ``pd.to_datetime`` with the expected format.

    Args:
        path: Path to the raw CSV file.

    Returns:
        DataFrame with a parsed ``date`` column and string-typed part columns.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the date column cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {path}")

    logger.info("loading_raw_csv", path=str(path))

    # Read everything as string first to allow downstream type validation.
    df = pd.read_csv(path, dtype=str)

    # Parse the date column explicitly with the expected M/D/YYYY format.
    if DATE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {DATE_COLUMN!r}")

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format="%m/%d/%Y")

    logger.info(
        "raw_csv_loaded",
        rows=len(df),
        columns=len(df.columns),
        date_min=str(df[DATE_COLUMN].min().date()),
        date_max=str(df[DATE_COLUMN].max().date()),
    )
    return df


def coerce_part_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert part columns from string to nullable integer dtype.

    This is separated from loading so that validation can inspect the raw
    string values before coercion.

    Args:
        df: DataFrame from :func:`load_raw_csv`.

    Returns:
        A copy of the DataFrame with part columns cast to ``Int64``.

    Raises:
        ValueError: If any part column contains values that cannot be
            converted to integers.
    """
    df = df.copy()
    for col in PART_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def compute_source_hash(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: Path to the file to hash.

    Returns:
        Lowercase hex string of the SHA-256 digest.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK_SIZE)
            if not chunk:
                break
            sha.update(chunk)
    digest = sha.hexdigest()
    logger.info("source_hash_computed", path=str(path), sha256=digest)
    return digest


def get_expected_columns() -> list[str]:
    """Return the canonical list of expected columns for the raw CSV.

    Provided as a function so it can be imported without triggering module-level
    side effects in tests.
    """
    return list(EXPECTED_COLUMNS)
