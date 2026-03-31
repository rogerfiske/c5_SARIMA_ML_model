"""Working dataset builder — produces Parquet variants with manifests.

This module implements Story 1.4: it builds versioned working datasets from
the validated, annotated raw data and writes machine-readable manifests for
provenance tracking.

Two variants are supported:

- **raw** (default): validated + annotated + type-coerced, all rows preserved.
- **curated**: raw variant with explicit, auditable transform steps applied.

Key invariants:
- No rows are silently dropped.  Curated transforms are declared and logged.
- Reviewed exception rows are preserved in both variants.
- Each output includes a JSON manifest with source SHA-256, variant name,
  row count, date range, build timestamp, and transform steps.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from c5_forecasting.data.annotation import (
    CLASS_UNREVIEWED,
    COL_TOTAL_CLASS,
    EventAnnotationConfig,
    annotate_dataset,
)
from c5_forecasting.data.loader import coerce_part_columns, compute_source_hash, load_raw_csv
from c5_forecasting.data.validation import validate_raw_dataset
from c5_forecasting.domain.constants import DATE_COLUMN

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Variant names
# ---------------------------------------------------------------------------
VARIANT_RAW = "raw"
VARIANT_CURATED = "curated"
VALID_VARIANTS = {VARIANT_RAW, VARIANT_CURATED}
DEFAULT_VARIANT = VARIANT_RAW


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
@dataclass
class DatasetManifest:
    """Machine-readable manifest for a built working dataset."""

    variant_name: str
    source_path: str
    source_sha256: str
    output_path: str
    output_sha256: str
    row_count: int
    column_count: int
    date_min: str
    date_max: str
    build_timestamp: str
    transform_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary suitable for JSON."""
        return {
            "variant_name": self.variant_name,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "build_timestamp": self.build_timestamp,
            "transform_steps": self.transform_steps,
        }


def write_manifest(manifest: DatasetManifest, output_dir: Path) -> Path:
    """Write a dataset manifest as JSON.

    Args:
        manifest: The completed manifest.
        output_dir: Directory to write into.

    Returns:
        Path to the written JSON manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{manifest.variant_name}_v1_manifest.json"
    path = output_dir / filename
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2)
    logger.info("manifest_written", path=str(path))
    return path


def _compute_parquet_hash(path: Path) -> str:
    """Compute SHA-256 of a Parquet file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest()


# ---------------------------------------------------------------------------
# Raw variant builder
# ---------------------------------------------------------------------------
def build_raw_dataset(
    csv_path: Path,
    annotation_config: EventAnnotationConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, DatasetManifest]:
    """Build the raw working dataset: validated, annotated, type-coerced.

    All rows are preserved.  No transforms are applied beyond:
    1. Schema validation (fail-fast on errors)
    2. Part-column type coercion (str -> Int64)
    3. Event annotation (adds 5 enrichment columns)

    Args:
        csv_path: Path to the raw CSV.
        annotation_config: Parsed event-annotation configuration.
        output_dir: Directory to write the Parquet file and manifest.

    Returns:
        Tuple of (DataFrame, DatasetManifest).

    Raises:
        ValueError: If validation fails.
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    source_sha256 = compute_source_hash(csv_path)

    # Validate
    val_result = validate_raw_dataset(csv_path)
    if not val_result.is_valid:
        raise ValueError(f"Validation failed: {val_result.errors}")

    # Load and coerce types
    df = load_raw_csv(csv_path)
    df = coerce_part_columns(df)

    # Annotate
    df_annotated, ann_result = annotate_dataset(df, annotation_config)

    # Write Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "raw_v1.parquet"
    df_annotated.to_parquet(parquet_path, index=False, engine="pyarrow")
    output_sha256 = _compute_parquet_hash(parquet_path)

    transform_steps = [
        "schema_validation",
        "part_column_coercion_str_to_Int64",
        "event_annotation",
    ]

    manifest = DatasetManifest(
        variant_name=VARIANT_RAW,
        source_path=str(csv_path),
        source_sha256=source_sha256,
        output_path=str(parquet_path),
        output_sha256=output_sha256,
        row_count=len(df_annotated),
        column_count=len(df_annotated.columns),
        date_min=str(df_annotated[DATE_COLUMN].min().date()),
        date_max=str(df_annotated[DATE_COLUMN].max().date()),
        build_timestamp=timestamp,
        transform_steps=transform_steps,
    )

    logger.info(
        "raw_dataset_built",
        rows=manifest.row_count,
        columns=manifest.column_count,
        output=str(parquet_path),
    )

    return df_annotated, manifest


# ---------------------------------------------------------------------------
# Curated variant builder
# ---------------------------------------------------------------------------
def build_curated_dataset(
    csv_path: Path,
    annotation_config: EventAnnotationConfig,
    output_dir: Path,
) -> tuple[pd.DataFrame, DatasetManifest]:
    """Build the curated working dataset with explicit, auditable transforms.

    Starts from the raw annotated dataset and applies:
    1. **Exclude unreviewed exceptions** — rows where ``total_class`` is
       ``unreviewed_exception`` are removed.  Reviewed exceptions are preserved.
    2. **Sort by date** — ensures strict chronological order.

    All transform steps are declared in the manifest.

    Args:
        csv_path: Path to the raw CSV.
        annotation_config: Parsed event-annotation configuration.
        output_dir: Directory to write the Parquet file and manifest.

    Returns:
        Tuple of (DataFrame, DatasetManifest).
    """
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    source_sha256 = compute_source_hash(csv_path)

    # Build from raw first
    val_result = validate_raw_dataset(csv_path)
    if not val_result.is_valid:
        raise ValueError(f"Validation failed: {val_result.errors}")

    df = load_raw_csv(csv_path)
    df = coerce_part_columns(df)
    df_annotated, _ = annotate_dataset(df, annotation_config)

    transform_steps = [
        "schema_validation",
        "part_column_coercion_str_to_Int64",
        "event_annotation",
    ]

    # --- Curated transform 1: exclude unreviewed exceptions ---
    pre_count = len(df_annotated)
    df_curated = df_annotated[df_annotated[COL_TOTAL_CLASS] != CLASS_UNREVIEWED].copy()
    dropped = pre_count - len(df_curated)
    transform_steps.append(f"exclude_unreviewed_exceptions(dropped={dropped})")

    # --- Curated transform 2: sort by date ---
    df_curated = df_curated.sort_values(DATE_COLUMN).reset_index(drop=True)
    transform_steps.append("sort_by_date")

    # Write Parquet
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "curated_v1.parquet"
    df_curated.to_parquet(parquet_path, index=False, engine="pyarrow")
    output_sha256 = _compute_parquet_hash(parquet_path)

    manifest = DatasetManifest(
        variant_name=VARIANT_CURATED,
        source_path=str(csv_path),
        source_sha256=source_sha256,
        output_path=str(parquet_path),
        output_sha256=output_sha256,
        row_count=len(df_curated),
        column_count=len(df_curated.columns),
        date_min=str(df_curated[DATE_COLUMN].min().date()),
        date_max=str(df_curated[DATE_COLUMN].max().date()),
        build_timestamp=timestamp,
        transform_steps=transform_steps,
    )

    logger.info(
        "curated_dataset_built",
        rows=manifest.row_count,
        columns=manifest.column_count,
        dropped_unreviewed=dropped,
        output=str(parquet_path),
    )

    return df_curated, manifest
