"""Unit tests for the working dataset builder."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from c5_forecasting.data.annotation import (
    ANNOTATION_COLUMNS,
    CLASS_REVIEWED,
    CLASS_UNREVIEWED,
    COL_TOTAL_CLASS,
    EventAnnotationConfig,
    ExceptionEntry,
)
from c5_forecasting.data.dataset_builder import (
    DEFAULT_VARIANT,
    VARIANT_CURATED,
    VARIANT_RAW,
    DatasetManifest,
    build_curated_dataset,
    build_raw_dataset,
    write_manifest,
)
from c5_forecasting.domain.constants import DATE_COLUMN, PART_COLUMNS, STANDARD_DAILY_TOTAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_valid_csv(tmp_path: Path, rows: list[tuple[str, int]] | None = None) -> Path:
    """Write a minimal valid CSV with the given date/total pairs."""
    if rows is None:
        rows = [("9/8/2008", 30), ("9/9/2008", 30), ("9/10/2008", 30)]

    header = ",".join(["date"] + PART_COLUMNS)
    lines = [header]
    for date_str, total in rows:
        values = ",".join(["1" if i < total else "0" for i in range(39)])
        lines.append(f"{date_str},{values}")

    csv_path = tmp_path / "test.csv"
    csv_path.write_text("\n".join(lines))
    return csv_path


def _make_config(
    exceptions: list[tuple[str, int, str, str]] | None = None,
) -> EventAnnotationConfig:
    """Build a config with optional exception entries."""
    entries = [
        ExceptionEntry(date=d, total=t, label=lbl, category=c)
        for d, t, lbl, c in (exceptions or [])
    ]
    return EventAnnotationConfig(
        standard_daily_total=STANDARD_DAILY_TOTAL,
        reviewed_exceptions=entries,
    )


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------
class TestDatasetManifest:
    """Tests for DatasetManifest."""

    def test_to_dict(self) -> None:
        """to_dict should return all fields."""
        m = DatasetManifest(
            variant_name="raw",
            source_path="data/raw/test.csv",
            source_sha256="abc123",
            output_path="data/processed/raw_v1.parquet",
            output_sha256="def456",
            row_count=100,
            column_count=45,
            date_min="2020-01-01",
            date_max="2020-12-31",
            build_timestamp="20200101T000000Z",
            transform_steps=["step1", "step2"],
        )
        d = m.to_dict()
        assert d["variant_name"] == "raw"
        assert d["row_count"] == 100
        assert d["transform_steps"] == ["step1", "step2"]
        assert "source_sha256" in d
        assert "output_sha256" in d

    def test_write_manifest(self, tmp_path: Path) -> None:
        """write_manifest should produce a valid JSON file."""
        m = DatasetManifest(
            variant_name="raw",
            source_path="test.csv",
            source_sha256="abc",
            output_path="raw_v1.parquet",
            output_sha256="def",
            row_count=10,
            column_count=45,
            date_min="2020-01-01",
            date_max="2020-12-31",
            build_timestamp="20200101T000000Z",
            transform_steps=["validation"],
        )
        out_dir = tmp_path / "manifests"
        path = write_manifest(m, out_dir)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["variant_name"] == "raw"
        assert data["row_count"] == 10

    def test_manifest_creates_directory(self, tmp_path: Path) -> None:
        """write_manifest should create the output directory if needed."""
        m = DatasetManifest(
            variant_name="test",
            source_path="",
            source_sha256="",
            output_path="",
            output_sha256="",
            row_count=0,
            column_count=0,
            date_min="",
            date_max="",
            build_timestamp="",
        )
        deep_dir = tmp_path / "a" / "b" / "c"
        write_manifest(m, deep_dir)
        assert deep_dir.exists()


# ---------------------------------------------------------------------------
# Raw variant tests
# ---------------------------------------------------------------------------
class TestBuildRawDataset:
    """Tests for build_raw_dataset."""

    def test_raw_preserves_all_rows(self, tmp_path: Path) -> None:
        """Raw variant must preserve all rows from the source."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        df, manifest = build_raw_dataset(csv_path, config, output_dir)
        assert manifest.row_count == 3
        assert len(df) == 3

    def test_raw_output_file_exists(self, tmp_path: Path) -> None:
        """Raw variant must write a Parquet file."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        build_raw_dataset(csv_path, config, output_dir)
        assert (output_dir / "raw_v1.parquet").exists()

    def test_raw_has_annotation_columns(self, tmp_path: Path) -> None:
        """Raw variant must include annotation columns."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        df, _ = build_raw_dataset(csv_path, config, output_dir)
        for col in ANNOTATION_COLUMNS:
            assert col in df.columns

    def test_raw_manifest_fields(self, tmp_path: Path) -> None:
        """Raw manifest must have all required fields."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        _, manifest = build_raw_dataset(csv_path, config, output_dir)

        assert manifest.variant_name == VARIANT_RAW
        assert manifest.source_sha256
        assert manifest.output_sha256
        assert manifest.build_timestamp
        assert len(manifest.transform_steps) > 0

    def test_raw_manifest_transform_steps(self, tmp_path: Path) -> None:
        """Raw variant transforms should be validation, coercion, annotation."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        _, manifest = build_raw_dataset(csv_path, config, output_dir)
        assert "schema_validation" in manifest.transform_steps
        assert "event_annotation" in manifest.transform_steps

    def test_raw_preserves_exception_rows(self, tmp_path: Path) -> None:
        """Raw variant must include reviewed exception rows."""
        csv_path = _write_valid_csv(
            tmp_path,
            rows=[("12/25/2008", 20), ("9/8/2008", 30), ("9/9/2008", 30)],
        )
        config = _make_config([("2008-12-25", 20, "Christmas", "reduced_output")])
        output_dir = tmp_path / "processed"
        df, manifest = build_raw_dataset(csv_path, config, output_dir)
        assert manifest.row_count == 3
        reviewed = df[df[COL_TOTAL_CLASS] == CLASS_REVIEWED]
        assert len(reviewed) == 1

    def test_raw_parquet_roundtrip(self, tmp_path: Path) -> None:
        """Parquet file should be readable and match the DataFrame."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        df, _ = build_raw_dataset(csv_path, config, output_dir)
        df_read = pd.read_parquet(output_dir / "raw_v1.parquet")
        assert len(df_read) == len(df)
        assert list(df_read.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# Curated variant tests
# ---------------------------------------------------------------------------
class TestBuildCuratedDataset:
    """Tests for build_curated_dataset."""

    def test_curated_excludes_unreviewed(self, tmp_path: Path) -> None:
        """Curated variant must exclude unreviewed exception rows."""
        csv_path = _write_valid_csv(
            tmp_path,
            rows=[("9/8/2008", 30), ("9/9/2008", 25), ("9/10/2008", 30)],
        )
        config = _make_config()  # no reviewed exceptions => 25-total is unreviewed
        output_dir = tmp_path / "processed"
        df, manifest = build_curated_dataset(csv_path, config, output_dir)
        assert manifest.row_count == 2  # dropped 1 unreviewed
        assert CLASS_UNREVIEWED not in df[COL_TOTAL_CLASS].values

    def test_curated_preserves_reviewed(self, tmp_path: Path) -> None:
        """Curated variant must preserve reviewed exception rows."""
        csv_path = _write_valid_csv(
            tmp_path,
            rows=[("12/25/2008", 20), ("9/8/2008", 30), ("9/9/2008", 30)],
        )
        config = _make_config([("2008-12-25", 20, "Christmas", "reduced_output")])
        output_dir = tmp_path / "processed"
        df, manifest = build_curated_dataset(csv_path, config, output_dir)
        assert manifest.row_count == 3  # all kept (reviewed is preserved)
        reviewed = df[df[COL_TOTAL_CLASS] == CLASS_REVIEWED]
        assert len(reviewed) == 1

    def test_curated_sorted_by_date(self, tmp_path: Path) -> None:
        """Curated variant must be sorted by date."""
        csv_path = _write_valid_csv(
            tmp_path,
            rows=[("9/10/2008", 30), ("9/8/2008", 30), ("9/9/2008", 30)],
        )
        config = _make_config()
        output_dir = tmp_path / "processed"
        df, _ = build_curated_dataset(csv_path, config, output_dir)
        dates = df[DATE_COLUMN].tolist()
        assert dates == sorted(dates)

    def test_curated_output_file_exists(self, tmp_path: Path) -> None:
        """Curated variant must write a Parquet file."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        build_curated_dataset(csv_path, config, output_dir)
        assert (output_dir / "curated_v1.parquet").exists()

    def test_curated_manifest_documents_transforms(self, tmp_path: Path) -> None:
        """Curated manifest must declare all transform steps."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        _, manifest = build_curated_dataset(csv_path, config, output_dir)
        assert manifest.variant_name == VARIANT_CURATED
        step_str = " ".join(manifest.transform_steps)
        assert "exclude_unreviewed_exceptions" in step_str
        assert "sort_by_date" in step_str

    def test_curated_has_annotation_columns(self, tmp_path: Path) -> None:
        """Curated variant must include annotation columns."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        output_dir = tmp_path / "processed"
        df, _ = build_curated_dataset(csv_path, config, output_dir)
        for col in ANNOTATION_COLUMNS:
            assert col in df.columns


# ---------------------------------------------------------------------------
# Variant selection tests
# ---------------------------------------------------------------------------
class TestVariantDefaults:
    """Tests for variant constants and defaults."""

    def test_default_variant_is_raw(self) -> None:
        """Default dataset variant should be 'raw'."""
        assert DEFAULT_VARIANT == VARIANT_RAW

    def test_raw_and_curated_have_distinct_transforms(self, tmp_path: Path) -> None:
        """Raw and curated transforms should differ."""
        csv_path = _write_valid_csv(tmp_path)
        config = _make_config()
        raw_dir = tmp_path / "raw_out"
        cur_dir = tmp_path / "cur_out"
        _, raw_m = build_raw_dataset(csv_path, config, raw_dir)
        _, cur_m = build_curated_dataset(csv_path, config, cur_dir)
        assert raw_m.transform_steps != cur_m.transform_steps
