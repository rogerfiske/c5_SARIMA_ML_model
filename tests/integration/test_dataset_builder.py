"""Integration tests for the working dataset builder against the real CSV."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.data.annotation import (
    ANNOTATION_COLUMNS,
    CLASS_REVIEWED,
    CLASS_UNREVIEWED,
    COL_TOTAL_CLASS,
    load_annotation_config,
)
from c5_forecasting.data.dataset_builder import (
    VARIANT_CURATED,
    VARIANT_RAW,
    build_curated_dataset,
    build_raw_dataset,
    write_manifest,
)
from c5_forecasting.domain.constants import DATE_COLUMN

_RAW_CSV = Path("data/raw/c5_aggregated_matrix.csv")
_CONFIG_YAML = Path("configs/datasets/event_annotations.yaml")


@pytest.fixture()
def real_csv() -> Path:
    if not _RAW_CSV.exists():
        pytest.skip("Real CSV not available")
    return _RAW_CSV


@pytest.fixture()
def annotation_config_path() -> Path:
    if not _CONFIG_YAML.exists():
        pytest.skip("Annotation config not available")
    return _CONFIG_YAML


class TestRealRawDataset:
    """Integration tests for the raw working dataset."""

    def test_raw_row_count_matches_source(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Raw variant must have the same row count as the source CSV."""
        config = load_annotation_config(annotation_config_path)
        df, manifest = build_raw_dataset(real_csv, config, tmp_path)
        assert manifest.row_count == 6412
        assert len(df) == 6412

    def test_raw_has_annotation_columns(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Raw variant must include all annotation columns."""
        config = load_annotation_config(annotation_config_path)
        df, _ = build_raw_dataset(real_csv, config, tmp_path)
        for col in ANNOTATION_COLUMNS:
            assert col in df.columns

    def test_raw_parquet_readable(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """The raw Parquet file must be readable by pyarrow."""
        config = load_annotation_config(annotation_config_path)
        build_raw_dataset(real_csv, config, tmp_path)
        df = pd.read_parquet(tmp_path / "raw_v1.parquet")
        assert len(df) == 6412

    def test_raw_manifest_has_source_sha256(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Raw manifest must include the source SHA-256."""
        config = load_annotation_config(annotation_config_path)
        _, manifest = build_raw_dataset(real_csv, config, tmp_path)
        assert len(manifest.source_sha256) == 64
        assert manifest.source_sha256 == (
            "0333d82f881674d5a832b19050c898b918300960f39ccfe0d44707d5feb7c78a"
        )

    def test_raw_manifest_json_valid(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Raw manifest written as JSON must be valid and contain all fields."""
        config = load_annotation_config(annotation_config_path)
        _, manifest = build_raw_dataset(real_csv, config, tmp_path)
        manifest_dir = tmp_path / "manifests"
        path = write_manifest(manifest, manifest_dir)
        data = json.loads(path.read_text())
        assert data["variant_name"] == VARIANT_RAW
        assert data["row_count"] == 6412
        assert "source_sha256" in data
        assert "output_sha256" in data
        assert "build_timestamp" in data
        assert "transform_steps" in data

    def test_raw_part_columns_are_typed(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Part columns in the raw variant should be Int64 (not string)."""
        config = load_annotation_config(annotation_config_path)
        df, _ = build_raw_dataset(real_csv, config, tmp_path)
        assert df["P_1"].dtype.name == "Int64"

    def test_raw_preserves_all_exception_rows(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Raw variant must contain all 9 reviewed exception rows."""
        config = load_annotation_config(annotation_config_path)
        df, _ = build_raw_dataset(real_csv, config, tmp_path)
        reviewed = df[df[COL_TOTAL_CLASS] == CLASS_REVIEWED]
        assert len(reviewed) == 9


class TestRealCuratedDataset:
    """Integration tests for the curated working dataset."""

    def test_curated_row_count(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Curated variant row count should match raw (0 unreviewed in current data)."""
        config = load_annotation_config(annotation_config_path)
        df, manifest = build_curated_dataset(real_csv, config, tmp_path)
        # Current data has 0 unreviewed, so curated == raw row count
        assert manifest.row_count == 6412

    def test_curated_preserves_reviewed(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Curated variant must preserve all reviewed exception rows."""
        config = load_annotation_config(annotation_config_path)
        df, _ = build_curated_dataset(real_csv, config, tmp_path)
        reviewed = df[df[COL_TOTAL_CLASS] == CLASS_REVIEWED]
        assert len(reviewed) == 9

    def test_curated_no_unreviewed(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Curated variant must have zero unreviewed exception rows."""
        config = load_annotation_config(annotation_config_path)
        df, _ = build_curated_dataset(real_csv, config, tmp_path)
        unreviewed = df[df[COL_TOTAL_CLASS] == CLASS_UNREVIEWED]
        assert len(unreviewed) == 0

    def test_curated_sorted_by_date(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Curated variant must be sorted by date."""
        config = load_annotation_config(annotation_config_path)
        df, _ = build_curated_dataset(real_csv, config, tmp_path)
        dates = df[DATE_COLUMN].tolist()
        assert dates == sorted(dates)

    def test_curated_manifest_documents_transforms(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Curated manifest must declare exclude and sort transforms."""
        config = load_annotation_config(annotation_config_path)
        _, manifest = build_curated_dataset(real_csv, config, tmp_path)
        step_str = " ".join(manifest.transform_steps)
        assert "exclude_unreviewed_exceptions" in step_str
        assert "sort_by_date" in step_str

    def test_curated_parquet_readable(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """The curated Parquet file must be readable."""
        config = load_annotation_config(annotation_config_path)
        build_curated_dataset(real_csv, config, tmp_path)
        df = pd.read_parquet(tmp_path / "curated_v1.parquet")
        assert len(df) == 6412


class TestRawVsCurated:
    """Tests that raw and curated behaviors are distinct and auditable."""

    def test_different_variant_names(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Raw and curated manifests must have different variant names."""
        config = load_annotation_config(annotation_config_path)
        _, raw_m = build_raw_dataset(real_csv, config, tmp_path / "raw")
        _, cur_m = build_curated_dataset(real_csv, config, tmp_path / "cur")
        assert raw_m.variant_name == VARIANT_RAW
        assert cur_m.variant_name == VARIANT_CURATED

    def test_different_transform_steps(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Raw and curated must have distinct transform step lists."""
        config = load_annotation_config(annotation_config_path)
        _, raw_m = build_raw_dataset(real_csv, config, tmp_path / "raw")
        _, cur_m = build_curated_dataset(real_csv, config, tmp_path / "cur")
        assert raw_m.transform_steps != cur_m.transform_steps

    def test_same_source_sha256(
        self, real_csv: Path, annotation_config_path: Path, tmp_path: Path
    ) -> None:
        """Both variants must reference the same source SHA-256."""
        config = load_annotation_config(annotation_config_path)
        _, raw_m = build_raw_dataset(real_csv, config, tmp_path / "raw")
        _, cur_m = build_curated_dataset(real_csv, config, tmp_path / "cur")
        assert raw_m.source_sha256 == cur_m.source_sha256


class TestBuildDatasetCli:
    """Integration tests for the build-dataset CLI command."""

    def test_cli_build_raw_exits_zero(self) -> None:
        """Running build-dataset (raw) must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "build-dataset", "--variant", "raw"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Build PASSED" in result.stdout

    def test_cli_build_curated_exits_zero(self) -> None:
        """Running build-dataset (curated) must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "build-dataset", "--variant", "curated"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Build PASSED" in result.stdout

    def test_cli_build_default_is_raw(self) -> None:
        """Running build-dataset without --variant should use 'raw'."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "build-dataset"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "raw" in result.stdout
