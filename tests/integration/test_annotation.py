"""Integration tests for event annotation against the real CSV and config."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from c5_forecasting.data.annotation import (
    CLASS_REVIEWED,
    CLASS_STANDARD,
    COL_EVENT_LABEL,
    COL_IS_EXCEPTION,
    COL_QUALITY_FLAGS,
    COL_ROW_TOTAL,
    COL_TOTAL_CLASS,
    annotate_dataset,
    load_annotation_config,
)
from c5_forecasting.data.loader import load_raw_csv
from c5_forecasting.domain.constants import PART_COLUMNS, STANDARD_DAILY_TOTAL

_RAW_CSV = Path("data/raw/c5_aggregated_matrix.csv")
_CONFIG_YAML = Path("configs/datasets/event_annotations.yaml")

# The 9 PO-reviewed exception dates and their expected totals.
REVIEWED_EXCEPTIONS = {
    "2008-12-25": 20,
    "2009-12-25": 20,
    "2010-12-25": 20,
    "2011-07-03": 25,
    "2011-08-28": 25,
    "2011-12-25": 25,
    "2012-05-15": 35,
    "2012-11-29": 35,
    "2012-12-25": 25,
}


@pytest.fixture()
def real_csv() -> Path:
    """Resolve the path to the real raw CSV, skipping if not found."""
    if not _RAW_CSV.exists():
        pytest.skip("Real CSV not available")
    return _RAW_CSV


@pytest.fixture()
def annotation_config() -> Path:
    """Resolve the path to the annotation config, skipping if not found."""
    if not _CONFIG_YAML.exists():
        pytest.skip("Annotation config not available")
    return _CONFIG_YAML


class TestRealDatasetAnnotation:
    """Integration tests that annotate the actual source CSV."""

    def test_config_loads(self, annotation_config: Path) -> None:
        """The real annotation config must load without error."""
        config = load_annotation_config(annotation_config)
        assert config.standard_daily_total == STANDARD_DAILY_TOTAL
        assert len(config.reviewed_exceptions) == 9

    def test_annotation_completes(self, real_csv: Path, annotation_config: Path) -> None:
        """Annotation must run without error on the real dataset."""
        df = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        df_out, result = annotate_dataset(df, config)
        assert result.row_count == 6412
        assert (
            result.standard_count
            + result.reviewed_exception_count
            + (result.unreviewed_exception_count)
            == result.row_count
        )

    def test_all_reviewed_dates_classified(self, real_csv: Path, annotation_config: Path) -> None:
        """Every reviewed exception date must be classified as reviewed_exception."""
        df = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        df_out, _ = annotate_dataset(df, config)

        date_strings = df_out["date"].dt.strftime("%Y-%m-%d")
        for date_str, expected_total in REVIEWED_EXCEPTIONS.items():
            mask = date_strings == date_str
            assert mask.any(), f"Date {date_str} not found in dataset"
            row = df_out[mask].iloc[0]
            assert row[COL_TOTAL_CLASS] == CLASS_REVIEWED, (
                f"Date {date_str} should be reviewed_exception"
            )
            assert row[COL_IS_EXCEPTION]
            assert row[COL_ROW_TOTAL] == expected_total
            assert row[COL_EVENT_LABEL] != ""

    def test_reviewed_exception_count_is_nine(
        self, real_csv: Path, annotation_config: Path
    ) -> None:
        """There should be exactly 9 reviewed exception rows."""
        df = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        _, result = annotate_dataset(df, config)
        assert result.reviewed_exception_count == 9

    def test_standard_rows_are_majority(self, real_csv: Path, annotation_config: Path) -> None:
        """The vast majority of rows should be standard_output."""
        df = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        _, result = annotate_dataset(df, config)
        assert result.standard_count > 6000

    def test_raw_counts_preserved(self, real_csv: Path, annotation_config: Path) -> None:
        """Annotation must not alter any raw part-column values."""
        df_original = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        df_annotated, _ = annotate_dataset(df_original.copy(), config)

        for col in PART_COLUMNS:
            assert (df_annotated[col] == df_original[col]).all(), (
                f"Column {col} was modified by annotation"
            )

    def test_no_unreviewed_exceptions_in_current_data(
        self, real_csv: Path, annotation_config: Path
    ) -> None:
        """With all 9 exceptions reviewed, there should be 0 unreviewed rows."""
        df = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        _, result = annotate_dataset(df, config)
        assert result.unreviewed_exception_count == 0, (
            f"Found {result.unreviewed_exception_count} unreviewed exceptions — "
            f"all non-30 dates should be in the reviewed list"
        )

    def test_quality_flags_on_reviewed_rows(self, real_csv: Path, annotation_config: Path) -> None:
        """Reviewed exception rows should have non-empty quality flags."""
        df = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        df_out, _ = annotate_dataset(df, config)

        reviewed_mask = df_out[COL_TOTAL_CLASS] == CLASS_REVIEWED
        for _, row in df_out[reviewed_mask].iterrows():
            assert row[COL_QUALITY_FLAGS] != "", "Reviewed row should have quality flags"
            assert row[COL_QUALITY_FLAGS].startswith("reviewed:")

    def test_standard_rows_have_empty_flags(self, real_csv: Path, annotation_config: Path) -> None:
        """Standard rows should have empty quality flags and labels."""
        df = load_raw_csv(real_csv)
        config = load_annotation_config(annotation_config)
        df_out, _ = annotate_dataset(df, config)

        standard_mask = df_out[COL_TOTAL_CLASS] == CLASS_STANDARD
        assert (df_out.loc[standard_mask, COL_QUALITY_FLAGS] == "").all()
        assert (df_out.loc[standard_mask, COL_EVENT_LABEL] == "").all()
        assert (~df_out.loc[standard_mask, COL_IS_EXCEPTION]).all()


class TestAnnotateDatasetCli:
    """Integration tests for the annotate-dataset CLI command."""

    def test_cli_annotate_exits_zero(self) -> None:
        """Running ``python -m c5_forecasting annotate-dataset`` must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "annotate-dataset"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Annotation PASSED" in result.stdout

    def test_cli_annotate_shows_counts(self) -> None:
        """The CLI output must include row counts."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "annotate-dataset"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "Standard days:" in result.stdout
        assert "Reviewed exceptions:" in result.stdout
