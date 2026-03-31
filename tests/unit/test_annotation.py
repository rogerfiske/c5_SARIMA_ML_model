"""Unit tests for the event annotation module."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.data.annotation import (
    ANNOTATION_COLUMNS,
    CLASS_REVIEWED,
    CLASS_STANDARD,
    CLASS_UNREVIEWED,
    COL_EVENT_LABEL,
    COL_IS_EXCEPTION,
    COL_QUALITY_FLAGS,
    COL_ROW_TOTAL,
    COL_TOTAL_CLASS,
    EventAnnotationConfig,
    ExceptionEntry,
    annotate_dataset,
    load_annotation_config,
)
from c5_forecasting.domain.constants import (
    DATE_COLUMN,
    PART_COLUMNS,
    STANDARD_DAILY_TOTAL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_row(date_str: str, total: int) -> dict:
    """Build a single-row dict with the given date and total spread across parts."""
    row: dict = {DATE_COLUMN: pd.Timestamp(date_str)}
    for i, col in enumerate(PART_COLUMNS):
        row[col] = 1 if i < total else 0
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from row dicts."""
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        df[col] = df[col].astype("Int64")
    return df


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
# Config loader tests
# ---------------------------------------------------------------------------
class TestLoadAnnotationConfig:
    """Tests for load_annotation_config."""

    def test_valid_config(self, tmp_path: Path) -> None:
        """A well-formed YAML should parse correctly."""
        cfg_file = tmp_path / "events.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            standard_daily_total: 30
            reviewed_exceptions:
              - date: "2008-12-25"
                total: 20
                label: "Christmas"
                category: "reduced_output"
            """)
        )
        config = load_annotation_config(cfg_file)
        assert config.standard_daily_total == 30
        assert len(config.reviewed_exceptions) == 1
        assert config.reviewed_exceptions[0].date == "2008-12-25"
        assert config.reviewed_exceptions[0].total == 20

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A missing config file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_annotation_config(tmp_path / "nonexistent.yaml")

    def test_empty_exceptions_list(self, tmp_path: Path) -> None:
        """A config with no exceptions should parse with empty list."""
        cfg_file = tmp_path / "events.yaml"
        cfg_file.write_text("standard_daily_total: 30\nreviewed_exceptions: []\n")
        config = load_annotation_config(cfg_file)
        assert len(config.reviewed_exceptions) == 0

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        """An exception entry missing required fields should raise ValueError."""
        cfg_file = tmp_path / "events.yaml"
        cfg_file.write_text(
            textwrap.dedent("""\
            standard_daily_total: 30
            reviewed_exceptions:
              - date: "2008-12-25"
                total: 20
            """)
        )
        with pytest.raises(ValueError, match="missing required fields"):
            load_annotation_config(cfg_file)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        """A YAML file that is not a mapping should raise ValueError."""
        cfg_file = tmp_path / "events.yaml"
        cfg_file.write_text("- just a list\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_annotation_config(cfg_file)

    def test_exception_date_set(self) -> None:
        """exception_date_set should return a set of date strings."""
        config = _make_config(
            [
                ("2008-12-25", 20, "Christmas", "reduced_output"),
                ("2009-12-25", 20, "Christmas", "reduced_output"),
            ]
        )
        assert config.exception_date_set == {"2008-12-25", "2009-12-25"}

    def test_get_exception_found(self) -> None:
        """get_exception should return the entry for a known date."""
        config = _make_config([("2008-12-25", 20, "Christmas", "reduced_output")])
        entry = config.get_exception("2008-12-25")
        assert entry is not None
        assert entry.label == "Christmas"

    def test_get_exception_not_found(self) -> None:
        """get_exception should return None for an unknown date."""
        config = _make_config([("2008-12-25", 20, "Christmas", "reduced_output")])
        assert config.get_exception("2020-01-01") is None


# ---------------------------------------------------------------------------
# Annotation tests
# ---------------------------------------------------------------------------
class TestAnnotateDataset:
    """Tests for annotate_dataset."""

    def test_standard_row_classification(self) -> None:
        """A row with total == 30 should be classified as standard_output."""
        df = _make_df([_make_row("2020-01-01", 30)])
        config = _make_config()
        df_out, result = annotate_dataset(df, config)

        assert df_out[COL_TOTAL_CLASS].iloc[0] == CLASS_STANDARD
        assert not df_out[COL_IS_EXCEPTION].iloc[0]
        assert df_out[COL_EVENT_LABEL].iloc[0] == ""
        assert result.standard_count == 1
        assert result.reviewed_exception_count == 0
        assert result.unreviewed_exception_count == 0

    def test_reviewed_exception_classification(self) -> None:
        """A row matching a reviewed exception date should be classified correctly."""
        df = _make_df([_make_row("2008-12-25", 20)])
        config = _make_config([("2008-12-25", 20, "Christmas", "reduced_output")])
        df_out, result = annotate_dataset(df, config)

        assert df_out[COL_TOTAL_CLASS].iloc[0] == CLASS_REVIEWED
        assert df_out[COL_IS_EXCEPTION].iloc[0]
        assert df_out[COL_EVENT_LABEL].iloc[0] == "Christmas"
        assert "reviewed:reduced_output" in df_out[COL_QUALITY_FLAGS].iloc[0]
        assert result.reviewed_exception_count == 1

    def test_unreviewed_exception_classification(self) -> None:
        """A non-30 row NOT in the reviewed list should be unreviewed_exception."""
        df = _make_df([_make_row("2025-06-15", 25)])
        config = _make_config()  # no exceptions defined
        df_out, result = annotate_dataset(df, config)

        assert df_out[COL_TOTAL_CLASS].iloc[0] == CLASS_UNREVIEWED
        assert df_out[COL_IS_EXCEPTION].iloc[0]
        assert df_out[COL_EVENT_LABEL].iloc[0] == ""
        assert "unreviewed_exception" in df_out[COL_QUALITY_FLAGS].iloc[0]
        assert result.unreviewed_exception_count == 1
        assert len(result.warnings) == 1

    def test_raw_counts_not_modified(self) -> None:
        """Annotation must never alter the original part-column values."""
        original_row = _make_row("2020-01-01", 30)
        df = _make_df([original_row])
        config = _make_config()

        df_out, _ = annotate_dataset(df, config)

        for col in PART_COLUMNS:
            assert df_out[col].iloc[0] == df.iloc[0][col]

    def test_annotation_columns_added(self) -> None:
        """All expected annotation columns should be present."""
        df = _make_df([_make_row("2020-01-01", 30)])
        config = _make_config()
        df_out, result = annotate_dataset(df, config)

        for col in ANNOTATION_COLUMNS:
            assert col in df_out.columns
        assert result.annotation_columns_added == list(ANNOTATION_COLUMNS)

    def test_row_total_computed_correctly(self) -> None:
        """row_total should equal the sum of part columns."""
        df = _make_df([_make_row("2020-01-01", 25)])
        config = _make_config()
        df_out, _ = annotate_dataset(df, config)

        assert df_out[COL_ROW_TOTAL].iloc[0] == 25

    def test_mixed_rows(self) -> None:
        """A mix of standard, reviewed, and unreviewed rows should all classify."""
        rows = [
            _make_row("2020-01-01", 30),  # standard
            _make_row("2008-12-25", 20),  # reviewed
            _make_row("2025-06-15", 25),  # unreviewed
        ]
        config = _make_config([("2008-12-25", 20, "Christmas", "reduced_output")])
        df_out, result = annotate_dataset(_make_df(rows), config)

        assert result.standard_count == 1
        assert result.reviewed_exception_count == 1
        assert result.unreviewed_exception_count == 1
        assert result.row_count == 3

        classes = df_out[COL_TOTAL_CLASS].tolist()
        assert classes == [CLASS_STANDARD, CLASS_REVIEWED, CLASS_UNREVIEWED]

    def test_does_not_mutate_input(self) -> None:
        """annotate_dataset should return a copy, not mutate the input."""
        df = _make_df([_make_row("2020-01-01", 30)])
        original_cols = list(df.columns)
        config = _make_config()

        annotate_dataset(df, config)

        assert list(df.columns) == original_cols
        assert COL_TOTAL_CLASS not in df.columns

    def test_result_to_dict(self) -> None:
        """AnnotationResult.to_dict() should produce a serializable dict."""
        df = _make_df([_make_row("2020-01-01", 30)])
        config = _make_config()
        _, result = annotate_dataset(df, config)

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "row_count" in d
        assert "annotation_columns_added" in d

    def test_additional_output_category(self) -> None:
        """A reviewed exception with additional_output category should annotate."""
        df = _make_df([_make_row("2012-05-15", 35)])
        config = _make_config([("2012-05-15", 35, "Additional output", "additional_output")])
        df_out, result = annotate_dataset(df, config)

        assert df_out[COL_TOTAL_CLASS].iloc[0] == CLASS_REVIEWED
        assert "reviewed:additional_output" in df_out[COL_QUALITY_FLAGS].iloc[0]
        assert result.reviewed_exception_count == 1
