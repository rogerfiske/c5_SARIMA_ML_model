"""Unit tests for the raw CSV loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.data.loader import (
    coerce_part_columns,
    compute_source_hash,
    get_expected_columns,
    load_raw_csv,
)
from c5_forecasting.domain.constants import DATE_COLUMN, EXPECTED_COLUMNS


def _write_csv(tmp_path: Path, content: str, filename: str = "test.csv") -> Path:
    """Helper: write CSV content to a temp file and return its path."""
    p = tmp_path / filename
    p.write_text(textwrap.dedent(content).strip())
    return p


class TestLoadRawCsv:
    """Tests for load_raw_csv."""

    def test_valid_csv_loads(self, tmp_path: Path) -> None:
        """A minimal valid CSV should load without error."""
        csv = _write_csv(
            tmp_path,
            """\
            date,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8,P_9,P_10,P_11,P_12,P_13,P_14,P_15,P_16,P_17,P_18,P_19,P_20,P_21,P_22,P_23,P_24,P_25,P_26,P_27,P_28,P_29,P_30,P_31,P_32,P_33,P_34,P_35,P_36,P_37,P_38,P_39
            9/8/2008,0,0,1,0,0,1,2,2,0,0,1,1,2,0,0,2,1,1,0,2,1,1,2,0,1,1,1,1,2,1,0,0,1,0,1,0,0,0,1
            """,
        )
        df = load_raw_csv(csv)
        assert len(df) == 1
        assert DATE_COLUMN in df.columns
        assert df[DATE_COLUMN].iloc[0] == pd.Timestamp("2008-09-08")

    def test_date_parsing_m_d_yyyy(self, tmp_path: Path) -> None:
        """Dates in M/D/YYYY format should parse correctly."""
        csv = _write_csv(
            tmp_path,
            """\
            date,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8,P_9,P_10,P_11,P_12,P_13,P_14,P_15,P_16,P_17,P_18,P_19,P_20,P_21,P_22,P_23,P_24,P_25,P_26,P_27,P_28,P_29,P_30,P_31,P_32,P_33,P_34,P_35,P_36,P_37,P_38,P_39
            1/1/2020,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            12/25/2020,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0
            """,
        )
        df = load_raw_csv(csv)
        assert df[DATE_COLUMN].iloc[0] == pd.Timestamp("2020-01-01")
        assert df[DATE_COLUMN].iloc[1] == pd.Timestamp("2020-12-25")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """A nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_raw_csv(tmp_path / "nonexistent.csv")

    def test_missing_date_column_raises(self, tmp_path: Path) -> None:
        """A CSV without a 'date' column should raise ValueError."""
        csv = _write_csv(
            tmp_path,
            """\
            timestamp,P_1
            9/8/2008,1
            """,
        )
        with pytest.raises(ValueError, match="Missing required column.*date"):
            load_raw_csv(csv)


class TestCoercePartColumns:
    """Tests for coerce_part_columns."""

    def test_valid_integers(self, tmp_path: Path) -> None:
        """String integers should coerce to Int64 cleanly."""
        csv = _write_csv(
            tmp_path,
            """\
            date,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8,P_9,P_10,P_11,P_12,P_13,P_14,P_15,P_16,P_17,P_18,P_19,P_20,P_21,P_22,P_23,P_24,P_25,P_26,P_27,P_28,P_29,P_30,P_31,P_32,P_33,P_34,P_35,P_36,P_37,P_38,P_39
            9/8/2008,0,0,1,0,0,1,2,2,0,0,1,1,2,0,0,2,1,1,0,2,1,1,2,0,1,1,1,1,2,1,0,0,1,0,1,0,0,0,1
            """,
        )
        df = load_raw_csv(csv)
        df_typed = coerce_part_columns(df)
        assert df_typed["P_1"].dtype.name == "Int64"
        assert df_typed["P_3"].iloc[0] == 1

    def test_non_integer_becomes_na(self, tmp_path: Path) -> None:
        """Non-integer strings in part columns should coerce to NA."""
        csv = _write_csv(
            tmp_path,
            """\
            date,P_1,P_2,P_3,P_4,P_5,P_6,P_7,P_8,P_9,P_10,P_11,P_12,P_13,P_14,P_15,P_16,P_17,P_18,P_19,P_20,P_21,P_22,P_23,P_24,P_25,P_26,P_27,P_28,P_29,P_30,P_31,P_32,P_33,P_34,P_35,P_36,P_37,P_38,P_39
            9/8/2008,abc,0,1,0,0,1,2,2,0,0,1,1,2,0,0,2,1,1,0,2,1,1,2,0,1,1,1,1,2,1,0,0,1,0,1,0,0,0,1
            """,
        )
        df = load_raw_csv(csv)
        df_typed = coerce_part_columns(df)
        assert pd.isna(df_typed["P_1"].iloc[0])


class TestComputeSourceHash:
    """Tests for compute_source_hash."""

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same file content should always produce the same hash."""
        p = tmp_path / "test.txt"
        p.write_text("hello world")
        h1 = compute_source_hash(p)
        h2 = compute_source_hash(p)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest length

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different file contents should produce different hashes."""
        p1 = tmp_path / "a.txt"
        p1.write_text("hello")
        p2 = tmp_path / "b.txt"
        p2.write_text("world")
        assert compute_source_hash(p1) != compute_source_hash(p2)


class TestGetExpectedColumns:
    """Tests for get_expected_columns."""

    def test_matches_constant(self) -> None:
        """Function output should match the EXPECTED_COLUMNS constant."""
        assert get_expected_columns() == list(EXPECTED_COLUMNS)
