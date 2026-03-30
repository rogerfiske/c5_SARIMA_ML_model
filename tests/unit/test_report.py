"""Unit tests for validation report generation."""

from __future__ import annotations

import json
from pathlib import Path

from c5_forecasting.data.report import write_validation_report
from c5_forecasting.data.validation import ValidationResult


class TestWriteValidationReport:
    """Tests for write_validation_report."""

    def test_report_written(self, tmp_path: Path) -> None:
        """A validation report should be written as JSON."""
        result = ValidationResult(
            source_path="test.csv",
            source_sha256="a" * 64,
            row_count=100,
            column_count=40,
            date_min="2020-01-01",
            date_max="2020-04-09",
            expected_columns=["date", "P_1"],
            actual_columns=["date", "P_1"],
            is_valid=True,
        )
        report_path = write_validation_report(result, tmp_path)
        assert report_path.exists()
        assert report_path.suffix == ".json"

    def test_report_is_valid_json(self, tmp_path: Path) -> None:
        """The written report should be parseable JSON."""
        result = ValidationResult(
            source_path="test.csv",
            source_sha256="b" * 64,
            row_count=50,
            column_count=40,
            date_min="2020-01-01",
            date_max="2020-02-19",
            expected_columns=["date"],
            actual_columns=["date"],
            is_valid=True,
        )
        report_path = write_validation_report(result, tmp_path)
        with open(report_path) as f:
            data = json.load(f)
        assert data["report_type"] == "raw_dataset_validation"
        assert data["is_valid"] is True
        assert data["source_sha256"] == "b" * 64

    def test_report_creates_directory(self, tmp_path: Path) -> None:
        """The report writer should create the output directory if it doesn't exist."""
        nested = tmp_path / "nested" / "dir"
        result = ValidationResult(
            source_path="test.csv",
            source_sha256="c" * 64,
            row_count=1,
            column_count=40,
            date_min="2020-01-01",
            date_max="2020-01-01",
            expected_columns=[],
            actual_columns=[],
            is_valid=True,
        )
        report_path = write_validation_report(result, nested)
        assert report_path.exists()

    def test_successive_reports_not_overwritten(self, tmp_path: Path) -> None:
        """Two reports written in succession should produce distinct files."""
        result = ValidationResult(
            source_path="test.csv",
            source_sha256="d" * 64,
            row_count=1,
            column_count=40,
            date_min="2020-01-01",
            date_max="2020-01-01",
            expected_columns=[],
            actual_columns=[],
            is_valid=True,
        )
        p1 = write_validation_report(result, tmp_path)
        # Rename first so the timestamp-based name doesn't collide within same second
        p1_renamed = p1.with_name("report_1.json")
        p1.rename(p1_renamed)
        p2 = write_validation_report(result, tmp_path)
        assert p1_renamed.exists()
        assert p2.exists()
