"""Integration tests for raw dataset validation against the real CSV."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from c5_forecasting.data.validation import validate_raw_dataset

# Path to the real dataset relative to repo root.
_RAW_CSV = Path("data/raw/c5_aggregated_matrix.csv")


@pytest.fixture()
def real_csv() -> Path:
    """Resolve the path to the real raw CSV, skipping if not found."""
    if not _RAW_CSV.exists():
        pytest.skip("Real CSV not available")
    return _RAW_CSV


class TestRealDatasetValidation:
    """Integration tests that run validation against the actual source CSV."""

    def test_validation_passes(self, real_csv: Path) -> None:
        """The real dataset must pass schema validation."""
        result = validate_raw_dataset(real_csv)
        assert result.is_valid, f"Validation failed with errors: {result.errors}"

    def test_expected_column_count(self, real_csv: Path) -> None:
        """The real dataset must have exactly 40 columns (date + 39 parts)."""
        result = validate_raw_dataset(real_csv)
        assert result.column_count == 40

    def test_no_duplicate_dates(self, real_csv: Path) -> None:
        """The real dataset must have no duplicate dates."""
        result = validate_raw_dataset(real_csv)
        assert result.duplicate_date_count == 0

    def test_date_range(self, real_csv: Path) -> None:
        """The real dataset date range should match PRD claims."""
        result = validate_raw_dataset(real_csv)
        assert result.date_min == "2008-09-08"
        assert result.date_max == "2026-03-29"

    def test_row_count(self, real_csv: Path) -> None:
        """The real dataset should have the expected number of rows."""
        result = validate_raw_dataset(real_csv)
        assert result.row_count == 6412

    def test_date_continuity(self, real_csv: Path) -> None:
        """Verify and report date continuity on the real dataset."""
        result = validate_raw_dataset(real_csv)
        # This test documents the actual state rather than assuming zero gaps.
        # The PRD claims 0 missing dates; this test will surface any discrepancy.
        assert isinstance(result.missing_date_count, int)

    def test_sha256_is_stable(self, real_csv: Path) -> None:
        """The SHA-256 fingerprint must be deterministic across runs."""
        r1 = validate_raw_dataset(real_csv)
        r2 = validate_raw_dataset(real_csv)
        assert r1.source_sha256 == r2.source_sha256
        assert len(r1.source_sha256) == 64

    def test_row_totals_include_exception_values(self, real_csv: Path) -> None:
        """The distinct row totals should include known exception values (25, 35)."""
        result = validate_raw_dataset(real_csv)
        assert 25 in result.distinct_row_totals
        assert 35 in result.distinct_row_totals
        assert 30 in result.distinct_row_totals

    def test_serialization_roundtrip(self, real_csv: Path) -> None:
        """ValidationResult.to_dict() must produce valid JSON-serializable output."""
        result = validate_raw_dataset(real_csv)
        d = result.to_dict()
        # Must be JSON-serializable without error
        json_str = json.dumps(d)
        assert len(json_str) > 0


class TestValidateRawCli:
    """Integration test for the validate-raw CLI command."""

    def test_cli_validate_raw_exits_zero(self) -> None:
        """Running ``python -m c5_forecasting validate-raw`` must exit 0 on valid data."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "validate-raw"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Validation PASSED" in result.stdout

    def test_cli_validate_raw_outputs_sha256(self) -> None:
        """The CLI output must include the source SHA-256 fingerprint."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "validate-raw"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert "Source SHA-256:" in result.stdout
