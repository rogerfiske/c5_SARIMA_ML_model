"""Unit tests for schema validation logic."""

from __future__ import annotations

from pathlib import Path

from c5_forecasting.data.validation import validate_raw_dataset
from c5_forecasting.domain.constants import PART_COLUMNS

# Build the header programmatically to avoid a 200+ char literal.
_HEADER = "date," + ",".join(PART_COLUMNS)
_ROW_TEMPLATE = "{date}," + ",".join(["1"] * 39)


def _make_csv(tmp_path: Path, rows: list[str], filename: str = "test.csv") -> Path:
    """Helper: write a CSV with header + rows to a temp file."""
    content = _HEADER + "\n" + "\n".join(rows)
    p = tmp_path / filename
    p.write_text(content)
    return p


def _valid_rows(dates: list[str]) -> list[str]:
    """Generate valid CSV data rows for the given dates."""
    return [_ROW_TEMPLATE.format(date=d) for d in dates]


class TestColumnValidation:
    """Tests for column presence and ordering checks."""

    def test_valid_columns_pass(self, tmp_path: Path) -> None:
        """A CSV with all expected columns should pass validation."""
        csv = _make_csv(tmp_path, _valid_rows(["1/1/2020", "1/2/2020", "1/3/2020"]))
        result = validate_raw_dataset(csv)
        assert result.is_valid

    def test_missing_column_detected(self, tmp_path: Path) -> None:
        """A CSV missing a part column should fail with an error."""
        # Write CSV with P_39 missing
        header = ",".join(_HEADER.split(",")[:-1])  # Remove P_39
        row = "1/1/2020," + ",".join(["1"] * 38)
        p = tmp_path / "test.csv"
        p.write_text(header + "\n" + row)
        result = validate_raw_dataset(p)
        assert not result.is_valid
        assert any("Missing columns" in e for e in result.errors)

    def test_extra_column_warned(self, tmp_path: Path) -> None:
        """A CSV with an extra column should still pass but produce a warning."""
        header = _HEADER + ",EXTRA"
        row = "1/1/2020," + ",".join(["1"] * 39) + ",99"
        p = tmp_path / "test.csv"
        p.write_text(header + "\n" + row)
        result = validate_raw_dataset(p)
        assert result.is_valid
        assert any("Extra columns" in w for w in result.warnings)


class TestNullDetection:
    """Tests for null/missing value detection."""

    def test_null_value_detected(self, tmp_path: Path) -> None:
        """A CSV with a null value in a part column should fail."""
        row = "1/1/2020," + ",".join(["1"] * 38) + ","  # P_39 is empty
        p = tmp_path / "test.csv"
        p.write_text(_HEADER + "\n" + row)
        result = validate_raw_dataset(p)
        assert not result.is_valid
        assert any("null" in e.lower() for e in result.errors)


class TestIntegerValidation:
    """Tests for integer type and non-negative value checks."""

    def test_non_integer_detected(self, tmp_path: Path) -> None:
        """A CSV with a non-integer value should fail."""
        row = "1/1/2020,abc," + ",".join(["1"] * 38)
        p = tmp_path / "test.csv"
        p.write_text(_HEADER + "\n" + row)
        result = validate_raw_dataset(p)
        assert not result.is_valid
        assert any("non-integer" in e.lower() for e in result.errors)

    def test_negative_value_detected(self, tmp_path: Path) -> None:
        """A CSV with a negative value should fail."""
        row = "1/1/2020,-1," + ",".join(["1"] * 38)
        p = tmp_path / "test.csv"
        p.write_text(_HEADER + "\n" + row)
        result = validate_raw_dataset(p)
        assert not result.is_valid
        assert any("negative" in e.lower() for e in result.errors)

    def test_zero_is_valid_count(self, tmp_path: Path) -> None:
        """Zero is a valid historical count value and should not cause an error."""
        row = "1/1/2020," + ",".join(["0"] * 39)
        p = tmp_path / "test.csv"
        p.write_text(_HEADER + "\n" + row)
        result = validate_raw_dataset(p)
        assert result.is_valid


class TestDuplicateDateDetection:
    """Tests for duplicate date detection."""

    def test_duplicate_date_detected(self, tmp_path: Path) -> None:
        """A CSV with duplicate dates should fail."""
        csv = _make_csv(
            tmp_path,
            _valid_rows(["1/1/2020", "1/1/2020", "1/2/2020"]),
        )
        result = validate_raw_dataset(csv)
        assert not result.is_valid
        assert result.duplicate_date_count == 1
        assert any("duplicate" in e.lower() for e in result.errors)


class TestDateContinuity:
    """Tests for date continuity checks."""

    def test_continuous_dates_no_gaps(self, tmp_path: Path) -> None:
        """A CSV with consecutive dates should report zero gaps."""
        csv = _make_csv(
            tmp_path,
            _valid_rows(["1/1/2020", "1/2/2020", "1/3/2020"]),
        )
        result = validate_raw_dataset(csv)
        assert result.missing_date_count == 0

    def test_gap_detected(self, tmp_path: Path) -> None:
        """A CSV with a gap in dates should report missing dates."""
        csv = _make_csv(
            tmp_path,
            _valid_rows(["1/1/2020", "1/3/2020"]),  # 1/2/2020 missing
        )
        result = validate_raw_dataset(csv)
        assert result.missing_date_count == 1
        assert "2020-01-02" in result.missing_dates


class TestRowTotals:
    """Tests for informational row-total computation."""

    def test_row_totals_computed(self, tmp_path: Path) -> None:
        """Row totals should be computed for valid data."""
        csv = _make_csv(tmp_path, _valid_rows(["1/1/2020"]))
        result = validate_raw_dataset(csv)
        # Each row has 39 columns all set to 1, so total = 39
        assert result.row_total_min == 39
        assert result.row_total_max == 39

    def test_non_30_total_not_error(self, tmp_path: Path) -> None:
        """Rows with totals != 30 must NOT produce errors (domain rule)."""
        # Row with total = 25 (reduced output day)
        row = "1/1/2020," + ",".join(["1"] * 25 + ["0"] * 14)
        p = tmp_path / "test.csv"
        p.write_text(_HEADER + "\n" + row)
        result = validate_raw_dataset(p)
        assert result.is_valid
        assert result.row_total_min == 25


class TestValidationResult:
    """Tests for ValidationResult serialization."""

    def test_to_dict_has_required_fields(self, tmp_path: Path) -> None:
        """The serialized result should contain all required fields."""
        csv = _make_csv(tmp_path, _valid_rows(["1/1/2020"]))
        result = validate_raw_dataset(csv)
        d = result.to_dict()
        required_keys = {
            "source_path",
            "source_sha256",
            "row_count",
            "column_count",
            "date_min",
            "date_max",
            "errors",
            "warnings",
            "missing_dates",
            "missing_date_count",
            "duplicate_dates",
            "duplicate_date_count",
            "is_valid",
        }
        assert required_keys.issubset(d.keys())

    def test_sha256_is_64_char_hex(self, tmp_path: Path) -> None:
        """The source hash should be a 64-character hex string."""
        csv = _make_csv(tmp_path, _valid_rows(["1/1/2020"]))
        result = validate_raw_dataset(csv)
        assert len(result.source_sha256) == 64
        assert all(c in "0123456789abcdef" for c in result.source_sha256)
