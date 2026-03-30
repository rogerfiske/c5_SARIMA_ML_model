"""Unit tests for domain constants and hard business rules."""

from c5_forecasting.domain.constants import (
    DATE_COLUMN,
    EXPECTED_COLUMNS,
    MAX_PART_ID,
    MIN_PART_ID,
    PART_COLUMNS,
    TOP_K,
    VALID_PART_IDS,
)


class TestValidPartIds:
    """Tests for the VALID_PART_IDS constant."""

    def test_zero_excluded(self) -> None:
        """0 must NEVER be a valid part ID (core domain rule)."""
        assert 0 not in VALID_PART_IDS

    def test_count(self) -> None:
        """There are exactly 39 valid part IDs."""
        assert len(VALID_PART_IDS) == 39

    def test_range(self) -> None:
        """Valid IDs span 1 through 39 inclusive."""
        assert min(VALID_PART_IDS) == MIN_PART_ID == 1
        assert max(VALID_PART_IDS) == MAX_PART_ID == 39

    def test_contiguous(self) -> None:
        """Valid IDs form a contiguous integer range with no gaps."""
        assert frozenset(range(1, 40)) == VALID_PART_IDS

    def test_negative_excluded(self) -> None:
        """Negative numbers must not be valid part IDs."""
        assert -1 not in VALID_PART_IDS


class TestPartColumns:
    """Tests for the PART_COLUMNS constant."""

    def test_count(self) -> None:
        """There are exactly 39 part columns."""
        assert len(PART_COLUMNS) == 39

    def test_names_match_ids(self) -> None:
        """Each column name P_x maps to a valid part ID x."""
        for col in PART_COLUMNS:
            part_id = int(col.split("_")[1])
            assert part_id in VALID_PART_IDS

    def test_first_and_last(self) -> None:
        """First column is P_1 and last is P_39."""
        assert PART_COLUMNS[0] == "P_1"
        assert PART_COLUMNS[-1] == "P_39"


class TestExpectedColumns:
    """Tests for the full expected column list."""

    def test_first_column_is_date(self) -> None:
        """The first expected column must be the date column."""
        assert EXPECTED_COLUMNS[0] == DATE_COLUMN == "date"

    def test_total_column_count(self) -> None:
        """Total expected columns = 1 date + 39 parts = 40."""
        assert len(EXPECTED_COLUMNS) == 40


class TestTopK:
    """Tests for the TOP_K forecast output size."""

    def test_value(self) -> None:
        """Top-K is 20 per PRD requirement."""
        assert TOP_K == 20

    def test_within_universe(self) -> None:
        """TOP_K must not exceed the number of valid part IDs."""
        assert len(VALID_PART_IDS) >= TOP_K
