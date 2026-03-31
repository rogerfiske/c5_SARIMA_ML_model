"""Unit tests for the frequency baseline scoring model."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS, VALID_PART_IDS
from c5_forecasting.models.baseline import MODEL_NAME, compute_frequency_scores


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from row dicts with Int64 typed part columns."""
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def _make_row(date_str: str, counts: dict[str, int] | None = None) -> dict:
    """Build a row dict. Unspecified parts default to 0."""
    row: dict = {"date": pd.Timestamp(date_str)}
    for col in PART_COLUMNS:
        row[col] = counts.get(col, 0) if counts else 0
    return row


class TestComputeFrequencyScores:
    """Tests for compute_frequency_scores."""

    def test_all_zeros_give_zero_scores(self) -> None:
        """If no part ever appears, all scores should be 0."""
        df = _make_df([_make_row("2020-01-01"), _make_row("2020-01-02")])
        scores = compute_frequency_scores(df)
        assert len(scores) == 39
        assert all(s.score == 0.0 for s in scores)

    def test_all_ones_give_full_scores(self) -> None:
        """If every part appears every day, all scores should be 1.0."""
        counts = {col: 1 for col in PART_COLUMNS}
        df = _make_df(
            [
                _make_row("2020-01-01", counts),
                _make_row("2020-01-02", counts),
            ]
        )
        scores = compute_frequency_scores(df)
        assert all(s.score == 1.0 for s in scores)

    def test_partial_occurrence(self) -> None:
        """A part appearing in 1 of 2 rows should have score 0.5."""
        df = _make_df(
            [
                _make_row("2020-01-01", {"P_1": 3}),
                _make_row("2020-01-02", {}),
            ]
        )
        scores = compute_frequency_scores(df)
        p1_score = next(s for s in scores if s.part_id == 1)
        assert p1_score.score == 0.5

    def test_returns_39_scores(self) -> None:
        """Should return exactly 39 scores (one per valid part ID)."""
        df = _make_df([_make_row("2020-01-01")])
        scores = compute_frequency_scores(df)
        assert len(scores) == 39
        ids = {s.part_id for s in scores}
        assert ids == set(VALID_PART_IDS)

    def test_no_part_columns_raises(self) -> None:
        """A DataFrame with no part columns should raise ValueError."""
        df = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]})
        with pytest.raises(ValueError, match="No part columns"):
            compute_frequency_scores(df)

    def test_empty_df_raises(self) -> None:
        """An empty DataFrame should raise ValueError."""
        df = _make_df([])
        with pytest.raises(ValueError):
            compute_frequency_scores(df)

    def test_model_name_constant(self) -> None:
        """MODEL_NAME should be 'frequency_baseline'."""
        assert MODEL_NAME == "frequency_baseline"

    def test_scores_are_between_0_and_1(self) -> None:
        """All scores must be in [0, 1]."""
        counts = {col: 1 for col in PART_COLUMNS[:10]}
        df = _make_df(
            [
                _make_row("2020-01-01", counts),
                _make_row("2020-01-02"),
            ]
        )
        scores = compute_frequency_scores(df)
        for s in scores:
            assert 0.0 <= s.score <= 1.0
