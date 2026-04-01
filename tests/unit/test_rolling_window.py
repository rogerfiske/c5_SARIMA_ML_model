"""Unit tests for the rolling-window frequency baseline scoring model."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS, VALID_PART_IDS
from c5_forecasting.models.baseline import compute_frequency_scores
from c5_forecasting.models.rolling_window import (
    MODEL_NAME,
    compute_rolling_window_scores,
)


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


class TestComputeRollingWindowScores:
    """Tests for compute_rolling_window_scores."""

    def test_returns_39_scores(self) -> None:
        """Should return exactly 39 scores (one per valid part ID)."""
        df = _make_df([_make_row("2020-01-01")])
        scores = compute_rolling_window_scores(df)
        assert len(scores) == 39
        ids = {s.part_id for s in scores}
        assert ids == set(VALID_PART_IDS)

    def test_all_zeros_give_zero_scores(self) -> None:
        """If no part ever appears in window, all scores should be 0."""
        df = _make_df([_make_row("2020-01-01"), _make_row("2020-01-02")])
        scores = compute_rolling_window_scores(df)
        assert all(s.score == 0.0 for s in scores)

    def test_scores_between_0_and_1(self) -> None:
        """All scores must be in [0, 1]."""
        counts = {col: 1 for col in PART_COLUMNS[:10]}
        df = _make_df([_make_row("2020-01-01", counts), _make_row("2020-01-02")])
        scores = compute_rolling_window_scores(df)
        for s in scores:
            assert 0.0 <= s.score <= 1.0

    def test_window_smaller_than_dataset(self) -> None:
        """With a small window, only recent rows should affect scores.

        P_1 appears in old rows (outside window) but not in recent rows.
        P_2 appears in recent rows only. With window=2, P_1 score should
        be 0 and P_2 score should be > 0.
        """
        rows = [
            _make_row("2020-01-01", {"P_1": 5}),  # old, outside window
            _make_row("2020-01-02", {"P_1": 5}),  # old, outside window
            _make_row("2020-01-03", {"P_2": 3}),  # in window
            _make_row("2020-01-04", {"P_2": 3}),  # in window
        ]
        df = _make_df(rows)
        scores = compute_rolling_window_scores(df, window_size=2)

        p1_score = next(s for s in scores if s.part_id == 1)
        p2_score = next(s for s in scores if s.part_id == 2)
        assert p1_score.score == 0.0
        assert p2_score.score == 1.0

    def test_window_larger_than_dataset(self) -> None:
        """When window > n_rows, all rows are used (no error)."""
        df = _make_df([_make_row("2020-01-01", {"P_1": 1}), _make_row("2020-01-02")])
        scores = compute_rolling_window_scores(df, window_size=1000)
        p1_score = next(s for s in scores if s.part_id == 1)
        assert p1_score.score == 0.5

    def test_equals_frequency_when_window_covers_all(self) -> None:
        """With window >= n_rows, rolling window should equal frequency baseline."""
        counts = {col: 1 for col in PART_COLUMNS[:15]}
        df = _make_df(
            [
                _make_row("2020-01-01", counts),
                _make_row("2020-01-02"),
                _make_row("2020-01-03", counts),
            ]
        )
        rolling_scores = compute_rolling_window_scores(df, window_size=1000)
        freq_scores = compute_frequency_scores(df)

        rolling_map = {s.part_id: s.score for s in rolling_scores}
        freq_map = {s.part_id: s.score for s in freq_scores}

        for pid in VALID_PART_IDS:
            assert rolling_map[pid] == pytest.approx(freq_map[pid])

    def test_model_name_constant(self) -> None:
        """MODEL_NAME should be 'rolling_window'."""
        assert MODEL_NAME == "rolling_window"

    def test_no_part_columns_raises(self) -> None:
        """A DataFrame with no part columns should raise ValueError."""
        df = pd.DataFrame({"date": [pd.Timestamp("2020-01-01")]})
        with pytest.raises(ValueError, match="No part columns"):
            compute_rolling_window_scores(df)

    def test_empty_df_raises(self) -> None:
        """An empty DataFrame should raise ValueError."""
        df = _make_df([])
        with pytest.raises(ValueError):
            compute_rolling_window_scores(df)

    def test_deterministic(self) -> None:
        """Two calls with the same input should produce identical results."""
        counts = {col: 1 for col in PART_COLUMNS[:5]}
        df = _make_df([_make_row("2020-01-01", counts), _make_row("2020-01-02")])
        scores1 = compute_rolling_window_scores(df)
        scores2 = compute_rolling_window_scores(df)
        for s1, s2 in zip(scores1, scores2, strict=False):
            assert s1.part_id == s2.part_id
            assert s1.score == s2.score
