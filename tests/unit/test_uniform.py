"""Unit tests for the uniform baseline scoring model."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.domain.constants import MAX_PART_ID, PART_COLUMNS, TOP_K, VALID_PART_IDS
from c5_forecasting.models.uniform import MODEL_NAME, UNIFORM_SCORE, compute_uniform_scores


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


class TestComputeUniformScores:
    """Tests for compute_uniform_scores."""

    def test_returns_39_scores(self) -> None:
        """Should return exactly 39 scores (one per valid part ID)."""
        df = _make_df([_make_row("2020-01-01")])
        scores = compute_uniform_scores(df)
        assert len(scores) == 39
        ids = {s.part_id for s in scores}
        assert ids == set(VALID_PART_IDS)

    def test_all_scores_equal(self) -> None:
        """All 39 scores should be identical."""
        df = _make_df([_make_row("2020-01-01")])
        scores = compute_uniform_scores(df)
        unique_scores = {s.score for s in scores}
        assert len(unique_scores) == 1

    def test_score_value_is_20_over_39(self) -> None:
        """Uniform score should equal TOP_K / MAX_PART_ID = 20/39."""
        df = _make_df([_make_row("2020-01-01")])
        scores = compute_uniform_scores(df)
        expected = TOP_K / MAX_PART_ID
        assert scores[0].score == pytest.approx(expected)
        assert pytest.approx(expected) == UNIFORM_SCORE

    def test_ignores_data_content(self) -> None:
        """Different data content should produce identical scores."""
        df_zeros = _make_df([_make_row("2020-01-01")])
        counts = {col: 5 for col in PART_COLUMNS}
        df_fives = _make_df([_make_row("2020-01-01", counts)])

        scores_zeros = compute_uniform_scores(df_zeros)
        scores_fives = compute_uniform_scores(df_fives)

        for s0, s5 in zip(scores_zeros, scores_fives, strict=False):
            assert s0.part_id == s5.part_id
            assert s0.score == s5.score

    def test_model_name_constant(self) -> None:
        """MODEL_NAME should be 'uniform_baseline'."""
        assert MODEL_NAME == "uniform_baseline"

    def test_empty_df_raises(self) -> None:
        """An empty DataFrame should raise ValueError."""
        df = _make_df([])
        with pytest.raises(ValueError, match="empty"):
            compute_uniform_scores(df)

    def test_deterministic(self) -> None:
        """Two calls with the same input should produce identical results."""
        df = _make_df([_make_row("2020-01-01")])
        scores1 = compute_uniform_scores(df)
        scores2 = compute_uniform_scores(df)
        for s1, s2 in zip(scores1, scores2, strict=False):
            assert s1.part_id == s2.part_id
            assert s1.score == s2.score
