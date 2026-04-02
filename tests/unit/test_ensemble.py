"""Unit tests for the ensemble scoring methods."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore
from c5_forecasting.models.ensemble import (
    _ENSEMBLE_WEIGHTS,
    ensemble_avg_scoring,
    ensemble_rank_avg_scoring,
    ensemble_weighted_scoring,
)


def _make_df(n_rows: int = 200) -> pd.DataFrame:
    """Build a synthetic DataFrame with date and part columns."""
    rows = []
    for i in range(n_rows):
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        row: dict = {"date": date}
        for j in range(1, 40):
            row[f"P_{j}"] = (i + j) % 5
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        df[col] = df[col].astype("Int64")
    return df


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Provide a synthetic DataFrame for ensemble tests."""
    return _make_df(200)


class TestEnsembleAvgScoring:
    """Tests for the ensemble_avg_scoring function."""

    def test_returns_list_of_part_scores(self, synthetic_df: pd.DataFrame) -> None:
        """Output should be a list of PartScore objects."""
        result = ensemble_avg_scoring(synthetic_df)
        assert isinstance(result, list)
        assert all(isinstance(s, PartScore) for s in result)

    def test_returns_39_scores(self, synthetic_df: pd.DataFrame) -> None:
        """Should return exactly 39 scores (one per part)."""
        result = ensemble_avg_scoring(synthetic_df)
        assert len(result) == 39

    def test_all_part_ids_in_valid_range(self, synthetic_df: pd.DataFrame) -> None:
        """All part IDs should be in 1..39 and 0 should not appear."""
        result = ensemble_avg_scoring(synthetic_df)
        ids = {s.part_id for s in result}
        assert ids == VALID_PART_IDS
        assert 0 not in ids

    def test_no_duplicate_part_ids(self, synthetic_df: pd.DataFrame) -> None:
        """No duplicate part IDs should appear."""
        result = ensemble_avg_scoring(synthetic_df)
        ids = [s.part_id for s in result]
        assert len(ids) == len(set(ids))

    def test_scores_non_negative(self, synthetic_df: pd.DataFrame) -> None:
        """All scores should be >= 0."""
        result = ensemble_avg_scoring(synthetic_df)
        for s in result:
            assert s.score >= 0.0, f"Part {s.part_id} has negative score {s.score}"

    def test_sorted_descending_by_score(self, synthetic_df: pd.DataFrame) -> None:
        """Scores should be sorted descending."""
        result = ensemble_avg_scoring(synthetic_df)
        scores = [s.score for s in result]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_unit_interval(self, synthetic_df: pd.DataFrame) -> None:
        """All scores should be in [0, 1]."""
        result = ensemble_avg_scoring(synthetic_df)
        for s in result:
            assert 0.0 <= s.score <= 1.0, f"Part {s.part_id} score {s.score} out of [0,1]"


class TestEnsembleRankAvgScoring:
    """Tests for the ensemble_rank_avg_scoring function."""

    def test_returns_list_of_part_scores(self, synthetic_df: pd.DataFrame) -> None:
        """Output should be a list of PartScore objects."""
        result = ensemble_rank_avg_scoring(synthetic_df)
        assert isinstance(result, list)
        assert all(isinstance(s, PartScore) for s in result)

    def test_returns_39_scores(self, synthetic_df: pd.DataFrame) -> None:
        """Should return exactly 39 scores (one per part)."""
        result = ensemble_rank_avg_scoring(synthetic_df)
        assert len(result) == 39

    def test_all_part_ids_in_valid_range(self, synthetic_df: pd.DataFrame) -> None:
        """All part IDs should be in 1..39 and 0 should not appear."""
        result = ensemble_rank_avg_scoring(synthetic_df)
        ids = {s.part_id for s in result}
        assert ids == VALID_PART_IDS
        assert 0 not in ids

    def test_no_duplicate_part_ids(self, synthetic_df: pd.DataFrame) -> None:
        """No duplicate part IDs should appear."""
        result = ensemble_rank_avg_scoring(synthetic_df)
        ids = [s.part_id for s in result]
        assert len(ids) == len(set(ids))

    def test_scores_non_negative(self, synthetic_df: pd.DataFrame) -> None:
        """All scores should be >= 0."""
        result = ensemble_rank_avg_scoring(synthetic_df)
        for s in result:
            assert s.score >= 0.0, f"Part {s.part_id} has negative score {s.score}"

    def test_sorted_descending_by_score(self, synthetic_df: pd.DataFrame) -> None:
        """Scores should be sorted descending."""
        result = ensemble_rank_avg_scoring(synthetic_df)
        scores = [s.score for s in result]
        assert scores == sorted(scores, reverse=True)

    def test_rank_1_maps_to_score_near_1(self, synthetic_df: pd.DataFrame) -> None:
        """Top-scoring part should have score close to 1.0."""
        result = ensemble_rank_avg_scoring(synthetic_df)
        top_score = result[0].score
        # Rank 1 → score = 1.0 - (1-1)/38 = 1.0 (if all models agree)
        # In practice, the top part averages to near 1.0 but not exactly
        assert top_score >= 0.7, f"Top score {top_score} should be reasonably high"


class TestEnsembleWeightedScoring:
    """Tests for the ensemble_weighted_scoring function."""

    def test_returns_39_scores(self, synthetic_df: pd.DataFrame) -> None:
        """Should return exactly 39 scores (one per part)."""
        result = ensemble_weighted_scoring(synthetic_df)
        assert len(result) == 39

    def test_all_part_ids_in_valid_range(self, synthetic_df: pd.DataFrame) -> None:
        """All part IDs should be in 1..39 and 0 should not appear."""
        result = ensemble_weighted_scoring(synthetic_df)
        ids = {s.part_id for s in result}
        assert ids == VALID_PART_IDS
        assert 0 not in ids

    def test_scores_non_negative(self, synthetic_df: pd.DataFrame) -> None:
        """All scores should be >= 0."""
        result = ensemble_weighted_scoring(synthetic_df)
        for s in result:
            assert s.score >= 0.0, f"Part {s.part_id} has negative score {s.score}"

    def test_sorted_descending_by_score(self, synthetic_df: pd.DataFrame) -> None:
        """Scores should be sorted descending."""
        result = ensemble_weighted_scoring(synthetic_df)
        scores = [s.score for s in result]
        assert scores == sorted(scores, reverse=True)


class TestEnsembleConstants:
    """Tests for ensemble module constants."""

    def test_weights_sum_to_one(self) -> None:
        """Ensemble weights should sum to 1.0."""
        total = sum(_ENSEMBLE_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_all_weight_components_registered(self) -> None:
        """All models in _ENSEMBLE_WEIGHTS should be registered."""
        from c5_forecasting.models.registry import get_model_registry

        registry = get_model_registry()
        for model_name in _ENSEMBLE_WEIGHTS:
            assert model_name in registry, f"{model_name} not in registry"
