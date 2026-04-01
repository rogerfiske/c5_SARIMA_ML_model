"""Unit tests for the GBM ranking challenger model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS, TOP_K, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore
from c5_forecasting.models.gbm_ranking import (
    _build_features,
    _fit_and_forecast_gbm,
    _normalize_scores,
    gbm_ranking_scoring,
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
    """Provide a synthetic DataFrame for GBM tests."""
    return _make_df(200)


class TestGbmRankingScoring:
    """Tests for the gbm_ranking_scoring function."""

    def test_returns_list_of_part_scores(self, synthetic_df: pd.DataFrame) -> None:
        """Output should be a list of PartScore objects."""
        result = gbm_ranking_scoring(synthetic_df)
        assert isinstance(result, list)
        assert all(isinstance(s, PartScore) for s in result)

    def test_returns_39_scores(self, synthetic_df: pd.DataFrame) -> None:
        """Should return exactly 39 scores (one per part)."""
        result = gbm_ranking_scoring(synthetic_df)
        assert len(result) == 39

    def test_all_part_ids_in_valid_range(self, synthetic_df: pd.DataFrame) -> None:
        """All part IDs should be in 1..39 and 0 should not appear."""
        result = gbm_ranking_scoring(synthetic_df)
        ids = {s.part_id for s in result}
        assert ids == VALID_PART_IDS
        assert 0 not in ids

    def test_no_duplicate_part_ids(self, synthetic_df: pd.DataFrame) -> None:
        """No duplicate part IDs should appear."""
        result = gbm_ranking_scoring(synthetic_df)
        ids = [s.part_id for s in result]
        assert len(ids) == len(set(ids))

    def test_scores_non_negative(self, synthetic_df: pd.DataFrame) -> None:
        """All scores should be >= 0."""
        result = gbm_ranking_scoring(synthetic_df)
        for s in result:
            assert s.score >= 0.0, f"Part {s.part_id} has negative score {s.score}"

    def test_top_20_have_highest_scores(self, synthetic_df: pd.DataFrame) -> None:
        """First 20 scores should be >= all remaining scores."""
        result = gbm_ranking_scoring(synthetic_df)
        top_20_min = result[TOP_K - 1].score
        for s in result[TOP_K:]:
            assert s.score <= top_20_min

    def test_sorted_descending_by_score(self, synthetic_df: pd.DataFrame) -> None:
        """Scores should be sorted descending."""
        result = gbm_ranking_scoring(synthetic_df)
        scores = [s.score for s in result]
        assert scores == sorted(scores, reverse=True)


class TestNormalizeScores:
    """Tests for _normalize_scores."""

    def test_basic_normalization(self) -> None:
        """Min-max normalization should map to [0, 1]."""
        raw = {"P_1": 10.0, "P_2": 20.0, "P_3": 30.0}
        normalized = _normalize_scores(raw)
        assert normalized["P_1"] == pytest.approx(0.0)
        assert normalized["P_2"] == pytest.approx(0.5)
        assert normalized["P_3"] == pytest.approx(1.0)

    def test_identical_values_uniform(self) -> None:
        """All-identical values should produce uniform scores."""
        raw = {f"P_{i}": 5.0 for i in range(1, 40)}
        normalized = _normalize_scores(raw)
        expected_uniform = TOP_K / 39
        for v in normalized.values():
            assert v == pytest.approx(expected_uniform)


class TestBuildFeatures:
    """Tests for _build_features."""

    def test_shape_with_all_features(self) -> None:
        """Feature matrix should have shape (n, 6)."""
        series = np.arange(50, dtype=np.float64)
        dates = pd.date_range("2020-01-01", periods=50).to_numpy()
        features = _build_features(series, dates)
        assert features.shape == (50, 6)

    def test_lag_values_correct(self) -> None:
        """Lag-1 column should contain shifted values."""
        series = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        dates = pd.date_range("2020-01-01", periods=5).to_numpy()
        features = _build_features(series, dates)
        # lag_1 is column 0: [0, 10, 20, 30, 40]
        assert features[0, 0] == pytest.approx(0.0)  # fill
        assert features[1, 0] == pytest.approx(10.0)
        assert features[2, 0] == pytest.approx(20.0)


class TestFitAndForecastGbm:
    """Tests for _fit_and_forecast_gbm."""

    def test_constant_series(self) -> None:
        """Constant series should return a non-negative, finite forecast."""
        series = np.full(100, 3.0)
        dates = pd.date_range("2020-01-01", periods=100).to_numpy()
        result = _fit_and_forecast_gbm(series, dates)
        assert np.isfinite(result)
        assert result >= 0.0

    def test_short_series(self) -> None:
        """A length-1 series should use the fallback."""
        series = np.array([5.0])
        dates = pd.date_range("2020-01-01", periods=1).to_numpy()
        result = _fit_and_forecast_gbm(series, dates)
        assert result == pytest.approx(5.0)

    def test_empty_series(self) -> None:
        """Empty series should return 0.0."""
        series = np.array([], dtype=np.float64)
        dates = np.array([], dtype="datetime64[ns]")
        result = _fit_and_forecast_gbm(series, dates)
        assert result == 0.0
