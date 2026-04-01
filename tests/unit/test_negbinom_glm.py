"""Unit tests for the Negative Binomial GLM challenger model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS, TOP_K, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore
from c5_forecasting.models.negbinom_glm import (
    _build_lag_features,
    _fit_and_forecast_glm,
    _normalize_scores,
    negbinom_glm_scoring,
)


def _make_df(n_rows: int = 200) -> pd.DataFrame:
    """Build a synthetic DataFrame with enough rows for GLM fitting."""
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


class TestNegbinomGlmScoring:
    """Tests for negbinom_glm_scoring()."""

    def test_returns_list_of_part_scores(self) -> None:
        """negbinom_glm_scoring should return a list of PartScore instances."""
        df = _make_df()
        result = negbinom_glm_scoring(df)
        assert isinstance(result, list)
        assert all(isinstance(s, PartScore) for s in result)

    def test_returns_39_scores(self) -> None:
        """negbinom_glm_scoring should return exactly 39 scores."""
        df = _make_df()
        result = negbinom_glm_scoring(df)
        assert len(result) == 39

    def test_all_part_ids_in_valid_range(self) -> None:
        """All part IDs should be in 1..39, never 0."""
        df = _make_df()
        result = negbinom_glm_scoring(df)
        ids = {s.part_id for s in result}
        assert ids == VALID_PART_IDS
        assert 0 not in ids

    def test_no_duplicate_part_ids(self) -> None:
        """All part IDs should be unique."""
        df = _make_df()
        result = negbinom_glm_scoring(df)
        ids = [s.part_id for s in result]
        assert len(ids) == len(set(ids))

    def test_scores_non_negative(self) -> None:
        """All scores should be >= 0."""
        df = _make_df()
        result = negbinom_glm_scoring(df)
        for s in result:
            assert s.score >= 0.0, f"Part {s.part_id} has negative score {s.score}"

    def test_top_20_have_highest_scores(self) -> None:
        """First 20 entries should have scores >= the remaining 19."""
        df = _make_df()
        result = negbinom_glm_scoring(df)
        top_20_min = result[TOP_K - 1].score
        for s in result[TOP_K:]:
            assert s.score <= top_20_min

    def test_sorted_descending_by_score(self) -> None:
        """Scores should be sorted descending."""
        df = _make_df()
        result = negbinom_glm_scoring(df)
        scores = [s.score for s in result]
        assert scores == sorted(scores, reverse=True)


class TestNormalizeScores:
    """Tests for _normalize_scores()."""

    def test_basic_normalization(self) -> None:
        """Min-max normalization should map to [0, 1]."""
        raw = {"P_1": 10.0, "P_2": 20.0, "P_3": 30.0}
        normalized = _normalize_scores(raw)
        assert normalized["P_1"] == pytest.approx(0.0)
        assert normalized["P_2"] == pytest.approx(0.5)
        assert normalized["P_3"] == pytest.approx(1.0)

    def test_identical_values_uniform(self) -> None:
        """When all forecasts are identical, return uniform score = TOP_K/n."""
        raw = {f"P_{i}": 5.0 for i in range(1, 40)}
        normalized = _normalize_scores(raw)
        expected_uniform = TOP_K / 39
        for v in normalized.values():
            assert v == pytest.approx(expected_uniform)


class TestBuildLagFeatures:
    """Tests for _build_lag_features()."""

    def test_shape_with_lag7(self) -> None:
        """With lag7, output should have shape (n, 2)."""
        series = np.arange(20, dtype=float)
        features = _build_lag_features(series, include_lag7=True)
        assert features.shape == (20, 2)

    def test_shape_without_lag7(self) -> None:
        """Without lag7, output should have shape (n, 1)."""
        series = np.arange(20, dtype=float)
        features = _build_lag_features(series, include_lag7=False)
        assert features.shape == (20, 1)


class TestFitAndForecastGlm:
    """Tests for _fit_and_forecast_glm()."""

    def test_constant_series(self) -> None:
        """Constant input should return a non-negative forecast."""
        series = np.full(100, 5.0)
        forecast = _fit_and_forecast_glm(series)
        assert forecast >= 0.0
        assert np.isfinite(forecast)

    def test_short_series(self) -> None:
        """Very short series should still produce a result via fallback."""
        series = np.array([3.0, 5.0])
        forecast = _fit_and_forecast_glm(series)
        assert forecast >= 0.0
        assert np.isfinite(forecast)

    def test_empty_series(self) -> None:
        """Empty series should return 0.0."""
        series = np.array([], dtype=float)
        forecast = _fit_and_forecast_glm(series)
        assert forecast == 0.0
