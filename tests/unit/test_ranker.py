"""Unit tests for the ranking module."""

from __future__ import annotations

import pytest

from c5_forecasting.domain.constants import TOP_K, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore
from c5_forecasting.ranking.ranker import (
    ForecastValidationError,
    RankedForecast,
    rank_and_select,
    validate_forecast,
)


def _make_scores(n: int = 39) -> list[PartScore]:
    """Make scores for parts 1..n with distinct values."""
    return [PartScore(part_id=i, score=(n - i) / n) for i in range(1, n + 1)]


class TestRankAndSelect:
    """Tests for rank_and_select."""

    def test_selects_top_20(self) -> None:
        """Should select exactly TOP_K=20 entries."""
        scores = _make_scores()
        forecast = rank_and_select(scores)
        assert len(forecast.rankings) == TOP_K

    def test_descending_score_order(self) -> None:
        """Rankings should be in descending score order."""
        scores = _make_scores()
        forecast = rank_and_select(scores)
        result_scores = [r.score for r in forecast.rankings]
        assert result_scores == sorted(result_scores, reverse=True)

    def test_rank_numbers_are_sequential(self) -> None:
        """Rank numbers should be 1 through K."""
        scores = _make_scores()
        forecast = rank_and_select(scores)
        ranks = [r.rank for r in forecast.rankings]
        assert ranks == list(range(1, TOP_K + 1))

    def test_all_ids_are_valid(self) -> None:
        """All ranked part IDs must be in VALID_PART_IDS."""
        scores = _make_scores()
        forecast = rank_and_select(scores)
        for r in forecast.rankings:
            assert r.part_id in VALID_PART_IDS

    def test_no_zero_in_output(self) -> None:
        """Part ID 0 must never appear in output."""
        scores = _make_scores()
        forecast = rank_and_select(scores)
        ids = [r.part_id for r in forecast.rankings]
        assert 0 not in ids

    def test_no_duplicate_ids(self) -> None:
        """No duplicate part IDs in output."""
        scores = _make_scores()
        forecast = rank_and_select(scores)
        ids = [r.part_id for r in forecast.rankings]
        assert len(set(ids)) == len(ids)

    def test_deterministic_tie_breaking(self) -> None:
        """Tied scores should be broken by lower part ID first."""
        # All parts have the same score
        scores = [PartScore(part_id=i, score=0.5) for i in range(1, 40)]
        forecast = rank_and_select(scores)
        ids = [r.part_id for r in forecast.rankings]
        # Lower IDs should win when scores are equal
        assert ids == list(range(1, 21))

    def test_deterministic_across_runs(self) -> None:
        """Same input should always produce same output."""
        scores = [PartScore(part_id=i, score=0.5) for i in range(1, 40)]
        f1 = rank_and_select(scores)
        f2 = rank_and_select(scores)
        ids1 = [r.part_id for r in f1.rankings]
        ids2 = [r.part_id for r in f2.rankings]
        assert ids1 == ids2

    def test_custom_k(self) -> None:
        """Should respect a custom K value."""
        scores = _make_scores()
        forecast = rank_and_select(scores, k=5)
        assert len(forecast.rankings) == 5

    def test_model_name_in_forecast(self) -> None:
        """Model name should be captured in the forecast."""
        scores = _make_scores()
        forecast = rank_and_select(scores, model_name="test_model")
        assert forecast.model_name == "test_model"


class TestValidateForecast:
    """Tests for validate_forecast."""

    def test_valid_forecast_passes(self) -> None:
        """A correctly formed forecast should pass validation."""
        scores = _make_scores()
        forecast = rank_and_select(scores)
        # Should not raise
        validate_forecast(forecast)

    def test_wrong_count_raises(self) -> None:
        """Wrong number of entries should raise."""
        from c5_forecasting.ranking.ranker import RankedEntry

        forecast = RankedForecast(
            rankings=[RankedEntry(rank=1, part_id=1, score=0.5)],
            model_name="test",
            k=20,
        )
        with pytest.raises(ForecastValidationError, match="Expected 20"):
            validate_forecast(forecast)

    def test_zero_in_output_raises(self) -> None:
        """Part ID 0 in output should raise."""
        from c5_forecasting.ranking.ranker import RankedEntry

        entries = [RankedEntry(rank=i + 1, part_id=i, score=0.5) for i in range(0, 20)]
        forecast = RankedForecast(rankings=entries, model_name="test", k=20)
        with pytest.raises(ForecastValidationError, match="Part ID 0"):
            validate_forecast(forecast)

    def test_invalid_id_raises(self) -> None:
        """Part ID outside 1..39 should raise."""
        from c5_forecasting.ranking.ranker import RankedEntry

        entries = [RankedEntry(rank=i + 1, part_id=i + 1, score=0.5) for i in range(19)]
        entries.append(RankedEntry(rank=20, part_id=40, score=0.1))
        forecast = RankedForecast(rankings=entries, model_name="test", k=20)
        with pytest.raises(ForecastValidationError, match="Invalid part IDs"):
            validate_forecast(forecast)

    def test_duplicate_ids_raises(self) -> None:
        """Duplicate part IDs should raise."""
        from c5_forecasting.ranking.ranker import RankedEntry

        entries = [RankedEntry(rank=i + 1, part_id=i + 1, score=0.5) for i in range(19)]
        entries.append(RankedEntry(rank=20, part_id=1, score=0.1))  # duplicate of P_1
        forecast = RankedForecast(rankings=entries, model_name="test", k=20)
        with pytest.raises(ForecastValidationError, match="Duplicate"):
            validate_forecast(forecast)
