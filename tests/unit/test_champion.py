"""Unit tests for champion model state persistence and promotion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from c5_forecasting.evaluation.champion import (
    ChampionRecord,
    load_champion,
    promote_champion,
    save_champion,
)
from c5_forecasting.evaluation.comparison import (
    CandidateVerdict,
    ComparisonConfig,
    ComparisonEntry,
    ComparisonResult,
)
from c5_forecasting.evaluation.metrics import MetricSummary


def _make_metric_summary(
    ndcg: float = 0.5,
    wr: float = 0.5,
    brier: float = 0.25,
) -> MetricSummary:
    """Build a MetricSummary with specified primary metrics."""
    return MetricSummary(
        total_folds=4,
        ndcg_20_mean=ndcg,
        ndcg_20_min=ndcg - 0.05,
        ndcg_20_max=ndcg + 0.05,
        weighted_recall_20_mean=wr,
        weighted_recall_20_min=wr - 0.05,
        weighted_recall_20_max=wr + 0.05,
        brier_score_mean=brier,
        brier_score_min=brier - 0.02,
        brier_score_max=brier + 0.02,
        precision_20_mean=0.5,
        recall_20_mean=0.5,
        jaccard_20_mean=0.3,
    )


def _make_champion_record(**kwargs: object) -> ChampionRecord:
    """Build a ChampionRecord with sensible defaults."""
    defaults: dict[str, object] = {
        "model_name": "frequency_baseline",
        "ndcg_20_mean": 0.57,
        "weighted_recall_20_mean": 0.60,
        "brier_score_mean": 0.245,
        "promoted_at": "2026-01-01T00:00:00Z",
        "promoted_from_comparison": "test-comparison-id",
        "backtest_config": {"min_train_rows": 365, "step": 1},
        "dataset_variant": "raw",
        "approver": "PO",
    }
    defaults.update(kwargs)
    return ChampionRecord(**defaults)  # type: ignore[arg-type]


def _make_comparison_result(
    *,
    champion_candidate: str | None = "model_a",
    entries: list[ComparisonEntry] | None = None,
) -> ComparisonResult:
    """Build a minimal ComparisonResult for promote tests."""
    if entries is None:
        entries = [
            ComparisonEntry(
                model_name="model_a",
                metric_summary=_make_metric_summary(ndcg=0.60),
                verdict=CandidateVerdict.ELIGIBLE,
                delta_vs_champion={"ndcg_20_mean": 0.03},
                is_best_in_report=True,
            )
        ]
    return ComparisonResult(
        comparison_id="test-comp-id",
        comparison_timestamp="2026-04-01T00:00:00Z",
        entries=entries,
        best_in_report=entries[0].model_name if entries else "",
        champion_candidate=champion_candidate,
        current_champion=None,
        config=ComparisonConfig(),
        backtest_config={"min_train_rows": 365, "step": 1},
        dataset_variant="raw",
    )


class TestChampionRecord:
    """Tests for ChampionRecord dataclass."""

    def test_to_dict_serializable(self) -> None:
        """ChampionRecord.to_dict() should be JSON-serializable."""
        record = _make_champion_record()
        json_str = json.dumps(record.to_dict())
        assert json_str

    def test_round_trip(self) -> None:
        """ChampionRecord(**record.to_dict()) should produce equivalent record."""
        record = _make_champion_record()
        restored = ChampionRecord(**record.to_dict())
        assert restored.model_name == record.model_name
        assert restored.ndcg_20_mean == pytest.approx(record.ndcg_20_mean)
        assert restored.approver == record.approver


class TestLoadChampion:
    """Tests for load_champion."""

    def test_returns_none_when_no_file(self, tmp_path: Path) -> None:
        """load_champion returns None when champion.json does not exist."""
        result = load_champion(tmp_path)
        assert result is None

    def test_loads_valid_champion(self, tmp_path: Path) -> None:
        """load_champion reads back a previously saved champion."""
        record = _make_champion_record()
        save_champion(record, tmp_path)
        loaded = load_champion(tmp_path)
        assert loaded is not None
        assert loaded.model_name == record.model_name
        assert loaded.ndcg_20_mean == pytest.approx(record.ndcg_20_mean)


class TestSaveChampion:
    """Tests for save_champion."""

    def test_writes_valid_json(self, tmp_path: Path) -> None:
        """save_champion creates a valid JSON file."""
        record = _make_champion_record()
        path = save_champion(record, tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert data["model_name"] == "frequency_baseline"

    def test_round_trip_with_load(self, tmp_path: Path) -> None:
        """save then load should return equivalent record."""
        record = _make_champion_record(model_name="test_model")
        save_champion(record, tmp_path)
        loaded = load_champion(tmp_path)
        assert loaded is not None
        assert loaded.model_name == "test_model"
        assert loaded.approver == record.approver

    def test_writes_to_correct_path(self, tmp_path: Path) -> None:
        """File should be at artifacts_dir/champion.json."""
        record = _make_champion_record()
        save_champion(record, tmp_path)
        assert (tmp_path / "champion.json").exists()


class TestPromoteChampion:
    """Tests for promote_champion."""

    def test_creates_correct_record(self, tmp_path: Path) -> None:
        """promote_champion should write champion.json with correct fields."""
        comp_result = _make_comparison_result()
        record = promote_champion(comp_result, "PO", tmp_path)
        assert record.model_name == "model_a"
        assert record.approver == "PO"
        assert (tmp_path / "champion.json").exists()

    def test_raises_if_no_candidate(self, tmp_path: Path) -> None:
        """promote_champion should raise ValueError if no candidate."""
        comp_result = _make_comparison_result(champion_candidate=None)
        with pytest.raises(ValueError, match="No champion candidate"):
            promote_champion(comp_result, "PO", tmp_path)

    def test_eligible_verdict_works(self, tmp_path: Path) -> None:
        """ELIGIBLE candidate should be promotable."""
        entries = [
            ComparisonEntry(
                model_name="good_model",
                metric_summary=_make_metric_summary(ndcg=0.65),
                verdict=CandidateVerdict.ELIGIBLE,
                delta_vs_champion={"ndcg_20_mean": 0.08},
                is_best_in_report=True,
            )
        ]
        comp_result = _make_comparison_result(champion_candidate="good_model", entries=entries)
        record = promote_champion(comp_result, "PO", tmp_path)
        assert record.model_name == "good_model"

    def test_no_champion_verdict_works(self, tmp_path: Path) -> None:
        """NO_CHAMPION (first-time) candidate should be promotable."""
        entries = [
            ComparisonEntry(
                model_name="first_model",
                metric_summary=_make_metric_summary(ndcg=0.55),
                verdict=CandidateVerdict.NO_CHAMPION,
                delta_vs_champion={"ndcg_20_mean": 0.0},
                is_best_in_report=True,
            )
        ]
        comp_result = _make_comparison_result(champion_candidate="first_model", entries=entries)
        record = promote_champion(comp_result, "PO", tmp_path)
        assert record.model_name == "first_model"
