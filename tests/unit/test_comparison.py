"""Unit tests for experiment comparison engine."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from c5_forecasting.evaluation.backtest import BacktestConfig
from c5_forecasting.evaluation.champion import ChampionRecord
from c5_forecasting.evaluation.comparison import (
    CandidateVerdict,
    ComparisonConfig,
    compare_to_champion,
    write_comparison_report,
)
from c5_forecasting.evaluation.ladder import LadderEntry, LadderResult
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


def _make_ladder_entry(
    name: str,
    ndcg: float = 0.5,
    wr: float = 0.5,
    brier: float = 0.25,
) -> LadderEntry:
    """Build a LadderEntry with a synthetic MetricSummary and no backtest result."""
    from unittest.mock import MagicMock

    return LadderEntry(
        model_name=name,
        metric_summary=_make_metric_summary(ndcg=ndcg, wr=wr, brier=brier),
        backtest_result=MagicMock(),
    )


def _make_ladder_result(
    entries_data: list[tuple[str, float, float, float]],
) -> LadderResult:
    """Build a LadderResult from [(name, ndcg, wr, brier), ...]."""
    entries = [_make_ladder_entry(name, ndcg, wr, brier) for name, ndcg, wr, brier in entries_data]
    config = BacktestConfig(min_train_rows=365, step=100)
    return LadderResult(
        entries=entries,
        config=config,
        best_model=entries[0].model_name if entries else "",
    )


def _make_champion(
    ndcg: float = 0.50,
    wr: float = 0.50,
    brier: float = 0.25,
    model_name: str = "old_champion",
) -> ChampionRecord:
    """Build a ChampionRecord with specified metrics."""
    return ChampionRecord(
        model_name=model_name,
        ndcg_20_mean=ndcg,
        weighted_recall_20_mean=wr,
        brier_score_mean=brier,
        promoted_at="2026-01-01T00:00:00Z",
        promoted_from_comparison="prev-comp-id",
        backtest_config={"min_train_rows": 365, "step": 1},
        dataset_variant="raw",
        approver="PO",
    )


class TestCompareToChampion:
    """Tests for compare_to_champion()."""

    def test_no_champion_best_is_candidate(self) -> None:
        """With no champion, best_in_report gets NO_CHAMPION and becomes candidate."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.60, 0.55, 0.24),
                ("model_b", 0.50, 0.50, 0.25),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        assert result.champion_candidate == "model_a"
        best_entry = result.entries[0]
        assert best_entry.verdict == CandidateVerdict.NO_CHAMPION
        assert best_entry.is_best_in_report is True

    def test_clear_winner_eligible(self) -> None:
        """Model beating champion by > min_delta gets ELIGIBLE."""
        champion = _make_champion(ndcg=0.50)
        ladder = _make_ladder_result(
            [
                ("model_a", 0.55, 0.55, 0.24),
            ]
        )
        config = ComparisonConfig(min_ndcg_delta=0.01)
        result = compare_to_champion(ladder, champion, config)
        assert result.champion_candidate == "model_a"
        assert result.entries[0].verdict == CandidateVerdict.ELIGIBLE

    def test_close_winner_blocked_below_delta(self) -> None:
        """Model beating champion by < min_delta gets BLOCKED_BELOW_DELTA."""
        champion = _make_champion(ndcg=0.50)
        ladder = _make_ladder_result(
            [
                ("model_a", 0.505, 0.55, 0.24),
            ]
        )
        config = ComparisonConfig(min_ndcg_delta=0.01)
        result = compare_to_champion(ladder, champion, config)
        assert result.champion_candidate is None
        assert result.entries[0].verdict == CandidateVerdict.BLOCKED_BELOW_DELTA

    def test_tied_blocked(self) -> None:
        """Model with same nDCG as champion gets BLOCKED_TIED."""
        champion = _make_champion(ndcg=0.50)
        ladder = _make_ladder_result(
            [
                ("model_a", 0.50, 0.55, 0.24),
            ]
        )
        result = compare_to_champion(ladder, champion)
        assert result.champion_candidate is None
        assert result.entries[0].verdict == CandidateVerdict.BLOCKED_TIED

    def test_only_best_in_report_can_be_candidate(self) -> None:
        """Even if non-best models beat champion, only best is candidate."""
        champion = _make_champion(ndcg=0.48)
        ladder = _make_ladder_result(
            [
                ("model_a", 0.60, 0.55, 0.24),
                ("model_b", 0.55, 0.50, 0.25),
                ("model_c", 0.50, 0.50, 0.25),
            ]
        )
        result = compare_to_champion(ladder, champion)
        assert result.champion_candidate == "model_a"
        # model_b also beats champion but is not candidate
        model_b = next(e for e in result.entries if e.model_name == "model_b")
        assert model_b.verdict != CandidateVerdict.ELIGIBLE

    def test_entries_sorted_by_ndcg_desc(self) -> None:
        """ComparisonResult.entries should be sorted by nDCG descending."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.50, 0.50, 0.25),
                ("model_b", 0.70, 0.60, 0.20),
                ("model_c", 0.30, 0.40, 0.30),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        ndcg_values = [e.metric_summary.ndcg_20_mean for e in result.entries]
        assert ndcg_values == sorted(ndcg_values, reverse=True)

    def test_tiebreak_by_wr(self) -> None:
        """When nDCG is tied, sort by WR@20 descending."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.50, 0.60, 0.25),
                ("model_b", 0.50, 0.40, 0.25),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        assert result.entries[0].model_name == "model_a"

    def test_tiebreak_by_brier(self) -> None:
        """When nDCG and WR tied, sort by Brier ascending (lower is better)."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.50, 0.50, 0.30),
                ("model_b", 0.50, 0.50, 0.20),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        assert result.entries[0].model_name == "model_b"

    def test_tiebreak_by_name(self) -> None:
        """When all metrics tied, sort alphabetically."""
        ladder = _make_ladder_result(
            [
                ("beta_model", 0.50, 0.50, 0.25),
                ("alpha_model", 0.50, 0.50, 0.25),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        assert result.entries[0].model_name == "alpha_model"

    def test_deltas_computed_correctly(self) -> None:
        """delta_vs_champion should reflect candidate - champion."""
        champion = _make_champion(ndcg=0.50, wr=0.50, brier=0.25)
        ladder = _make_ladder_result(
            [
                ("model_a", 0.55, 0.60, 0.20),
            ]
        )
        result = compare_to_champion(ladder, champion)
        d = result.entries[0].delta_vs_champion
        assert d["ndcg_20_mean"] == pytest.approx(0.05)
        assert d["weighted_recall_20_mean"] == pytest.approx(0.10)
        # Brier delta: champion - candidate = 0.25 - 0.20 = 0.05 (positive = improvement)
        assert d["brier_score_mean"] == pytest.approx(0.05)

    def test_custom_config_respected(self) -> None:
        """Non-default ComparisonConfig thresholds should be used."""
        champion = _make_champion(ndcg=0.50)
        ladder = _make_ladder_result(
            [
                ("model_a", 0.55, 0.55, 0.24),
            ]
        )
        # High threshold: 0.10 — candidate only beats by 0.05
        config = ComparisonConfig(min_ndcg_delta=0.10)
        result = compare_to_champion(ladder, champion, config)
        assert result.champion_candidate is None
        assert result.entries[0].verdict == CandidateVerdict.BLOCKED_BELOW_DELTA

    def test_comparison_result_to_dict_serializable(self) -> None:
        """ComparisonResult.to_dict() must be JSON-serializable."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.55, 0.55, 0.24),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        json_str = json.dumps(result.to_dict())
        assert json_str


class TestWriteComparisonReport:
    """Tests for write_comparison_report."""

    def test_json_report_content(self, tmp_path: Path) -> None:
        """comparison_report.json should contain expected keys."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.55, 0.55, 0.24),
                ("model_b", 0.50, 0.50, 0.25),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        write_comparison_report(result, tmp_path)
        with open(tmp_path / "comparison_report.json") as f:
            data = json.load(f)
        assert "comparison_id" in data
        assert "entries" in data
        assert "champion_candidate" in data
        assert len(data["entries"]) == 2

    def test_md_report_content(self, tmp_path: Path) -> None:
        """comparison_report.md should contain model names and verdicts."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.55, 0.55, 0.24),
            ]
        )
        result = compare_to_champion(ladder, champion=None)
        write_comparison_report(result, tmp_path)
        content = (tmp_path / "comparison_report.md").read_text()
        assert "model_a" in content
        assert "no_champion" in content
        assert "Champion Candidate Decision" in content

    def test_deterministic_output(self) -> None:
        """Two calls with same inputs produce same entries and verdicts."""
        ladder = _make_ladder_result(
            [
                ("model_a", 0.60, 0.55, 0.24),
                ("model_b", 0.50, 0.50, 0.25),
            ]
        )
        r1 = compare_to_champion(ladder, champion=None)
        r2 = compare_to_champion(ladder, champion=None)

        names1 = [e.model_name for e in r1.entries]
        names2 = [e.model_name for e in r2.entries]
        assert names1 == names2

        verdicts1 = [e.verdict for e in r1.entries]
        verdicts2 = [e.verdict for e in r2.entries]
        assert verdicts1 == verdicts2
