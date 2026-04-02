"""Unit tests for the baseline ladder runner and comparison artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS
from c5_forecasting.evaluation.backtest import BacktestConfig
from c5_forecasting.evaluation.ladder import run_ladder, write_ladder_artifacts


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from row dicts with Int64 typed part columns."""
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _make_row(date_str: str, counts: dict[str, int] | None = None) -> dict:
    """Build a row dict. Unspecified parts default to 0."""
    row: dict = {"date": date_str}
    for col in PART_COLUMNS:
        row[col] = counts.get(col, 0) if counts else 0
    return row


def _build_synthetic_df(n_rows: int = 400) -> pd.DataFrame:
    """Build a synthetic DataFrame with enough rows for backtesting."""
    rows = []
    for i in range(n_rows):
        date_str = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        # Vary which parts appear
        counts = {}
        for j in range(1, 40):
            if (i + j) % 3 == 0:
                counts[f"P_{j}"] = (i + j) % 5 + 1
        rows.append(_make_row(str(date_str.date()), counts))
    return _make_df(rows)


@pytest.fixture()
def synthetic_df() -> pd.DataFrame:
    """Provide a synthetic DataFrame for ladder tests."""
    return _build_synthetic_df(400)


@pytest.fixture()
def ladder_config() -> BacktestConfig:
    """Provide a fast backtest config for ladder tests."""
    return BacktestConfig(min_train_rows=50, step=100, max_windows=5)


class TestRunLadder:
    """Tests for run_ladder."""

    def test_ladder_runs_all_models(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig
    ) -> None:
        """LadderResult should have entries for all 10 registered models."""
        result = run_ladder(synthetic_df, ladder_config)
        assert len(result.entries) == 10
        model_names = {e.model_name for e in result.entries}
        expected = {
            "ensemble_avg",
            "ensemble_rank_avg",
            "ensemble_weighted",
            "frequency_baseline",
            "gbm_ranking",
            "negbinom_glm",
            "recency_weighted",
            "rolling_window",
            "sarima",
            "uniform_baseline",
        }
        assert model_names == expected

    def test_ladder_runs_subset(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig
    ) -> None:
        """Specifying model_names should run only those models."""
        subset = ["frequency_baseline", "uniform_baseline"]
        result = run_ladder(synthetic_df, ladder_config, model_names=subset)
        assert len(result.entries) == 2
        names = {e.model_name for e in result.entries}
        assert names == {"frequency_baseline", "uniform_baseline"}

    def test_entries_sorted_by_ndcg(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig
    ) -> None:
        """Entries should be sorted by nDCG@20 descending."""
        result = run_ladder(synthetic_df, ladder_config)
        ndcg_values = [e.metric_summary.ndcg_20_mean for e in result.entries]
        assert ndcg_values == sorted(ndcg_values, reverse=True)

    def test_best_model_is_top_entry(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig
    ) -> None:
        """best_model should match the name of the first entry."""
        result = run_ladder(synthetic_df, ladder_config)
        assert result.best_model == result.entries[0].model_name

    def test_unknown_model_in_list_raises(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig
    ) -> None:
        """Invalid model name in subset should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model"):
            run_ladder(synthetic_df, ladder_config, model_names=["nonexistent"])

    def test_ladder_result_to_dict(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig
    ) -> None:
        """to_dict() should be JSON-serializable."""
        result = run_ladder(synthetic_df, ladder_config)
        d = result.to_dict()
        # Verify it's JSON-serializable
        json_str = json.dumps(d)
        assert json_str


class TestWriteLadderArtifacts:
    """Tests for write_ladder_artifacts."""

    def test_json_artifact_written(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig, tmp_path: Path
    ) -> None:
        """comparison_results.json should be created."""
        result = run_ladder(synthetic_df, ladder_config)
        write_ladder_artifacts(result, tmp_path)
        assert (tmp_path / "comparison_results.json").exists()

    def test_csv_artifact_written(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig, tmp_path: Path
    ) -> None:
        """comparison_summary.csv should be created."""
        result = run_ladder(synthetic_df, ladder_config)
        write_ladder_artifacts(result, tmp_path)
        assert (tmp_path / "comparison_summary.csv").exists()

    def test_csv_has_correct_columns(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig, tmp_path: Path
    ) -> None:
        """CSV should have rank, model, and all metric columns."""
        result = run_ladder(synthetic_df, ladder_config)
        write_ladder_artifacts(result, tmp_path)
        with open(tmp_path / "comparison_summary.csv") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        assert headers is not None
        expected = {
            "rank",
            "model",
            "ndcg_20_mean",
            "weighted_recall_20_mean",
            "brier_score_mean",
            "precision_20_mean",
            "recall_20_mean",
            "jaccard_20_mean",
            "total_folds",
        }
        assert set(headers) == expected

    def test_csv_row_count_matches_models(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig, tmp_path: Path
    ) -> None:
        """CSV should have one row per model."""
        result = run_ladder(synthetic_df, ladder_config)
        write_ladder_artifacts(result, tmp_path)
        csv_df = pd.read_csv(tmp_path / "comparison_summary.csv")
        assert len(csv_df) == len(result.entries)

    def test_markdown_artifact_written(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig, tmp_path: Path
    ) -> None:
        """comparison_summary.md should be created."""
        result = run_ladder(synthetic_df, ladder_config)
        write_ladder_artifacts(result, tmp_path)
        assert (tmp_path / "comparison_summary.md").exists()

    def test_markdown_contains_model_names(
        self, synthetic_df: pd.DataFrame, ladder_config: BacktestConfig, tmp_path: Path
    ) -> None:
        """Markdown should include each model name."""
        result = run_ladder(synthetic_df, ladder_config)
        write_ladder_artifacts(result, tmp_path)
        content = (tmp_path / "comparison_summary.md").read_text()
        for entry in result.entries:
            assert entry.model_name in content
