"""Unit tests for backtest artifact generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from c5_forecasting.domain.constants import PART_COLUMNS
from c5_forecasting.evaluation.artifacts import write_backtest_artifacts
from c5_forecasting.evaluation.backtest import (
    BacktestConfig,
    run_backtest,
)
from c5_forecasting.models.baseline import compute_frequency_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_df(n_rows: int = 50) -> pd.DataFrame:
    """Build a synthetic dataset for artifact testing."""
    rows = []
    for i in range(n_rows):
        row: dict[str, Any] = {"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)}
        for j, col in enumerate(PART_COLUMNS):
            row[col] = 1 if j < 30 else 0
        row["row_total"] = 30
        row["total_class"] = "standard_output"
        row["is_exception_day"] = False
        row["domain_event_label"] = ""
        row["quality_flags"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        df[col] = df[col].astype("Int64")
    return df


def _run_small_backtest() -> tuple:
    """Run a small backtest and return (result, config)."""
    df = _make_df(50)
    config = BacktestConfig(min_train_rows=10, step=10)
    result = run_backtest(df, compute_frequency_scores, config)
    return result, config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestWriteBacktestArtifacts:
    """Tests for backtest artifact generation."""

    def test_json_artifact_written(self, tmp_path: Path) -> None:
        """JSON backtest results file must be created."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        assert (out_dir / "backtest_results.json").exists()

    def test_json_contains_provenance(self, tmp_path: Path) -> None:
        """JSON must contain a provenance key."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        data = json.loads((out_dir / "backtest_results.json").read_text())
        assert "provenance" in data
        assert data["provenance"]["model_name"] == "frequency_baseline"

    def test_json_contains_all_folds(self, tmp_path: Path) -> None:
        """JSON folds array must match the number of folds."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        data = json.loads((out_dir / "backtest_results.json").read_text())
        assert len(data["folds"]) == len(result.folds)

    def test_csv_artifact_written(self, tmp_path: Path) -> None:
        """CSV summary file must be created."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        assert (out_dir / "backtest_summary.csv").exists()

    def test_csv_has_correct_columns(self, tmp_path: Path) -> None:
        """CSV must have the expected column schema."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        df = pd.read_csv(out_dir / "backtest_summary.csv")
        expected_cols = [
            "fold_index",
            "cutoff_date",
            "target_date",
            "train_rows",
            "hit_count",
            "actual_row_total",
            "actual_active_count",
            "predicted_top_1",
        ]
        assert list(df.columns) == expected_cols

    def test_csv_row_count_matches_folds(self, tmp_path: Path) -> None:
        """CSV should have one row per fold."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        df = pd.read_csv(out_dir / "backtest_summary.csv")
        assert len(df) == len(result.folds)

    def test_markdown_artifact_written(self, tmp_path: Path) -> None:
        """Markdown summary file must be created."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        assert (out_dir / "backtest_summary.md").exists()

    def test_markdown_contains_provenance(self, tmp_path: Path) -> None:
        """Markdown should include provenance header."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        write_backtest_artifacts(result, out_dir)
        content = (out_dir / "backtest_summary.md").read_text()
        assert "Run ID" in content
        assert "frequency_baseline" in content

    def test_returns_three_artifact_paths(self, tmp_path: Path) -> None:
        """Should return exactly 3 artifact paths."""
        result, _ = _run_small_backtest()
        out_dir = tmp_path / "out"
        paths = write_backtest_artifacts(result, out_dir)
        assert len(paths) == 3
