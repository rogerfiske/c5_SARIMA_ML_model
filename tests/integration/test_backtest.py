"""Integration tests for rolling-origin backtesting against the real dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.data.annotation import load_annotation_config
from c5_forecasting.data.dataset_builder import build_raw_dataset
from c5_forecasting.domain.constants import TOP_K, VALID_PART_IDS
from c5_forecasting.evaluation.artifacts import write_backtest_artifacts
from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
from c5_forecasting.models.baseline import compute_frequency_scores

_RAW_CSV = Path("data/raw/c5_aggregated_matrix.csv")
_CONFIG_YAML = Path("configs/datasets/event_annotations.yaml")


@pytest.fixture()
def real_dataset(tmp_path: Path) -> Path:
    """Build the real raw dataset as Parquet and return its path."""
    if not _RAW_CSV.exists():
        pytest.skip("Real CSV not available")
    if not _CONFIG_YAML.exists():
        pytest.skip("Annotation config not available")
    config = load_annotation_config(_CONFIG_YAML)
    build_raw_dataset(_RAW_CSV, config, tmp_path)
    return tmp_path / "raw_v1.parquet"


class TestRealBacktest:
    """Integration tests for backtest on the real dataset."""

    def test_backtest_completes(self, real_dataset: Path, tmp_path: Path) -> None:
        """A backtest with large step on real data should complete."""
        df = pd.read_parquet(real_dataset)
        config = BacktestConfig(min_train_rows=365, step=500)
        result = run_backtest(df, compute_frequency_scores, config)
        assert result.summary.total_folds > 0

    def test_no_future_leakage_real_data(self, real_dataset: Path) -> None:
        """Every fold cutoff date must be strictly less than target date."""
        df = pd.read_parquet(real_dataset)
        config = BacktestConfig(min_train_rows=365, step=1000)
        result = run_backtest(df, compute_frequency_scores, config)
        for fold in result.folds:
            assert fold.cutoff_date < fold.target_date

    def test_all_forecasts_valid(self, real_dataset: Path) -> None:
        """Every fold's forecast must have exactly 20 valid entries."""
        df = pd.read_parquet(real_dataset)
        config = BacktestConfig(min_train_rows=365, step=1000)
        result = run_backtest(df, compute_frequency_scores, config)
        for fold in result.folds:
            assert len(fold.predicted_ranking) == TOP_K
            for entry in fold.predicted_ranking:
                assert entry["part_id"] in VALID_PART_IDS
                assert entry["part_id"] != 0

    def test_hit_counts_are_plausible(self, real_dataset: Path) -> None:
        """Mean hit count on real data should be > 0 and <= 20."""
        df = pd.read_parquet(real_dataset)
        config = BacktestConfig(min_train_rows=365, step=500)
        result = run_backtest(df, compute_frequency_scores, config)
        assert result.summary.mean_hit_count > 0
        assert result.summary.mean_hit_count <= TOP_K

    def test_artifacts_written(self, real_dataset: Path, tmp_path: Path) -> None:
        """All three artifact files should exist after running."""
        df = pd.read_parquet(real_dataset)
        config = BacktestConfig(min_train_rows=365, step=1000)
        result = run_backtest(df, compute_frequency_scores, config)
        out_dir = tmp_path / "artifacts"
        paths = write_backtest_artifacts(result, out_dir)
        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists()

    def test_json_artifact_has_folds(self, real_dataset: Path, tmp_path: Path) -> None:
        """JSON artifact should contain all fold results."""
        df = pd.read_parquet(real_dataset)
        config = BacktestConfig(min_train_rows=365, step=1000)
        result = run_backtest(df, compute_frequency_scores, config)
        out_dir = tmp_path / "artifacts"
        write_backtest_artifacts(result, out_dir)
        data = json.loads((out_dir / "backtest_results.json").read_text())
        assert len(data["folds"]) == result.summary.total_folds

    def test_deterministic_on_real_data(self, real_dataset: Path) -> None:
        """Running twice on the same real dataset produces identical results."""
        df = pd.read_parquet(real_dataset)
        config = BacktestConfig(min_train_rows=365, step=2000)
        r1 = run_backtest(df, compute_frequency_scores, config)
        r2 = run_backtest(df, compute_frequency_scores, config)
        for f1, f2 in zip(r1.folds, r2.folds, strict=True):
            assert f1.predicted_ranking == f2.predicted_ranking
            assert f1.hit_count == f2.hit_count


class TestBacktestCli:
    """Integration tests for the backtest CLI command."""

    def test_cli_backtest_exits_zero(self) -> None:
        """Running the backtest command must exit 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "c5_forecasting",
                "backtest",
                "--step",
                "2000",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Backtest PASSED" in result.stdout

    def test_cli_backtest_prints_summary(self) -> None:
        """CLI output should include summary statistics."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "c5_forecasting",
                "backtest",
                "--step",
                "2000",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Total folds" in result.stdout
        assert "Mean hit count" in result.stdout
