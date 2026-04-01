"""Integration tests for the baseline ladder on real data."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.config.settings import get_settings
from c5_forecasting.domain.constants import VALID_PART_IDS
from c5_forecasting.evaluation.backtest import BacktestConfig
from c5_forecasting.evaluation.ladder import LadderResult, run_ladder, write_ladder_artifacts


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    """Load real dataset for integration tests."""
    settings = get_settings()
    parquet_path = settings.processed_data_dir / "raw_v1.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Real dataset not found at {parquet_path}")
    return pd.read_parquet(parquet_path)


@pytest.fixture(scope="module")
def ladder_result(real_df: pd.DataFrame) -> LadderResult:
    """Run ladder once on real data with large step for speed."""
    config = BacktestConfig(min_train_rows=365, step=2000, max_windows=None)
    return run_ladder(real_df, config)


class TestRealLadder:
    """Integration tests for ladder on real data."""

    def test_ladder_completes_on_real_data(self, ladder_result: LadderResult) -> None:
        """Full ladder should run without error on real dataset."""
        assert isinstance(ladder_result, LadderResult)
        assert len(ladder_result.entries) == 4

    def test_all_models_produce_valid_forecasts(self, ladder_result: LadderResult) -> None:
        """Each model's folds should have exactly 20 valid IDs in 1..39."""
        for entry in ladder_result.entries:
            for fold in entry.backtest_result.folds:
                predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
                n = len(predicted_ids)
                assert n == 20, f"{entry.model_name}: fold {fold.fold_index} has {n} predictions"
                assert 0 not in predicted_ids, (
                    f"{entry.model_name}: fold {fold.fold_index} contains 0"
                )
                for pid in predicted_ids:
                    assert pid in VALID_PART_IDS
                assert len(set(predicted_ids)) == 20

    def test_no_future_leakage_any_model(self, ladder_result: LadderResult) -> None:
        """Cutoff date < target date for all folds in all models."""
        for entry in ladder_result.entries:
            for fold in entry.backtest_result.folds:
                assert fold.cutoff_date < fold.target_date, (
                    f"{entry.model_name}: leakage at fold {fold.fold_index}"
                )

    def test_comparison_artifacts_exist(self, ladder_result: LadderResult, tmp_path: Path) -> None:
        """All 3 comparison files should be written."""
        paths = write_ladder_artifacts(ladder_result, tmp_path)
        assert len(paths) == 3
        for p in paths:
            assert Path(p).exists()

    def test_frequency_baseline_beats_uniform(self, ladder_result: LadderResult) -> None:
        """Frequency baseline nDCG@20 > uniform (sanity check)."""
        freq = next(e for e in ladder_result.entries if e.model_name == "frequency_baseline")
        unif = next(e for e in ladder_result.entries if e.model_name == "uniform_baseline")
        assert freq.metric_summary.ndcg_20_mean > unif.metric_summary.ndcg_20_mean

    def test_deterministic_ladder(self, real_df: pd.DataFrame) -> None:
        """Running ladder twice should produce same model ranking."""
        config = BacktestConfig(min_train_rows=365, step=2000)
        result1 = run_ladder(real_df, config)
        result2 = run_ladder(real_df, config)

        names1 = [e.model_name for e in result1.entries]
        names2 = [e.model_name for e in result2.entries]
        assert names1 == names2

        for e1, e2 in zip(result1.entries, result2.entries, strict=False):
            assert e1.metric_summary.ndcg_20_mean == pytest.approx(e2.metric_summary.ndcg_20_mean)


class TestLadderCli:
    """CLI integration tests for ladder and backtest --model."""

    def test_cli_ladder_exits_zero(self) -> None:
        """ladder --step 2000 should exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "ladder", "--step", "2000"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "Ladder PASSED" in result.stdout

    def test_cli_backtest_with_model_flag(self) -> None:
        """backtest --model recency_weighted --step 2000 should exit 0."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "c5_forecasting",
                "backtest",
                "--model",
                "recency_weighted",
                "--step",
                "2000",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "Backtest PASSED" in result.stdout
