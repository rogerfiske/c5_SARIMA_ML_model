"""Integration tests for the SARIMA model on real data."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.config.settings import get_settings
from c5_forecasting.domain.constants import VALID_PART_IDS
from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
from c5_forecasting.evaluation.ladder import run_ladder
from c5_forecasting.models.sarima import sarima_scoring


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    """Load real dataset for integration tests."""
    settings = get_settings()
    parquet_path = settings.processed_data_dir / "raw_v1.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Real dataset not found at {parquet_path}")
    return pd.read_parquet(parquet_path)


@pytest.fixture(scope="module")
def sarima_backtest(real_df: pd.DataFrame):
    """Run a single SARIMA backtest with large step for speed."""
    config = BacktestConfig(min_train_rows=365, step=2000, max_windows=None)
    return run_backtest(real_df, sarima_scoring, config)


class TestSarimaBacktest:
    """Integration tests for SARIMA on real data."""

    def test_sarima_single_fold_backtest(self, sarima_backtest) -> None:
        """SARIMA backtest should complete without error."""
        assert sarima_backtest is not None
        assert len(sarima_backtest.folds) > 0

    def test_sarima_output_valid_part_ids(self, sarima_backtest) -> None:
        """All predicted IDs should be in 1..39."""
        for fold in sarima_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            for pid in predicted_ids:
                assert pid in VALID_PART_IDS, f"Invalid part ID {pid} in fold {fold.fold_index}"

    def test_sarima_no_zero_part_ids(self, sarima_backtest) -> None:
        """No fold should contain part ID 0."""
        for fold in sarima_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert 0 not in predicted_ids, f"Part ID 0 in fold {fold.fold_index}"

    def test_sarima_twenty_predictions_per_fold(self, sarima_backtest) -> None:
        """Each fold should produce exactly 20 predictions."""
        for fold in sarima_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert len(predicted_ids) == 20, (
                f"Fold {fold.fold_index} has {len(predicted_ids)} predictions"
            )

    def test_sarima_deterministic(self, real_df: pd.DataFrame) -> None:
        """Two SARIMA backtests should produce the same results."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        r1 = run_backtest(real_df, sarima_scoring, config)
        r2 = run_backtest(real_df, sarima_scoring, config)
        for f1, f2 in zip(r1.folds, r2.folds, strict=True):
            ids1 = [r["part_id"] for r in f1.predicted_ranking]
            ids2 = [r["part_id"] for r in f2.predicted_ranking]
            assert ids1 == ids2

    def test_sarima_in_ladder(self, real_df: pd.DataFrame) -> None:
        """run_ladder should include sarima in its results."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        result = run_ladder(real_df, config, model_names=["sarima", "frequency_baseline"])
        model_names = {e.model_name for e in result.entries}
        assert "sarima" in model_names

    def test_sarima_backtest_artifacts(self, sarima_backtest) -> None:
        """Backtest result should have non-empty fold data."""
        for fold in sarima_backtest.folds:
            assert fold.predicted_ranking
            assert fold.cutoff_date < fold.target_date
