"""Integration tests for the GBM ranking model on real data."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.config.settings import get_settings
from c5_forecasting.domain.constants import VALID_PART_IDS
from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
from c5_forecasting.evaluation.ladder import run_ladder
from c5_forecasting.models.gbm_ranking import gbm_ranking_scoring


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    """Load real dataset for integration tests."""
    settings = get_settings()
    parquet_path = settings.processed_data_dir / "raw_v1.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Real dataset not found at {parquet_path}")
    return pd.read_parquet(parquet_path)


@pytest.fixture(scope="module")
def gbm_backtest(real_df: pd.DataFrame):
    """Run a single GBM ranking backtest with large step for speed."""
    config = BacktestConfig(min_train_rows=365, step=2000, max_windows=None)
    return run_backtest(real_df, gbm_ranking_scoring, config)


class TestGbmBacktest:
    """Integration tests for GBM ranking on real data."""

    def test_gbm_single_fold_backtest(self, gbm_backtest) -> None:
        """GBM backtest should complete without error."""
        assert gbm_backtest is not None
        assert len(gbm_backtest.folds) > 0

    def test_gbm_output_valid_part_ids(self, gbm_backtest) -> None:
        """All predicted IDs should be in 1..39."""
        for fold in gbm_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            for pid in predicted_ids:
                assert pid in VALID_PART_IDS, f"Invalid part ID {pid} in fold {fold.fold_index}"

    def test_gbm_no_zero_part_ids(self, gbm_backtest) -> None:
        """No fold should contain part ID 0."""
        for fold in gbm_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert 0 not in predicted_ids, f"Part ID 0 in fold {fold.fold_index}"

    def test_gbm_twenty_predictions_per_fold(self, gbm_backtest) -> None:
        """Each fold should produce exactly 20 predictions."""
        for fold in gbm_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert len(predicted_ids) == 20, (
                f"Fold {fold.fold_index} has {len(predicted_ids)} predictions"
            )

    def test_gbm_deterministic(self, real_df: pd.DataFrame) -> None:
        """Two GBM backtests should produce the same results."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        r1 = run_backtest(real_df, gbm_ranking_scoring, config)
        r2 = run_backtest(real_df, gbm_ranking_scoring, config)
        for f1, f2 in zip(r1.folds, r2.folds, strict=True):
            ids1 = [r["part_id"] for r in f1.predicted_ranking]
            ids2 = [r["part_id"] for r in f2.predicted_ranking]
            assert ids1 == ids2

    def test_gbm_in_ladder(self, real_df: pd.DataFrame) -> None:
        """run_ladder should include gbm_ranking in its results."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        result = run_ladder(real_df, config, model_names=["gbm_ranking", "frequency_baseline"])
        model_names = {e.model_name for e in result.entries}
        assert "gbm_ranking" in model_names

    def test_gbm_backtest_artifacts(self, gbm_backtest) -> None:
        """Backtest result should have non-empty fold data."""
        for fold in gbm_backtest.folds:
            assert fold.predicted_ranking
            assert fold.cutoff_date < fold.target_date
