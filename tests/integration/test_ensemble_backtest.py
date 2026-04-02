"""Integration tests for ensemble models on real data."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.config.settings import get_settings
from c5_forecasting.domain.constants import VALID_PART_IDS
from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
from c5_forecasting.evaluation.ladder import run_ladder
from c5_forecasting.models.ensemble import (
    ensemble_avg_scoring,
    ensemble_rank_avg_scoring,
    ensemble_weighted_scoring,
)


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    """Load real dataset for integration tests."""
    settings = get_settings()
    parquet_path = settings.processed_data_dir / "raw_v1.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Real dataset not found at {parquet_path}")
    return pd.read_parquet(parquet_path)


@pytest.fixture(scope="module")
def ensemble_avg_backtest(real_df: pd.DataFrame):
    """Run ensemble_avg backtest with large step for speed."""
    config = BacktestConfig(min_train_rows=365, step=2000, max_windows=None)
    return run_backtest(real_df, ensemble_avg_scoring, config)


@pytest.fixture(scope="module")
def ensemble_rank_avg_backtest(real_df: pd.DataFrame):
    """Run ensemble_rank_avg backtest with large step for speed."""
    config = BacktestConfig(min_train_rows=365, step=2000, max_windows=None)
    return run_backtest(real_df, ensemble_rank_avg_scoring, config)


@pytest.fixture(scope="module")
def ensemble_weighted_backtest(real_df: pd.DataFrame):
    """Run ensemble_weighted backtest with large step for speed."""
    config = BacktestConfig(min_train_rows=365, step=2000, max_windows=None)
    return run_backtest(real_df, ensemble_weighted_scoring, config)


class TestEnsembleAvgBacktest:
    """Integration tests for ensemble_avg on real data."""

    def test_backtest_completes(self, ensemble_avg_backtest) -> None:
        """Ensemble_avg backtest should complete without error."""
        assert ensemble_avg_backtest is not None
        assert len(ensemble_avg_backtest.folds) > 0

    def test_output_valid_part_ids(self, ensemble_avg_backtest) -> None:
        """All predicted IDs should be in 1..39."""
        for fold in ensemble_avg_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            for pid in predicted_ids:
                assert pid in VALID_PART_IDS, f"Invalid part ID {pid}"

    def test_no_zero_part_ids(self, ensemble_avg_backtest) -> None:
        """No fold should contain part ID 0."""
        for fold in ensemble_avg_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert 0 not in predicted_ids, f"Part ID 0 in fold {fold.fold_index}"

    def test_twenty_predictions_per_fold(self, ensemble_avg_backtest) -> None:
        """Each fold should produce exactly 20 predictions."""
        for fold in ensemble_avg_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert len(predicted_ids) == 20

    def test_deterministic(self, real_df: pd.DataFrame) -> None:
        """Two ensemble_avg backtests should produce identical results."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        r1 = run_backtest(real_df, ensemble_avg_scoring, config)
        r2 = run_backtest(real_df, ensemble_avg_scoring, config)
        for f1, f2 in zip(r1.folds, r2.folds, strict=True):
            ids1 = [r["part_id"] for r in f1.predicted_ranking]
            ids2 = [r["part_id"] for r in f2.predicted_ranking]
            assert ids1 == ids2


class TestEnsembleRankAvgBacktest:
    """Integration tests for ensemble_rank_avg on real data."""

    def test_backtest_completes(self, ensemble_rank_avg_backtest) -> None:
        """Ensemble_rank_avg backtest should complete without error."""
        assert ensemble_rank_avg_backtest is not None
        assert len(ensemble_rank_avg_backtest.folds) > 0

    def test_output_valid_part_ids(self, ensemble_rank_avg_backtest) -> None:
        """All predicted IDs should be in 1..39."""
        for fold in ensemble_rank_avg_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            for pid in predicted_ids:
                assert pid in VALID_PART_IDS

    def test_no_zero_part_ids(self, ensemble_rank_avg_backtest) -> None:
        """No fold should contain part ID 0."""
        for fold in ensemble_rank_avg_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert 0 not in predicted_ids

    def test_twenty_predictions_per_fold(self, ensemble_rank_avg_backtest) -> None:
        """Each fold should produce exactly 20 predictions."""
        for fold in ensemble_rank_avg_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert len(predicted_ids) == 20

    def test_deterministic(self, real_df: pd.DataFrame) -> None:
        """Two ensemble_rank_avg backtests should produce identical results."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        r1 = run_backtest(real_df, ensemble_rank_avg_scoring, config)
        r2 = run_backtest(real_df, ensemble_rank_avg_scoring, config)
        for f1, f2 in zip(r1.folds, r2.folds, strict=True):
            ids1 = [r["part_id"] for r in f1.predicted_ranking]
            ids2 = [r["part_id"] for r in f2.predicted_ranking]
            assert ids1 == ids2


class TestEnsembleWeightedBacktest:
    """Integration tests for ensemble_weighted on real data."""

    def test_backtest_completes(self, ensemble_weighted_backtest) -> None:
        """Ensemble_weighted backtest should complete without error."""
        assert ensemble_weighted_backtest is not None
        assert len(ensemble_weighted_backtest.folds) > 0

    def test_output_valid_part_ids(self, ensemble_weighted_backtest) -> None:
        """All predicted IDs should be in 1..39."""
        for fold in ensemble_weighted_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            for pid in predicted_ids:
                assert pid in VALID_PART_IDS

    def test_no_zero_part_ids(self, ensemble_weighted_backtest) -> None:
        """No fold should contain part ID 0."""
        for fold in ensemble_weighted_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert 0 not in predicted_ids

    def test_twenty_predictions_per_fold(self, ensemble_weighted_backtest) -> None:
        """Each fold should produce exactly 20 predictions."""
        for fold in ensemble_weighted_backtest.folds:
            predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
            assert len(predicted_ids) == 20

    def test_deterministic(self, real_df: pd.DataFrame) -> None:
        """Two ensemble_weighted backtests should produce identical results."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        r1 = run_backtest(real_df, ensemble_weighted_scoring, config)
        r2 = run_backtest(real_df, ensemble_weighted_scoring, config)
        for f1, f2 in zip(r1.folds, r2.folds, strict=True):
            ids1 = [r["part_id"] for r in f1.predicted_ranking]
            ids2 = [r["part_id"] for r in f2.predicted_ranking]
            assert ids1 == ids2


class TestEnsemblesInLadder:
    """Integration tests for ensembles in ladder comparison."""

    def test_all_ensembles_in_ladder(self, real_df: pd.DataFrame) -> None:
        """run_ladder should include all 3 ensemble models."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        ensemble_names = ["ensemble_avg", "ensemble_rank_avg", "ensemble_weighted"]
        result = run_ladder(real_df, config, model_names=ensemble_names)
        model_names = {e.model_name for e in result.entries}
        assert model_names == set(ensemble_names)

    def test_no_leakage_in_ensembles(self, real_df: pd.DataFrame) -> None:
        """Ensemble predictions should not leak future data."""
        config = BacktestConfig(min_train_rows=365, step=2000, max_windows=2)
        result = run_backtest(real_df, ensemble_avg_scoring, config)
        for fold in result.folds:
            # Cutoff date must be before target date
            assert fold.cutoff_date < fold.target_date
            # No actual counts from target date should be visible
            train_end = real_df[real_df["date"] <= fold.cutoff_date].index.max()
            target_idx = real_df[real_df["date"] == fold.target_date].index[0]
            assert train_end < target_idx
