"""Integration tests for prediction export on real data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.config.settings import get_settings
from c5_forecasting.domain.constants import VALID_PART_IDS
from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
from c5_forecasting.evaluation.metrics import compute_backtest_metrics
from c5_forecasting.evaluation.prediction_export import write_daily_predictions_csv
from c5_forecasting.models.baseline import compute_frequency_scores


@pytest.fixture(scope="module")
def real_df() -> pd.DataFrame:
    """Load real dataset for integration tests."""
    settings = get_settings()
    parquet_path = settings.processed_data_dir / "raw_v1.parquet"
    if not parquet_path.exists():
        pytest.skip(f"Real dataset not found at {parquet_path}")
    return pd.read_parquet(parquet_path)


class TestPredictionExportIntegration:
    """Integration tests for prediction export on real data."""

    def test_export_completes_on_real_data(self, real_df: pd.DataFrame, tmp_path: Path) -> None:
        """Full export with frequency_baseline completes successfully."""
        config = BacktestConfig(
            min_train_rows=365, step=100, max_windows=10, model_name="frequency_baseline"
        )

        result = run_backtest(real_df, compute_frequency_scores, config)
        fold_metrics, _ = compute_backtest_metrics(result)

        output_path = tmp_path / "predictions.csv"
        export_path = write_daily_predictions_csv(result, fold_metrics, output_path)

        assert export_path.exists()
        assert export_path == output_path

    def test_export_has_multiple_rows(self, real_df: pd.DataFrame, tmp_path: Path) -> None:
        """Export produces multiple rows."""
        config = BacktestConfig(
            min_train_rows=365, step=100, max_windows=10, model_name="frequency_baseline"
        )

        result = run_backtest(real_df, compute_frequency_scores, config)
        fold_metrics, _ = compute_backtest_metrics(result)

        output_path = tmp_path / "predictions.csv"
        write_daily_predictions_csv(result, fold_metrics, output_path)

        df = pd.read_csv(output_path)
        assert len(df) == 10  # max_windows=10

    def test_export_first_row_valid(self, real_df: pd.DataFrame, tmp_path: Path) -> None:
        """First row has valid dates and predictions."""
        config = BacktestConfig(
            min_train_rows=365, step=100, max_windows=5, model_name="frequency_baseline"
        )

        result = run_backtest(real_df, compute_frequency_scores, config)
        fold_metrics, _ = compute_backtest_metrics(result)

        output_path = tmp_path / "predictions.csv"
        write_daily_predictions_csv(result, fold_metrics, output_path)

        df = pd.read_csv(output_path)

        # Check first row
        first_row = df.iloc[0]

        # Dates should be strings
        assert isinstance(first_row["target_date"], str)
        assert isinstance(first_row["cutoff_date"], str)

        # Model name should match
        assert first_row["model_name"] == "frequency_baseline"

        # Predictions should be valid
        pred_01 = first_row["pred_01"]
        assert pred_01 in VALID_PART_IDS

    def test_export_no_zeros_in_predictions(self, real_df: pd.DataFrame, tmp_path: Path) -> None:
        """Scan all pred_XX columns for zeros."""
        config = BacktestConfig(
            min_train_rows=365, step=100, max_windows=10, model_name="frequency_baseline"
        )

        result = run_backtest(real_df, compute_frequency_scores, config)
        fold_metrics, _ = compute_backtest_metrics(result)

        output_path = tmp_path / "predictions.csv"
        write_daily_predictions_csv(result, fold_metrics, output_path)

        df = pd.read_csv(output_path)

        # Check all pred_XX columns
        pred_cols = [f"pred_{i:02d}" for i in range(1, 21)]
        for col in pred_cols:
            assert 0 not in df[col].values, f"Found 0 in {col}"

    def test_export_deterministic(self, real_df: pd.DataFrame, tmp_path: Path) -> None:
        """Two exports with same params produce identical output."""
        config = BacktestConfig(
            min_train_rows=365, step=100, max_windows=5, model_name="frequency_baseline"
        )

        # First export
        result1 = run_backtest(real_df, compute_frequency_scores, config)
        fold_metrics1, _ = compute_backtest_metrics(result1)
        output1 = tmp_path / "export1.csv"
        write_daily_predictions_csv(result1, fold_metrics1, output1)

        # Second export
        result2 = run_backtest(real_df, compute_frequency_scores, config)
        fold_metrics2, _ = compute_backtest_metrics(result2)
        output2 = tmp_path / "export2.csv"
        write_daily_predictions_csv(result2, fold_metrics2, output2)

        # Compare
        df1 = pd.read_csv(output1)
        df2 = pd.read_csv(output2)

        pd.testing.assert_frame_equal(df1, df2)
