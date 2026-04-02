"""Unit tests for prediction export module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.domain.constants import VALID_PART_IDS
from c5_forecasting.evaluation.backtest import BacktestFold, BacktestProvenance, BacktestResult
from c5_forecasting.evaluation.metrics import FoldMetrics
from c5_forecasting.evaluation.prediction_export import (
    _format_actual_parts,
    write_daily_predictions_csv,
    write_simple_predictions_csv,
)


def _make_mock_fold(
    fold_index: int,
    cutoff_date: str,
    target_date: str,
    train_rows: int,
) -> BacktestFold:
    """Create a mock BacktestFold with valid predictions."""
    # Create 20 valid predictions (IDs 1-20 for simplicity)
    predicted_ranking = [
        {"rank": i, "part_id": i, "score": 1.0 - (i - 1) * 0.04} for i in range(1, 21)
    ]

    # Mock actual parts (first 5 predictions are correct)
    actual_part_counts = {i: i * 2 for i in range(1, 6)}

    return BacktestFold(
        fold_index=fold_index,
        cutoff_date=cutoff_date,
        target_date=target_date,
        train_rows=train_rows,
        predicted_ranking=predicted_ranking,
        actual_active_parts=list(actual_part_counts.keys()),
        actual_row_total=sum(actual_part_counts.values()),
        hit_count=5,
        hit_parts=[1, 2, 3, 4, 5],
        miss_parts=[],
        actual_part_counts=actual_part_counts,
        all_scores=[{"part_id": i, "score": 0.5} for i in VALID_PART_IDS],
    )


def _make_mock_metrics() -> FoldMetrics:
    """Create mock FoldMetrics."""
    return FoldMetrics(
        fold_index=0,
        ndcg_20=0.8,
        weighted_recall_20=0.75,
        brier_score=0.15,
        precision_20=0.25,
        recall_20=0.5,
        jaccard_20=0.3,
    )


def _make_mock_result(folds: list[BacktestFold]) -> BacktestResult:
    """Create mock BacktestResult."""
    provenance = BacktestProvenance(
        run_id="test-run-001",
        run_timestamp="2024-01-01T00:00:00Z",
        model_name="frequency_baseline",
        dataset_variant="raw",
        dataset_fingerprint="abc123",
        source_fingerprint="def456",
        config={"min_train_rows": 365, "step": 1, "k": 20},
        dataset_row_count=1000,
        dataset_date_min="2008-09-08",
        dataset_date_max="2026-03-29",
        total_folds=len(folds),
    )

    @dataclass
    class MockSummary:
        total_folds: int
        mean_hit_count: float
        min_hit_count: int
        max_hit_count: int
        first_cutoff_date: str
        last_cutoff_date: str
        first_target_date: str
        last_target_date: str

    summary = MockSummary(
        total_folds=len(folds),
        mean_hit_count=5.0,
        min_hit_count=5,
        max_hit_count=5,
        first_cutoff_date=folds[0].cutoff_date if folds else "",
        last_cutoff_date=folds[-1].cutoff_date if folds else "",
        first_target_date=folds[0].target_date if folds else "",
        last_target_date=folds[-1].target_date if folds else "",
    )

    return BacktestResult(
        provenance=provenance,
        folds=folds,
        summary=summary,  # type: ignore
        artifacts=[],
    )


class TestFormatActualParts:
    """Tests for _format_actual_parts helper."""

    def test_empty_dict(self) -> None:
        """Empty dict returns empty string."""
        assert _format_actual_parts({}) == ""

    def test_single_part(self) -> None:
        """Single part returns single ID."""
        assert _format_actual_parts({5: 10}) == "5"

    def test_multiple_parts_sorted(self) -> None:
        """Multiple parts returned sorted ascending with pipe delimiter."""
        actual_counts = {30: 5, 12: 3, 5: 8, 27: 2}
        result = _format_actual_parts(actual_counts)
        assert result == "5|12|27|30"

    def test_all_valid_ids(self) -> None:
        """All part IDs should be valid (1-39)."""
        actual_counts = {i: i for i in VALID_PART_IDS}
        result = _format_actual_parts(actual_counts)
        parts = [int(p) for p in result.split("|")]
        assert all(p in VALID_PART_IDS for p in parts)


class TestWriteDailyPredictionsCSV:
    """Tests for write_daily_predictions_csv function."""

    def test_csv_structure(self, tmp_path: Path) -> None:
        """CSV has correct column count and names."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)

        # Verify column count (5 metadata + 20 preds + 20 scores + 2 actuals + 5 metrics)
        assert len(df.columns) == 52

        # Verify key columns exist
        expected_cols = {
            "target_date",
            "cutoff_date",
            "model_name",
            "dataset_variant",
            "train_rows",
            "pred_01",
            "pred_20",
            "score_01",
            "score_20",
            "actual_nonzero_parts",
            "actual_hit_count_top20",
            "ndcg_20",
            "weighted_recall_20",
            "precision_20",
            "recall_20",
            "jaccard_20",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_no_zeros_in_predictions(self, tmp_path: Path) -> None:
        """All pred_XX columns contain no zeros."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)

        # Check all pred_XX columns
        pred_cols = [f"pred_{i:02d}" for i in range(1, 21)]
        for col in pred_cols:
            assert 0 not in df[col].values, f"Found 0 in {col}"

    def test_one_row_per_fold(self, tmp_path: Path) -> None:
        """CSV has one row per fold."""
        folds = [
            _make_mock_fold(0, "2009-09-07", "2009-09-08", 365),
            _make_mock_fold(1, "2009-09-08", "2009-09-09", 366),
            _make_mock_fold(2, "2009-09-09", "2009-09-10", 367),
        ]
        metrics = [_make_mock_metrics() for _ in range(3)]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)
        assert len(df) == 3

    def test_deterministic_output(self, tmp_path: Path) -> None:
        """Same input produces same output."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output1 = tmp_path / "export1.csv"
        output2 = tmp_path / "export2.csv"

        write_daily_predictions_csv(result, metrics, output1)
        write_daily_predictions_csv(result, metrics, output2)

        df1 = pd.read_csv(output1)
        df2 = pd.read_csv(output2)

        pd.testing.assert_frame_equal(df1, df2)

    def test_pred_columns_are_valid_ids(self, tmp_path: Path) -> None:
        """All pred_XX values are in 1-39."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)

        pred_cols = [f"pred_{i:02d}" for i in range(1, 21)]
        for col in pred_cols:
            for val in df[col]:
                assert val in VALID_PART_IDS, f"{col} contains invalid ID {val}"

    def test_score_columns_are_floats(self, tmp_path: Path) -> None:
        """All score_XX values are floats."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)

        score_cols = [f"score_{i:02d}" for i in range(1, 21)]
        for col in score_cols:
            assert df[col].dtype == float, f"{col} is not float type"

    def test_actual_parts_format(self, tmp_path: Path) -> None:
        """actual_nonzero_parts is pipe-delimited and sorted."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)

        actual_parts_str = df["actual_nonzero_parts"].iloc[0]
        parts = [int(p) for p in actual_parts_str.split("|")]

        # Should be sorted ascending
        assert parts == sorted(parts)

        # Should be 1, 2, 3, 4, 5 (from mock fold)
        assert parts == [1, 2, 3, 4, 5]

    def test_metrics_columns_present(self, tmp_path: Path) -> None:
        """All metric columns are present with correct values."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)

        assert "ndcg_20" in df.columns
        assert "weighted_recall_20" in df.columns
        assert "precision_20" in df.columns
        assert "recall_20" in df.columns
        assert "jaccard_20" in df.columns

        # Verify values match mock metrics
        assert df["ndcg_20"].iloc[0] == pytest.approx(0.8)
        assert df["weighted_recall_20"].iloc[0] == pytest.approx(0.75)
        assert df["precision_20"].iloc[0] == pytest.approx(0.25)
        assert df["recall_20"].iloc[0] == pytest.approx(0.5)
        assert df["jaccard_20"].iloc[0] == pytest.approx(0.3)

    def test_model_name_column(self, tmp_path: Path) -> None:
        """model_name column contains correct model name."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics()]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"
        write_daily_predictions_csv(result, metrics, output_path)

        df = pd.read_csv(output_path)

        assert "model_name" in df.columns
        assert df["model_name"].iloc[0] == "frequency_baseline"

    def test_raises_on_zero_prediction(self, tmp_path: Path) -> None:
        """Raises ValueError if prediction contains part ID 0."""
        fold = _make_mock_fold(0, "2009-09-07", "2009-09-08", 365)

        # Inject a zero (invalid)
        fold.predicted_ranking[0]["part_id"] = 0

        metrics = [_make_mock_metrics()]
        result = _make_mock_result([fold])

        output_path = tmp_path / "test_export.csv"

        with pytest.raises(ValueError, match="part_id=0"):
            write_daily_predictions_csv(result, metrics, output_path)

    def test_raises_on_fold_metrics_mismatch(self, tmp_path: Path) -> None:
        """Raises ValueError if fold count doesn't match metrics count."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        metrics = [_make_mock_metrics(), _make_mock_metrics()]  # 2 metrics, 1 fold
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_export.csv"

        with pytest.raises(ValueError, match="Fold count mismatch"):
            write_daily_predictions_csv(result, metrics, output_path)


class TestWriteSimplePredictionsCSV:
    """Tests for write_simple_predictions_csv function."""

    def test_simple_csv_structure(self, tmp_path: Path) -> None:
        """Simple CSV has 21 columns (M/D/YYYY + pred-1..pred-20)."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_simple.csv"
        write_simple_predictions_csv(result, output_path)

        df = pd.read_csv(output_path)

        # Verify column count (1 date + 20 preds)
        assert len(df.columns) == 21

        # Verify column names
        expected_cols = ["M/D/YYYY"] + [f"pred-{i}" for i in range(1, 21)]
        assert list(df.columns) == expected_cols

    def test_simple_csv_date_format(self, tmp_path: Path) -> None:
        """Date column uses M/D/YYYY format."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_simple.csv"
        write_simple_predictions_csv(result, output_path)

        df = pd.read_csv(output_path)

        # Check first date is in M/D/YYYY format (9/8/2009)
        assert df["M/D/YYYY"].iloc[0] == "9/8/2009"

    def test_simple_csv_no_zeros(self, tmp_path: Path) -> None:
        """All pred-XX columns contain no zeros."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_simple.csv"
        write_simple_predictions_csv(result, output_path)

        df = pd.read_csv(output_path)

        # Check all pred-XX columns
        pred_cols = [f"pred-{i}" for i in range(1, 21)]
        for col in pred_cols:
            assert 0 not in df[col].values, f"Found 0 in {col}"

    def test_simple_csv_valid_part_ids(self, tmp_path: Path) -> None:
        """All predictions are valid part IDs (1-39)."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        result = _make_mock_result(folds)

        output_path = tmp_path / "test_simple.csv"
        write_simple_predictions_csv(result, output_path)

        df = pd.read_csv(output_path)

        pred_cols = [f"pred-{i}" for i in range(1, 21)]
        for col in pred_cols:
            for val in df[col]:
                assert val in VALID_PART_IDS, f"{col} contains invalid ID {val}"

    def test_simple_csv_deterministic(self, tmp_path: Path) -> None:
        """Same input produces same output."""
        folds = [_make_mock_fold(0, "2009-09-07", "2009-09-08", 365)]
        result = _make_mock_result(folds)

        output1 = tmp_path / "simple1.csv"
        output2 = tmp_path / "simple2.csv"

        write_simple_predictions_csv(result, output1)
        write_simple_predictions_csv(result, output2)

        df1 = pd.read_csv(output1)
        df2 = pd.read_csv(output2)

        pd.testing.assert_frame_equal(df1, df2)

    def test_simple_csv_raises_on_zero(self, tmp_path: Path) -> None:
        """Raises ValueError if prediction contains part ID 0."""
        fold = _make_mock_fold(0, "2009-09-07", "2009-09-08", 365)
        fold.predicted_ranking[0]["part_id"] = 0  # Inject zero

        result = _make_mock_result([fold])
        output_path = tmp_path / "test_simple.csv"

        with pytest.raises(ValueError, match="part_id=0"):
            write_simple_predictions_csv(result, output_path)
