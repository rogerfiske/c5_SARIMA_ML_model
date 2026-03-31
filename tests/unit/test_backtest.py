"""Unit tests for the rolling-origin backtest engine."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS, TOP_K, VALID_PART_IDS
from c5_forecasting.evaluation.backtest import (
    BacktestConfig,
    extract_actual_parts,
    generate_backtest_windows,
    run_backtest,
)
from c5_forecasting.models.baseline import compute_frequency_scores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_df(n_rows: int, active_count: int = 30) -> pd.DataFrame:
    """Build a synthetic dataset with sequential dates and uniform parts."""
    rows = []
    for i in range(n_rows):
        row: dict[str, Any] = {"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)}
        for j, col in enumerate(PART_COLUMNS):
            row[col] = 1 if j < active_count else 0
        row["row_total"] = active_count
        row["total_class"] = "standard_output"
        row["is_exception_day"] = False
        row["domain_event_label"] = ""
        row["quality_flags"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        df[col] = df[col].astype("Int64")
    return df


def _make_df_with_leak_marker(n_rows: int) -> pd.DataFrame:
    """Build a dataset where P_39 is only active on the last row.

    If the backtest leaks future data, P_39 would appear in folds
    predicting that last row.
    """
    rows = []
    for i in range(n_rows):
        row: dict[str, Any] = {"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)}
        for j, col in enumerate(PART_COLUMNS):
            if j == 38:  # P_39
                row[col] = 1 if i == n_rows - 1 else 0
            else:
                row[col] = 1 if j < 20 else 0
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


# ---------------------------------------------------------------------------
# BacktestConfig tests
# ---------------------------------------------------------------------------
class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_defaults(self) -> None:
        """Default values should be min_train_rows=365, step=1, max_windows=None."""
        config = BacktestConfig()
        assert config.min_train_rows == 365
        assert config.step == 1
        assert config.max_windows is None
        assert config.k == TOP_K
        assert config.model_name == "frequency_baseline"

    def test_to_dict(self) -> None:
        """to_dict should serialize all fields."""
        config = BacktestConfig(min_train_rows=100, step=7, max_windows=50)
        d = config.to_dict()
        assert d["min_train_rows"] == 100
        assert d["step"] == 7
        assert d["max_windows"] == 50
        assert d["k"] == TOP_K
        assert d["model_name"] == "frequency_baseline"


# ---------------------------------------------------------------------------
# Window generation tests
# ---------------------------------------------------------------------------
class TestGenerateBacktestWindows:
    """Tests for generate_backtest_windows."""

    def test_step_1_produces_all_windows(self) -> None:
        """step=1 should produce one window per eligible cutoff."""
        config = BacktestConfig(min_train_rows=10, step=1)
        windows = generate_backtest_windows(20, config)
        # First cutoff at index 9, last at index 18 => 10 windows
        assert len(windows) == 10

    def test_step_produces_fewer_windows(self) -> None:
        """Larger step should produce fewer windows."""
        config_1 = BacktestConfig(min_train_rows=10, step=1)
        config_5 = BacktestConfig(min_train_rows=10, step=5)
        w1 = generate_backtest_windows(100, config_1)
        w5 = generate_backtest_windows(100, config_5)
        assert len(w5) < len(w1)

    def test_max_windows_caps_output(self) -> None:
        """max_windows should limit the number of folds."""
        config = BacktestConfig(min_train_rows=10, step=1, max_windows=5)
        windows = generate_backtest_windows(100, config)
        assert len(windows) == 5

    def test_max_windows_keeps_most_recent(self) -> None:
        """When capped, the most recent windows should be kept."""
        config = BacktestConfig(min_train_rows=10, step=1, max_windows=3)
        windows = generate_backtest_windows(20, config)
        assert len(windows) == 3
        # Last window should target the last row
        assert windows[-1] == (18, 19)

    def test_too_few_rows_raises(self) -> None:
        """Fewer than min_train_rows+1 rows should raise ValueError."""
        config = BacktestConfig(min_train_rows=10)
        with pytest.raises(ValueError, match="need at least"):
            generate_backtest_windows(10, config)

    def test_exact_minimum_rows_produces_one_window(self) -> None:
        """min_train_rows+1 rows should produce exactly 1 window."""
        config = BacktestConfig(min_train_rows=10)
        windows = generate_backtest_windows(11, config)
        assert len(windows) == 1
        assert windows[0] == (9, 10)

    def test_custom_min_train_rows(self) -> None:
        """Custom min_train_rows should shift the first cutoff."""
        config = BacktestConfig(min_train_rows=50)
        windows = generate_backtest_windows(100, config)
        assert windows[0][0] == 49  # cutoff at index 49 = 50 training rows

    def test_window_indices_are_valid(self) -> None:
        """All cutoff_idx and target_idx must be within bounds."""
        config = BacktestConfig(min_train_rows=10, step=3)
        n = 50
        windows = generate_backtest_windows(n, config)
        for cutoff_idx, target_idx in windows:
            assert 0 <= cutoff_idx < n
            assert 0 <= target_idx < n
            assert target_idx == cutoff_idx + 1

    def test_cutoff_target_relationship(self) -> None:
        """target_idx should always be cutoff_idx + 1."""
        config = BacktestConfig(min_train_rows=5, step=2)
        windows = generate_backtest_windows(30, config)
        for cutoff_idx, target_idx in windows:
            assert target_idx == cutoff_idx + 1


# ---------------------------------------------------------------------------
# Extract actual parts tests
# ---------------------------------------------------------------------------
class TestExtractActualParts:
    """Tests for extract_actual_parts."""

    def test_returns_active_ids(self) -> None:
        """Parts with value > 0 should be returned."""
        row = pd.Series({"P_1": 3, "P_2": 0, "P_3": 1, "date": "2020-01-01"})
        parts, total, counts = extract_actual_parts(row)
        assert 1 in parts
        assert 3 in parts
        assert 2 not in parts

    def test_zero_parts_excluded(self) -> None:
        """Parts with value 0 should not appear."""
        data = {col: 0 for col in PART_COLUMNS}
        data["date"] = "2020-01-01"
        row = pd.Series(data)
        parts, total, counts = extract_actual_parts(row)
        assert len(parts) == 0
        assert total == 0
        assert len(counts) == 0

    def test_returns_sorted(self) -> None:
        """Output should be sorted ascending."""
        data = {col: 0 for col in PART_COLUMNS}
        data["P_30"] = 1
        data["P_5"] = 1
        data["P_15"] = 1
        data["date"] = "2020-01-01"
        row = pd.Series(data)
        parts, _, counts = extract_actual_parts(row)
        assert parts == sorted(parts)

    def test_row_total_is_sum(self) -> None:
        """Row total should be sum of all part column values."""
        data = {col: 0 for col in PART_COLUMNS}
        data["P_1"] = 3
        data["P_2"] = 5
        data["date"] = "2020-01-01"
        row = pd.Series(data)
        _, total, counts = extract_actual_parts(row)
        assert total == 8
        assert counts == {1: 3, 2: 5}


# ---------------------------------------------------------------------------
# Run backtest tests
# ---------------------------------------------------------------------------
class TestRunBacktest:
    """Tests for run_backtest."""

    def test_fold_count_matches_windows(self) -> None:
        """Number of folds should match the number of generated windows."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=1)
        result = run_backtest(df, compute_frequency_scores, config)
        expected_windows = generate_backtest_windows(50, config)
        assert len(result.folds) == len(expected_windows)

    def test_no_future_leakage(self) -> None:
        """Training set must never contain the target row's data.

        Uses a dataset where P_39 is only active on the final row.
        The fold predicting the final row must NOT have P_39 in its
        frequency scores (since P_39 only appears in the target).
        """
        n = 50
        df = _make_df_with_leak_marker(n)
        config = BacktestConfig(min_train_rows=10, step=1)
        result = run_backtest(df, compute_frequency_scores, config)

        # The last fold predicts the final row (where P_39 is active)
        last_fold = result.folds[-1]
        assert last_fold.target_date == str(
            (pd.Timestamp("2020-01-01") + pd.Timedelta(days=n - 1)).date()
        )
        # P_39 should have score 0.0 in training (never seen before the target)
        # It should NOT appear in the top-20
        predicted_ids = [r["part_id"] for r in last_fold.predicted_ranking]
        assert 39 not in predicted_ids

    def test_training_fold_grows(self) -> None:
        """Each subsequent fold should have more training rows than the previous."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=1)
        result = run_backtest(df, compute_frequency_scores, config)
        for i in range(1, len(result.folds)):
            assert result.folds[i].train_rows > result.folds[i - 1].train_rows

    def test_hit_count_correctness(self) -> None:
        """hit_count should equal len(set(predicted) & set(actual))."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=5)
        result = run_backtest(df, compute_frequency_scores, config)
        for fold in result.folds:
            predicted_ids = {r["part_id"] for r in fold.predicted_ranking}
            actual_set = set(fold.actual_active_parts)
            expected_hits = len(predicted_ids & actual_set)
            assert fold.hit_count == expected_hits

    def test_deterministic_results(self) -> None:
        """Running twice on same data should produce identical folds."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=5)
        r1 = run_backtest(df, compute_frequency_scores, config)
        r2 = run_backtest(df, compute_frequency_scores, config)
        for f1, f2 in zip(r1.folds, r2.folds, strict=True):
            assert f1.predicted_ranking == f2.predicted_ranking
            assert f1.hit_count == f2.hit_count

    def test_provenance_fields_populated(self) -> None:
        """All provenance fields should be non-empty."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=10)
        result = run_backtest(
            df,
            compute_frequency_scores,
            config,
            dataset_variant="raw",
            dataset_fingerprint="abc123",
            source_fingerprint="def456",
        )
        prov = result.provenance
        assert prov.run_id
        assert prov.run_timestamp
        assert prov.model_name == "frequency_baseline"
        assert prov.dataset_variant == "raw"
        assert prov.dataset_fingerprint == "abc123"
        assert prov.source_fingerprint == "def456"
        assert prov.dataset_row_count == 50
        assert prov.total_folds == len(result.folds)

    def test_summary_statistics_correct(self) -> None:
        """Summary mean/min/max hit counts should match fold data."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=5)
        result = run_backtest(df, compute_frequency_scores, config)
        hit_counts = [f.hit_count for f in result.folds]
        assert result.summary.total_folds == len(result.folds)
        assert result.summary.min_hit_count == min(hit_counts)
        assert result.summary.max_hit_count == max(hit_counts)
        assert abs(result.summary.mean_hit_count - sum(hit_counts) / len(hit_counts)) < 1e-6

    def test_result_is_json_serializable(self) -> None:
        """BacktestResult.to_dict() should produce valid JSON."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=10)
        result = run_backtest(df, compute_frequency_scores, config)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_all_predicted_ids_valid(self) -> None:
        """Every fold's predicted ranking should have valid IDs in 1..39."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=5)
        result = run_backtest(df, compute_frequency_scores, config)
        for fold in result.folds:
            for entry in fold.predicted_ranking:
                assert entry["part_id"] in VALID_PART_IDS

    def test_no_zero_in_any_fold(self) -> None:
        """Part ID 0 must never appear in any fold's predicted ranking."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=5)
        result = run_backtest(df, compute_frequency_scores, config)
        for fold in result.folds:
            ids = [r["part_id"] for r in fold.predicted_ranking]
            assert 0 not in ids

    def test_each_fold_has_k_predictions(self) -> None:
        """Each fold should produce exactly k ranked entries."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=5)
        result = run_backtest(df, compute_frequency_scores, config)
        for fold in result.folds:
            assert len(fold.predicted_ranking) == TOP_K

    def test_cutoff_dates_advance_chronologically(self) -> None:
        """Cutoff dates should advance through time."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=1)
        result = run_backtest(df, compute_frequency_scores, config)
        cutoff_dates = [f.cutoff_date for f in result.folds]
        assert cutoff_dates == sorted(cutoff_dates)

    def test_summary_date_ranges(self) -> None:
        """Summary date ranges should match first/last folds."""
        df = _make_df(50)
        config = BacktestConfig(min_train_rows=10, step=5)
        result = run_backtest(df, compute_frequency_scores, config)
        assert result.summary.first_cutoff_date == result.folds[0].cutoff_date
        assert result.summary.last_cutoff_date == result.folds[-1].cutoff_date
        assert result.summary.first_target_date == result.folds[0].target_date
        assert result.summary.last_target_date == result.folds[-1].target_date
