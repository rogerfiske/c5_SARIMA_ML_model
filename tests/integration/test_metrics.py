"""Integration tests for ranking and calibration metrics on real data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.data.annotation import load_annotation_config
from c5_forecasting.data.dataset_builder import build_raw_dataset
from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
from c5_forecasting.evaluation.metric_report import write_metric_report
from c5_forecasting.evaluation.metrics import compute_backtest_metrics
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


@pytest.fixture()
def real_backtest_result(real_dataset: Path):
    """Run a backtest on real data and return the result."""
    df = pd.read_parquet(real_dataset)
    config = BacktestConfig(min_train_rows=365, step=1000)
    return run_backtest(df, compute_frequency_scores, config)


class TestRealMetrics:
    """Integration tests for metrics computed on real dataset."""

    def test_ndcg_in_valid_range(self, real_backtest_result) -> None:
        """nDCG@20 should be in [0, 1] for all folds."""
        fold_metrics, summary = compute_backtest_metrics(real_backtest_result)
        for fm in fold_metrics:
            assert 0.0 <= fm.ndcg_20 <= 1.0
        assert 0.0 <= summary.ndcg_20_mean <= 1.0

    def test_weighted_recall_in_valid_range(self, real_backtest_result) -> None:
        """Weighted Recall@20 should be in [0, 1] for all folds."""
        fold_metrics, summary = compute_backtest_metrics(real_backtest_result)
        for fm in fold_metrics:
            assert 0.0 <= fm.weighted_recall_20 <= 1.0
        assert 0.0 <= summary.weighted_recall_20_mean <= 1.0

    def test_brier_in_valid_range(self, real_backtest_result) -> None:
        """Brier score should be in [0, 1] for all folds."""
        fold_metrics, summary = compute_backtest_metrics(real_backtest_result)
        for fm in fold_metrics:
            assert 0.0 <= fm.brier_score <= 1.0
        assert 0.0 <= summary.brier_score_mean <= 1.0

    def test_metrics_are_plausible(self, real_backtest_result) -> None:
        """Primary metrics should be non-trivial on real data."""
        _, summary = compute_backtest_metrics(real_backtest_result)
        # Frequency baseline should achieve reasonable nDCG and WR
        assert summary.ndcg_20_mean > 0.1
        assert summary.weighted_recall_20_mean > 0.1
        # Brier should be < 0.5 (better than random)
        assert summary.brier_score_mean < 0.5

    def test_metric_report_artifacts_exist(self, real_backtest_result, tmp_path: Path) -> None:
        """Metric report artifacts should be written."""
        fold_metrics, summary = compute_backtest_metrics(real_backtest_result)
        out_dir = tmp_path / "reports"
        paths = write_metric_report(
            fold_metrics, summary, real_backtest_result.provenance, out_dir
        )
        assert len(paths) == 2
        for p in paths:
            assert Path(p).exists()

    def test_metric_report_json_parseable(self, real_backtest_result, tmp_path: Path) -> None:
        """Metric report JSON should be valid."""
        fold_metrics, summary = compute_backtest_metrics(real_backtest_result)
        out_dir = tmp_path / "reports"
        write_metric_report(fold_metrics, summary, real_backtest_result.provenance, out_dir)
        data = json.loads((out_dir / "metric_report.json").read_text())
        assert "summary" in data
        assert "primary" in data["summary"]
        assert "secondary" in data["summary"]

    def test_metric_report_markdown_has_labels(self, real_backtest_result, tmp_path: Path) -> None:
        """Metric report Markdown should have primary/secondary sections."""
        fold_metrics, summary = compute_backtest_metrics(real_backtest_result)
        out_dir = tmp_path / "reports"
        write_metric_report(fold_metrics, summary, real_backtest_result.provenance, out_dir)
        content = (out_dir / "metric_report.md").read_text()
        assert "Primary Metrics" in content
        assert "Secondary Metrics" in content
        assert "nDCG@20" in content
        assert "Weighted Recall@20" in content

    def test_deterministic_metrics(self, real_backtest_result) -> None:
        """Running metrics twice on same result should produce identical values."""
        fm1, s1 = compute_backtest_metrics(real_backtest_result)
        fm2, s2 = compute_backtest_metrics(real_backtest_result)
        for a, b in zip(fm1, fm2, strict=True):
            assert a.ndcg_20 == b.ndcg_20
            assert a.weighted_recall_20 == b.weighted_recall_20
            assert a.brier_score == b.brier_score
        assert s1.ndcg_20_mean == s2.ndcg_20_mean
