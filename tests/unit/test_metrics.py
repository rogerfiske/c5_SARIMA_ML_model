"""Unit tests for ranking and calibration metrics."""

from __future__ import annotations

from c5_forecasting.evaluation.backtest import BacktestFold, BacktestResult
from c5_forecasting.evaluation.metrics import (
    FoldMetrics,
    MetricSummary,
    compute_backtest_metrics,
    compute_brier_score,
    compute_fold_metrics,
    compute_jaccard,
    compute_ndcg,
    compute_precision,
    compute_recall,
    compute_weighted_recall,
)


# ---------------------------------------------------------------------------
# nDCG tests
# ---------------------------------------------------------------------------
class TestComputeNDCG:
    """Tests for compute_ndcg."""

    def test_perfect_prediction(self) -> None:
        """Predicting the top-k parts in ideal order should give nDCG=1.0."""
        actual_counts = {1: 10, 2: 8, 3: 5}
        predicted_ids = [1, 2, 3]
        assert compute_ndcg(predicted_ids, actual_counts, k=3) == 1.0

    def test_empty_actuals_returns_zero(self) -> None:
        """No actual active parts should give nDCG=0."""
        assert compute_ndcg([1, 2, 3], {}, k=3) == 0.0

    def test_no_overlap_returns_zero(self) -> None:
        """Predicting parts not in actuals should give nDCG=0."""
        actual_counts = {1: 5, 2: 3}
        predicted_ids = [10, 11, 12]
        assert compute_ndcg(predicted_ids, actual_counts, k=3) == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap should give nDCG between 0 and 1."""
        actual_counts = {1: 10, 2: 8, 3: 5, 4: 3}
        predicted_ids = [1, 10, 3, 11]  # hits on rank 1 and 3
        result = compute_ndcg(predicted_ids, actual_counts, k=4)
        assert 0 < result < 1.0

    def test_order_matters(self) -> None:
        """Putting higher-count parts at higher ranks should give better nDCG."""
        actual_counts = {1: 10, 2: 1}
        good_order = [1, 2]  # high-count first
        bad_order = [2, 1]  # low-count first
        ndcg_good = compute_ndcg(good_order, actual_counts, k=2)
        ndcg_bad = compute_ndcg(bad_order, actual_counts, k=2)
        assert ndcg_good > ndcg_bad

    def test_count_weighting_matters(self) -> None:
        """Higher counts should produce higher DCG than binary presence."""
        counts_high = {1: 100, 2: 50}
        counts_low = {1: 1, 2: 1}
        predicted = [1, 2]
        # Both should give nDCG=1.0 (perfect order), but with different DCG
        assert compute_ndcg(predicted, counts_high, k=2) == 1.0
        assert compute_ndcg(predicted, counts_low, k=2) == 1.0

    def test_result_bounded_0_to_1(self) -> None:
        """nDCG should always be in [0, 1]."""
        actual_counts = {i: i for i in range(1, 21)}
        predicted_ids = list(range(5, 25))  # partial overlap
        result = compute_ndcg(predicted_ids, actual_counts, k=20)
        assert 0.0 <= result <= 1.0

    def test_single_part_at_rank_1(self) -> None:
        """A single correct part at rank 1 should give nDCG=1.0."""
        assert compute_ndcg([5], {5: 3}, k=1) == 1.0


# ---------------------------------------------------------------------------
# Weighted Recall tests
# ---------------------------------------------------------------------------
class TestComputeWeightedRecall:
    """Tests for compute_weighted_recall."""

    def test_perfect_recall(self) -> None:
        """Predicting all active parts should give WR=1.0."""
        actual_counts = {1: 5, 2: 3, 3: 2}
        predicted_ids = [1, 2, 3, 4, 5]
        assert compute_weighted_recall(predicted_ids, actual_counts) == 1.0

    def test_no_overlap_returns_zero(self) -> None:
        """No overlap should give WR=0."""
        actual_counts = {1: 5, 2: 3}
        predicted_ids = [10, 11, 12]
        assert compute_weighted_recall(predicted_ids, actual_counts) == 0.0

    def test_empty_actuals_returns_zero(self) -> None:
        """No actual active parts should give WR=0."""
        assert compute_weighted_recall([1, 2, 3], {}) == 0.0

    def test_count_weighting(self) -> None:
        """Hitting high-count parts should give higher WR than low-count."""
        actual_counts = {1: 10, 2: 1}
        # Hit part 1 (count=10): WR = 10/11
        wr_high = compute_weighted_recall([1], actual_counts)
        # Hit part 2 (count=1): WR = 1/11
        wr_low = compute_weighted_recall([2], actual_counts)
        assert wr_high > wr_low
        assert abs(wr_high - 10 / 11) < 1e-9
        assert abs(wr_low - 1 / 11) < 1e-9

    def test_partial_overlap(self) -> None:
        """Partial overlap should give WR between 0 and 1."""
        actual_counts = {1: 5, 2: 5, 3: 5, 4: 5}
        predicted_ids = [1, 2, 10, 11]  # 2 of 4 active parts
        wr = compute_weighted_recall(predicted_ids, actual_counts)
        assert abs(wr - 0.5) < 1e-9  # 10/20

    def test_result_bounded_0_to_1(self) -> None:
        """WR should always be in [0, 1]."""
        actual_counts = {i: i for i in range(1, 21)}
        predicted_ids = list(range(5, 25))
        result = compute_weighted_recall(predicted_ids, actual_counts)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Brier Score tests
# ---------------------------------------------------------------------------
class TestComputeBrierScore:
    """Tests for compute_brier_score."""

    def test_perfect_calibration(self) -> None:
        """Score=1.0 for active parts, 0.0 for inactive → Brier=0."""
        active = {1, 2, 3}
        all_scores = [{"part_id": i, "score": 1.0 if i in active else 0.0} for i in range(1, 40)]
        assert compute_brier_score(all_scores, active) == 0.0

    def test_worst_calibration(self) -> None:
        """Score=0.0 for active parts, 1.0 for inactive → Brier=1."""
        active = {1, 2, 3}
        all_scores = [{"part_id": i, "score": 0.0 if i in active else 1.0} for i in range(1, 40)]
        assert compute_brier_score(all_scores, active) == 1.0

    def test_uniform_scores(self) -> None:
        """Uniform 0.5 scores should give Brier=0.25."""
        active = {1, 2, 3}
        all_scores = [{"part_id": i, "score": 0.5} for i in range(1, 40)]
        # (0.5-1)^2 * 3 + (0.5-0)^2 * 36 = 0.25*3 + 0.25*36 = 0.25*39
        brier = compute_brier_score(all_scores, active)
        assert abs(brier - 0.25) < 1e-9

    def test_empty_scores_returns_worst(self) -> None:
        """No scores available should return worst-case 1.0."""
        assert compute_brier_score([], {1, 2}, n_parts=39) == 1.0

    def test_result_bounded_0_to_1(self) -> None:
        """Brier should always be in [0, 1]."""
        active = {1, 5, 10, 20, 30}
        all_scores = [{"part_id": i, "score": 0.7} for i in range(1, 40)]
        result = compute_brier_score(all_scores, active)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------
class TestComputePrecision:
    """Tests for compute_precision."""

    def test_perfect(self) -> None:
        """All predictions hit → P@20=1.0."""
        assert compute_precision(20, k=20) == 1.0

    def test_zero(self) -> None:
        """No hits → P@20=0."""
        assert compute_precision(0, k=20) == 0.0

    def test_partial(self) -> None:
        """10 hits out of 20 → P@20=0.5."""
        assert compute_precision(10, k=20) == 0.5

    def test_k_zero_returns_zero(self) -> None:
        """k=0 should return 0 to avoid division by zero."""
        assert compute_precision(5, k=0) == 0.0


# ---------------------------------------------------------------------------
# Recall tests
# ---------------------------------------------------------------------------
class TestComputeRecall:
    """Tests for compute_recall."""

    def test_perfect(self) -> None:
        """All actuals predicted → R=1.0."""
        assert compute_recall(10, 10) == 1.0

    def test_zero(self) -> None:
        """No hits → R=0."""
        assert compute_recall(0, 10) == 0.0

    def test_partial(self) -> None:
        """5 of 10 actuals predicted → R=0.5."""
        assert compute_recall(5, 10) == 0.5

    def test_no_actuals_returns_zero(self) -> None:
        """No actual active parts should return 0."""
        assert compute_recall(0, 0) == 0.0


# ---------------------------------------------------------------------------
# Jaccard tests
# ---------------------------------------------------------------------------
class TestComputeJaccard:
    """Tests for compute_jaccard."""

    def test_perfect_overlap(self) -> None:
        """Identical sets → J=1.0."""
        s = {1, 2, 3}
        assert compute_jaccard(s, s) == 1.0

    def test_no_overlap(self) -> None:
        """Disjoint sets → J=0."""
        assert compute_jaccard({1, 2}, {3, 4}) == 0.0

    def test_partial_overlap(self) -> None:
        """Partial overlap → 0 < J < 1."""
        j = compute_jaccard({1, 2, 3}, {2, 3, 4})
        # intersection=2, union=4 → J=0.5
        assert abs(j - 0.5) < 1e-9

    def test_both_empty(self) -> None:
        """Both sets empty → J=0."""
        assert compute_jaccard(set(), set()) == 0.0


# ---------------------------------------------------------------------------
# Fold metrics orchestrator
# ---------------------------------------------------------------------------
class TestComputeFoldMetrics:
    """Tests for compute_fold_metrics orchestrator."""

    def test_returns_fold_metrics(self) -> None:
        """Should return a FoldMetrics instance."""
        fold = BacktestFold(
            fold_index=0,
            cutoff_date="2020-01-10",
            target_date="2020-01-11",
            train_rows=10,
            predicted_ranking=[
                {"rank": 1, "part_id": 1, "score": 0.9},
                {"rank": 2, "part_id": 2, "score": 0.8},
            ],
            actual_active_parts=[1, 3],
            actual_row_total=10,
            hit_count=1,
            hit_parts=[1],
            miss_parts=[2],
            actual_part_counts={1: 5, 3: 5},
            all_scores=[{"part_id": i, "score": 0.5} for i in range(1, 40)],
        )
        result = compute_fold_metrics(fold)
        assert isinstance(result, FoldMetrics)
        assert result.fold_index == 0

    def test_all_metrics_bounded(self) -> None:
        """All metrics should be in valid ranges."""
        fold = BacktestFold(
            fold_index=0,
            cutoff_date="2020-01-10",
            target_date="2020-01-11",
            train_rows=10,
            predicted_ranking=[
                {"rank": i + 1, "part_id": i + 1, "score": 0.9 - i * 0.01} for i in range(20)
            ],
            actual_active_parts=list(range(1, 16)),
            actual_row_total=30,
            hit_count=15,
            hit_parts=list(range(1, 16)),
            miss_parts=list(range(16, 21)),
            actual_part_counts={i: 2 for i in range(1, 16)},
            all_scores=[{"part_id": i, "score": 0.5} for i in range(1, 40)],
        )
        m = compute_fold_metrics(fold)
        assert 0.0 <= m.ndcg_20 <= 1.0
        assert 0.0 <= m.weighted_recall_20 <= 1.0
        assert 0.0 <= m.brier_score <= 1.0
        assert 0.0 <= m.precision_20 <= 1.0
        assert 0.0 <= m.recall_20 <= 1.0
        assert 0.0 <= m.jaccard_20 <= 1.0

    def test_to_dict_keys(self) -> None:
        """FoldMetrics.to_dict should have expected keys."""
        fold = BacktestFold(
            fold_index=0,
            cutoff_date="2020-01-10",
            target_date="2020-01-11",
            train_rows=10,
            predicted_ranking=[{"rank": 1, "part_id": 1, "score": 0.9}],
            actual_active_parts=[1],
            actual_row_total=5,
            hit_count=1,
            hit_parts=[1],
            miss_parts=[],
            actual_part_counts={1: 5},
            all_scores=[{"part_id": i, "score": 0.5} for i in range(1, 40)],
        )
        d = compute_fold_metrics(fold).to_dict()
        expected_keys = {
            "fold_index",
            "ndcg_20",
            "weighted_recall_20",
            "brier_score",
            "precision_20",
            "recall_20",
            "jaccard_20",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Backtest metrics aggregator
# ---------------------------------------------------------------------------
class TestComputeBacktestMetrics:
    """Tests for compute_backtest_metrics aggregator."""

    def _make_result_with_folds(self, n_folds: int = 3) -> BacktestResult:
        """Build a minimal BacktestResult for testing."""
        from c5_forecasting.evaluation.backtest import (
            BacktestProvenance,
            BacktestSummary,
        )

        folds = []
        for i in range(n_folds):
            fold = BacktestFold(
                fold_index=i,
                cutoff_date=f"2020-01-{10 + i}",
                target_date=f"2020-01-{11 + i}",
                train_rows=10 + i,
                predicted_ranking=[
                    {"rank": j + 1, "part_id": j + 1, "score": 0.9 - j * 0.01} for j in range(20)
                ],
                actual_active_parts=list(range(1, 16)),
                actual_row_total=30,
                hit_count=15,
                hit_parts=list(range(1, 16)),
                miss_parts=list(range(16, 21)),
                actual_part_counts={j: 2 for j in range(1, 16)},
                all_scores=[{"part_id": j, "score": 0.5} for j in range(1, 40)],
            )
            folds.append(fold)

        summary = BacktestSummary(
            total_folds=n_folds,
            mean_hit_count=15.0,
            min_hit_count=15,
            max_hit_count=15,
            mean_actual_row_total=30.0,
            first_cutoff_date="2020-01-10",
            last_cutoff_date=f"2020-01-{10 + n_folds - 1}",
            first_target_date="2020-01-11",
            last_target_date=f"2020-01-{11 + n_folds - 1}",
        )
        provenance = BacktestProvenance(
            run_id="test",
            run_timestamp="20200101T000000Z",
            model_name="frequency_baseline",
            dataset_variant="raw",
            dataset_fingerprint="abc",
            source_fingerprint="def",
            config={},
            dataset_row_count=50,
            dataset_date_min="2020-01-01",
            dataset_date_max="2020-02-19",
            total_folds=n_folds,
        )
        return BacktestResult(provenance=provenance, folds=folds, summary=summary)

    def test_returns_fold_metrics_and_summary(self) -> None:
        """Should return (list of FoldMetrics, MetricSummary)."""
        result = self._make_result_with_folds(3)
        fold_metrics, summary = compute_backtest_metrics(result)
        assert len(fold_metrics) == 3
        assert isinstance(summary, MetricSummary)
        assert summary.total_folds == 3

    def test_all_folds_have_metrics(self) -> None:
        """Each fold should produce a FoldMetrics."""
        result = self._make_result_with_folds(5)
        fold_metrics, _ = compute_backtest_metrics(result)
        assert len(fold_metrics) == 5
        for fm in fold_metrics:
            assert isinstance(fm, FoldMetrics)

    def test_summary_mean_matches_fold_values(self) -> None:
        """Summary mean should equal the mean of fold values."""
        result = self._make_result_with_folds(3)
        fold_metrics, summary = compute_backtest_metrics(result)
        ndcg_values = [m.ndcg_20 for m in fold_metrics]
        expected_mean = sum(ndcg_values) / len(ndcg_values)
        assert abs(summary.ndcg_20_mean - expected_mean) < 1e-9

    def test_summary_min_max(self) -> None:
        """Summary min/max should match fold extremes."""
        result = self._make_result_with_folds(3)
        fold_metrics, summary = compute_backtest_metrics(result)
        brier_values = [m.brier_score for m in fold_metrics]
        assert summary.brier_score_min == min(brier_values)
        assert summary.brier_score_max == max(brier_values)

    def test_summary_to_dict_has_primary_secondary(self) -> None:
        """Summary to_dict should have primary and secondary sections."""
        result = self._make_result_with_folds(3)
        _, summary = compute_backtest_metrics(result)
        d = summary.to_dict()
        assert "primary" in d
        assert "secondary" in d
        assert "ndcg_20" in d["primary"]
        assert "weighted_recall_20" in d["primary"]
        assert "brier_score" in d["primary"]
        assert "precision_20" in d["secondary"]
        assert "recall_20" in d["secondary"]
        assert "jaccard_20" in d["secondary"]

    def test_deterministic(self) -> None:
        """Running twice on same data should produce identical metrics."""
        result = self._make_result_with_folds(3)
        fm1, s1 = compute_backtest_metrics(result)
        fm2, s2 = compute_backtest_metrics(result)
        for a, b in zip(fm1, fm2, strict=True):
            assert a.ndcg_20 == b.ndcg_20
            assert a.brier_score == b.brier_score
        assert s1.ndcg_20_mean == s2.ndcg_20_mean
