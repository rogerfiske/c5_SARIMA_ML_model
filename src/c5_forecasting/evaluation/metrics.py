"""Ranking and calibration metrics for backtest evaluation.

Implements primary metrics (nDCG@20, Weighted Recall@20, Brier Score) and
secondary metrics (Precision@20, Recall@20, Jaccard@20) as pure functions.
Each metric function is independently testable; the orchestrator functions
compute all metrics per fold and aggregate across folds.

Primary metrics use count-weighted relevance (actual part counts), not
binary presence. Secondary metrics use binary presence only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import structlog

from c5_forecasting.evaluation.backtest import BacktestFold, BacktestResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-fold metric container
# ---------------------------------------------------------------------------
@dataclass
class FoldMetrics:
    """Metrics computed for a single backtest fold."""

    fold_index: int
    # Primary
    ndcg_20: float
    weighted_recall_20: float
    brier_score: float
    # Secondary
    precision_20: float
    recall_20: float
    jaccard_20: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "ndcg_20": round(self.ndcg_20, 6),
            "weighted_recall_20": round(self.weighted_recall_20, 6),
            "brier_score": round(self.brier_score, 6),
            "precision_20": round(self.precision_20, 6),
            "recall_20": round(self.recall_20, 6),
            "jaccard_20": round(self.jaccard_20, 6),
        }


# ---------------------------------------------------------------------------
# Aggregate metric summary
# ---------------------------------------------------------------------------
@dataclass
class MetricSummary:
    """Aggregate metrics across all backtest folds."""

    total_folds: int
    # Primary — mean, min, max
    ndcg_20_mean: float
    ndcg_20_min: float
    ndcg_20_max: float
    weighted_recall_20_mean: float
    weighted_recall_20_min: float
    weighted_recall_20_max: float
    brier_score_mean: float
    brier_score_min: float
    brier_score_max: float
    # Secondary — mean only
    precision_20_mean: float
    recall_20_mean: float
    jaccard_20_mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_folds": self.total_folds,
            "primary": {
                "ndcg_20": {
                    "mean": round(self.ndcg_20_mean, 6),
                    "min": round(self.ndcg_20_min, 6),
                    "max": round(self.ndcg_20_max, 6),
                },
                "weighted_recall_20": {
                    "mean": round(self.weighted_recall_20_mean, 6),
                    "min": round(self.weighted_recall_20_min, 6),
                    "max": round(self.weighted_recall_20_max, 6),
                },
                "brier_score": {
                    "mean": round(self.brier_score_mean, 6),
                    "min": round(self.brier_score_min, 6),
                    "max": round(self.brier_score_max, 6),
                },
            },
            "secondary": {
                "precision_20": {"mean": round(self.precision_20_mean, 6)},
                "recall_20": {"mean": round(self.recall_20_mean, 6)},
                "jaccard_20": {"mean": round(self.jaccard_20_mean, 6)},
            },
        }


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------
def compute_ndcg(
    predicted_ids: list[int],
    actual_counts: dict[int, int],
    k: int = 20,
) -> float:
    """Compute nDCG@k using actual part counts as relevance weights.

    DCG@k  = Σ(i=1..k) rel(pred_i) / log₂(i+1)
    IDCG@k = DCG with the top-k actual counts sorted descending
    nDCG   = DCG / IDCG  (0 if IDCG = 0)

    Args:
        predicted_ids: Ordered list of predicted part IDs (rank order).
        actual_counts: Mapping of part_id → count for active parts.
        k: Cutoff rank (default 20).

    Returns:
        nDCG value in [0, 1].
    """
    if not actual_counts:
        return 0.0

    # DCG from predicted ranking
    dcg = 0.0
    for i, pid in enumerate(predicted_ids[:k]):
        rel = actual_counts.get(pid, 0)
        dcg += rel / math.log2(i + 2)  # i+2 because rank is 1-indexed

    # IDCG from ideal ranking (top-k actual counts, sorted descending)
    ideal_rels = sorted(actual_counts.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def compute_weighted_recall(
    predicted_ids: list[int],
    actual_counts: dict[int, int],
) -> float:
    """Compute Weighted Recall@k using actual counts as weights.

    WR = Σ count(p ∈ predicted ∩ actual) / Σ count(all active p)

    Args:
        predicted_ids: Predicted part IDs.
        actual_counts: Mapping of part_id → count for active parts.

    Returns:
        Weighted recall value in [0, 1].
    """
    total_weight = sum(actual_counts.values())
    if total_weight == 0:
        return 0.0

    predicted_set = set(predicted_ids)
    hit_weight = sum(count for pid, count in actual_counts.items() if pid in predicted_set)

    return hit_weight / total_weight


def compute_brier_score(
    all_scores: list[dict[str, Any]],
    actual_active: set[int],
    n_parts: int = 39,
) -> float:
    """Compute Brier score across all parts.

    Brier = (1/N) Σ(p=1..N) (score(p) - indicator(p active))²

    Only meaningful when scores represent calibrated probabilities.
    Lower is better (0 = perfect, 1 = worst).

    Args:
        all_scores: List of {part_id, score} dicts for all parts.
        actual_active: Set of part IDs that were active on the target date.
        n_parts: Total number of parts in the universe (default 39).

    Returns:
        Brier score in [0, 1].
    """
    if not all_scores:
        return 1.0  # worst case if no scores available

    total = 0.0
    for entry in all_scores:
        pid = entry["part_id"]
        score = entry["score"]
        indicator = 1.0 if pid in actual_active else 0.0
        total += (score - indicator) ** 2

    return total / n_parts


def compute_precision(hit_count: int, k: int = 20) -> float:
    """Compute binary Precision@k.

    P@k = |predicted ∩ actual| / k
    """
    if k == 0:
        return 0.0
    return hit_count / k


def compute_recall(hit_count: int, actual_count: int) -> float:
    """Compute binary Recall@k.

    R@k = |predicted ∩ actual| / |actual|
    """
    if actual_count == 0:
        return 0.0
    return hit_count / actual_count


def compute_jaccard(predicted_ids: set[int], actual_ids: set[int]) -> float:
    """Compute Jaccard similarity between predicted and actual sets.

    J = |predicted ∩ actual| / |predicted ∪ actual|
    """
    union = predicted_ids | actual_ids
    if not union:
        return 0.0
    return len(predicted_ids & actual_ids) / len(union)


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------
def compute_fold_metrics(fold: BacktestFold) -> FoldMetrics:
    """Compute all metrics for a single backtest fold.

    Args:
        fold: A BacktestFold with predicted ranking, actual parts, and scores.

    Returns:
        FoldMetrics with all primary and secondary metrics.
    """
    predicted_ids = [r["part_id"] for r in fold.predicted_ranking]
    predicted_set = set(predicted_ids)
    actual_set = set(fold.actual_active_parts)
    k = len(fold.predicted_ranking) if fold.predicted_ranking else 20

    return FoldMetrics(
        fold_index=fold.fold_index,
        ndcg_20=compute_ndcg(predicted_ids, fold.actual_part_counts, k=k),
        weighted_recall_20=compute_weighted_recall(predicted_ids, fold.actual_part_counts),
        brier_score=compute_brier_score(fold.all_scores, actual_set),
        precision_20=compute_precision(fold.hit_count, k=k),
        recall_20=compute_recall(fold.hit_count, len(fold.actual_active_parts)),
        jaccard_20=compute_jaccard(predicted_set, actual_set),
    )


def compute_backtest_metrics(
    result: BacktestResult,
) -> tuple[list[FoldMetrics], MetricSummary]:
    """Compute metrics for all folds and aggregate into a summary.

    Args:
        result: Complete BacktestResult from run_backtest.

    Returns:
        Tuple of (per-fold metrics list, aggregate MetricSummary).
    """
    fold_metrics = [compute_fold_metrics(fold) for fold in result.folds]

    ndcg_vals = [m.ndcg_20 for m in fold_metrics]
    wr_vals = [m.weighted_recall_20 for m in fold_metrics]
    brier_vals = [m.brier_score for m in fold_metrics]
    prec_vals = [m.precision_20 for m in fold_metrics]
    rec_vals = [m.recall_20 for m in fold_metrics]
    jacc_vals = [m.jaccard_20 for m in fold_metrics]

    n = len(fold_metrics)

    summary = MetricSummary(
        total_folds=n,
        ndcg_20_mean=sum(ndcg_vals) / n,
        ndcg_20_min=min(ndcg_vals),
        ndcg_20_max=max(ndcg_vals),
        weighted_recall_20_mean=sum(wr_vals) / n,
        weighted_recall_20_min=min(wr_vals),
        weighted_recall_20_max=max(wr_vals),
        brier_score_mean=sum(brier_vals) / n,
        brier_score_min=min(brier_vals),
        brier_score_max=max(brier_vals),
        precision_20_mean=sum(prec_vals) / n,
        recall_20_mean=sum(rec_vals) / n,
        jaccard_20_mean=sum(jacc_vals) / n,
    )

    logger.info(
        "backtest_metrics_computed",
        total_folds=n,
        ndcg_20_mean=round(summary.ndcg_20_mean, 4),
        weighted_recall_20_mean=round(summary.weighted_recall_20_mean, 4),
        brier_score_mean=round(summary.brier_score_mean, 4),
    )

    return fold_metrics, summary
