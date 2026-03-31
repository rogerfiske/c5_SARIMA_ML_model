"""Artifact writers for backtest results.

Produces three output formats:
- JSON: full machine-readable results (all folds + provenance)
- CSV: one row per fold with key metrics
- Markdown: human-readable summary report
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import structlog

from c5_forecasting.evaluation.backtest import BacktestResult
from c5_forecasting.evaluation.metrics import FoldMetrics, MetricSummary

logger = structlog.get_logger(__name__)


def write_backtest_artifacts(
    result: BacktestResult,
    output_dir: Path,
    fold_metrics: list[FoldMetrics] | None = None,
    metric_summary: MetricSummary | None = None,
) -> list[str]:
    """Write backtest result artifacts to disk.

    Produces:
    - backtest_results.json: Full machine-readable results
    - backtest_summary.csv: One row per fold with key metrics
    - backtest_summary.md: Human-readable Markdown report

    Args:
        result: The completed BacktestResult.
        output_dir: Directory to write into (created if needed).
        fold_metrics: Optional per-fold metrics to include in CSV.
        metric_summary: Optional aggregate metrics to include in Markdown.

    Returns:
        List of artifact file paths as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = _write_json_artifact(result, output_dir)
    csv_path = _write_csv_artifact(result, output_dir, fold_metrics)
    md_path = _write_markdown_artifact(result, output_dir, metric_summary)

    paths = [str(json_path), str(csv_path), str(md_path)]
    return paths


def _write_json_artifact(result: BacktestResult, output_dir: Path) -> Path:
    """Write full backtest results as JSON."""
    path = output_dir / "backtest_results.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("backtest_artifact_written", format="json", path=str(path))
    return path


def _write_csv_artifact(
    result: BacktestResult,
    output_dir: Path,
    fold_metrics: list[FoldMetrics] | None = None,
) -> Path:
    """Write a CSV with one row per fold."""
    path = output_dir / "backtest_summary.csv"

    # Build a lookup for metrics by fold_index
    metrics_by_fold: dict[int, FoldMetrics] = {}
    if fold_metrics:
        for fm_entry in fold_metrics:
            metrics_by_fold[fm_entry.fold_index] = fm_entry

    rows = []
    for fold in result.folds:
        row: dict[str, object] = {
            "fold_index": fold.fold_index,
            "cutoff_date": fold.cutoff_date,
            "target_date": fold.target_date,
            "train_rows": fold.train_rows,
            "hit_count": fold.hit_count,
            "actual_row_total": fold.actual_row_total,
            "actual_active_count": len(fold.actual_active_parts),
            "predicted_top_1": (
                fold.predicted_ranking[0]["part_id"] if fold.predicted_ranking else None
            ),
        }
        fold_fm = metrics_by_fold.get(fold.fold_index)
        if fold_fm is not None:
            row["ndcg_20"] = round(fold_fm.ndcg_20, 6)
            row["weighted_recall_20"] = round(fold_fm.weighted_recall_20, 6)
            row["precision_20"] = round(fold_fm.precision_20, 6)
            row["recall_20"] = round(fold_fm.recall_20, 6)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("backtest_artifact_written", format="csv", path=str(path))
    return path


def _write_markdown_artifact(
    result: BacktestResult,
    output_dir: Path,
    metric_summary: MetricSummary | None = None,
) -> Path:
    """Write a human-readable Markdown summary."""
    path = output_dir / "backtest_summary.md"
    prov = result.provenance
    s = result.summary
    lines = [
        f"# Backtest: {prov.model_name}",
        "",
        f"**Run ID:** {prov.run_id}",
        f"**Timestamp:** {prov.run_timestamp}",
        f"**Dataset variant:** {prov.dataset_variant}",
        f"**Dataset fingerprint:** {prov.dataset_fingerprint[:16]}..."
        if len(prov.dataset_fingerprint) > 16
        else f"**Dataset fingerprint:** {prov.dataset_fingerprint}",
        f"**Dataset rows:** {prov.dataset_row_count}",
        f"**Date range:** {prov.dataset_date_min} to {prov.dataset_date_max}",
        "",
        "## Configuration",
        "",
        f"- min_train_rows: {prov.config.get('min_train_rows', 'N/A')}",
        f"- step: {prov.config.get('step', 'N/A')}",
        f"- max_windows: {prov.config.get('max_windows', 'N/A')}",
        f"- k: {prov.config.get('k', 'N/A')}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total folds | {s.total_folds} |",
        f"| Mean hit count | {s.mean_hit_count:.2f} |",
        f"| Min hit count | {s.min_hit_count} |",
        f"| Max hit count | {s.max_hit_count} |",
        f"| Mean actual row total | {s.mean_actual_row_total:.2f} |",
        f"| Cutoff range | {s.first_cutoff_date} to {s.last_cutoff_date} |",
        f"| Target range | {s.first_target_date} to {s.last_target_date} |",
        "",
    ]

    # Metrics section (if computed)
    if metric_summary is not None:
        lines.extend(
            [
                "## Primary Metrics",
                "",
                "| Metric | Mean | Min | Max |",
                "|--------|------|-----|-----|",
                f"| nDCG@20 | {metric_summary.ndcg_20_mean:.4f} "
                f"| {metric_summary.ndcg_20_min:.4f} "
                f"| {metric_summary.ndcg_20_max:.4f} |",
                f"| Weighted Recall@20 | {metric_summary.weighted_recall_20_mean:.4f} "
                f"| {metric_summary.weighted_recall_20_min:.4f} "
                f"| {metric_summary.weighted_recall_20_max:.4f} |",
                f"| Brier Score | {metric_summary.brier_score_mean:.4f} "
                f"| {metric_summary.brier_score_min:.4f} "
                f"| {metric_summary.brier_score_max:.4f} |",
                "",
                "## Secondary Metrics",
                "",
                "| Metric | Mean |",
                "|--------|------|",
                f"| Precision@20 | {metric_summary.precision_20_mean:.4f} |",
                f"| Recall@20 | {metric_summary.recall_20_mean:.4f} |",
                f"| Jaccard@20 | {metric_summary.jaccard_20_mean:.4f} |",
                "",
            ]
        )

    # First 5 folds
    if result.folds:
        lines.append("## First 5 Folds")
        lines.append("")
        lines.append("| Fold | Cutoff | Target | Hits | Actual Total |")
        lines.append("|------|--------|--------|------|--------------|")
        for fold in result.folds[:5]:
            lines.append(
                f"| {fold.fold_index} | {fold.cutoff_date} "
                f"| {fold.target_date} | {fold.hit_count} "
                f"| {fold.actual_row_total} |"
            )
        lines.append("")

    # Last 5 folds
    if len(result.folds) > 5:
        lines.append("## Last 5 Folds")
        lines.append("")
        lines.append("| Fold | Cutoff | Target | Hits | Actual Total |")
        lines.append("|------|--------|--------|------|--------------|")
        for fold in result.folds[-5:]:
            lines.append(
                f"| {fold.fold_index} | {fold.cutoff_date} "
                f"| {fold.target_date} | {fold.hit_count} "
                f"| {fold.actual_row_total} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("backtest_artifact_written", format="markdown", path=str(path))
    return path
