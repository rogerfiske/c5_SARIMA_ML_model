"""Metric report writers for backtest evaluation results.

Produces two output formats:
- JSON: full per-fold metrics + summary with primary/secondary labels
- Markdown: human-readable summary report with primary/secondary sections
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from c5_forecasting.evaluation.backtest import BacktestProvenance
from c5_forecasting.evaluation.metrics import FoldMetrics, MetricSummary

logger = structlog.get_logger(__name__)


def write_metric_report(
    fold_metrics: list[FoldMetrics],
    summary: MetricSummary,
    provenance: BacktestProvenance,
    output_dir: Path,
) -> list[str]:
    """Write metric report artifacts to disk.

    Produces:
    - metric_report.json: Full per-fold metrics + summary + provenance
    - metric_report.md: Human-readable report with primary/secondary sections

    Args:
        fold_metrics: Per-fold metric results.
        summary: Aggregate MetricSummary.
        provenance: Backtest provenance metadata.
        output_dir: Directory to write into (created if needed).

    Returns:
        List of artifact file paths as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = _write_json_report(fold_metrics, summary, provenance, output_dir)
    md_path = _write_markdown_report(fold_metrics, summary, provenance, output_dir)

    paths = [str(json_path), str(md_path)]
    return paths


def _write_json_report(
    fold_metrics: list[FoldMetrics],
    summary: MetricSummary,
    provenance: BacktestProvenance,
    output_dir: Path,
) -> Path:
    """Write full metric results as JSON."""
    path = output_dir / "metric_report.json"
    data: dict[str, Any] = {
        "provenance": provenance.to_dict(),
        "summary": summary.to_dict(),
        "folds": [fm.to_dict() for fm in fold_metrics],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("metric_report_written", format="json", path=str(path))
    return path


def _write_markdown_report(
    fold_metrics: list[FoldMetrics],
    summary: MetricSummary,
    provenance: BacktestProvenance,
    output_dir: Path,
) -> Path:
    """Write a human-readable Markdown metric report."""
    path = output_dir / "metric_report.md"
    lines = [
        f"# Metric Report: {provenance.model_name}",
        "",
        f"**Run ID:** {provenance.run_id}",
        f"**Timestamp:** {provenance.run_timestamp}",
        f"**Dataset variant:** {provenance.dataset_variant}",
        f"**Dataset rows:** {provenance.dataset_row_count}",
        f"**Total folds:** {summary.total_folds}",
        "",
        "## Primary Metrics",
        "",
        "| Metric | Mean | Min | Max |",
        "|--------|------|-----|-----|",
        f"| nDCG@20 | {summary.ndcg_20_mean:.4f} "
        f"| {summary.ndcg_20_min:.4f} | {summary.ndcg_20_max:.4f} |",
        f"| Weighted Recall@20 | {summary.weighted_recall_20_mean:.4f} "
        f"| {summary.weighted_recall_20_min:.4f} "
        f"| {summary.weighted_recall_20_max:.4f} |",
        f"| Brier Score | {summary.brier_score_mean:.4f} "
        f"| {summary.brier_score_min:.4f} | {summary.brier_score_max:.4f} |",
        "",
        "## Secondary Metrics",
        "",
        "| Metric | Mean |",
        "|--------|------|",
        f"| Precision@20 | {summary.precision_20_mean:.4f} |",
        f"| Recall@20 | {summary.recall_20_mean:.4f} |",
        f"| Jaccard@20 | {summary.jaccard_20_mean:.4f} |",
        "",
    ]

    # First 5 folds detail
    if fold_metrics:
        lines.append("## First 5 Folds")
        lines.append("")
        lines.append("| Fold | nDCG@20 | WR@20 | Brier | P@20 | R@20 | Jaccard |")
        lines.append("|------|---------|-------|-------|------|------|---------|")
        for fm in fold_metrics[:5]:
            lines.append(
                f"| {fm.fold_index} "
                f"| {fm.ndcg_20:.4f} "
                f"| {fm.weighted_recall_20:.4f} "
                f"| {fm.brier_score:.4f} "
                f"| {fm.precision_20:.4f} "
                f"| {fm.recall_20:.4f} "
                f"| {fm.jaccard_20:.4f} |"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("metric_report_written", format="markdown", path=str(path))
    return path
