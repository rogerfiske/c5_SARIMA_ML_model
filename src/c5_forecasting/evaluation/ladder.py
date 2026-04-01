"""Baseline ladder — run all registered models through backtest and compare.

Orchestrates multiple backtest runs through the same engine with the same
config, collects MetricSummary for each, and writes comparison artifacts
(JSON, CSV, Markdown) ranking models by primary metrics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from c5_forecasting.evaluation.backtest import BacktestConfig, BacktestResult, run_backtest
from c5_forecasting.evaluation.metrics import MetricSummary, compute_backtest_metrics
from c5_forecasting.models.registry import get_model_registry

logger = structlog.get_logger(__name__)


@dataclass
class LadderEntry:
    """Results for a single model in the ladder."""

    model_name: str
    metric_summary: MetricSummary
    backtest_result: BacktestResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metrics": self.metric_summary.to_dict(),
            "provenance": self.backtest_result.provenance.to_dict(),
        }


@dataclass
class LadderResult:
    """Complete result of a baseline ladder run."""

    entries: list[LadderEntry]
    config: BacktestConfig
    best_model: str
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_model": self.best_model,
            "config": self.config.to_dict(),
            "models": [e.to_dict() for e in self.entries],
        }


def run_ladder(
    df: pd.DataFrame,
    config: BacktestConfig,
    dataset_variant: str = "raw",
    dataset_fingerprint: str = "",
    source_fingerprint: str = "",
    model_names: list[str] | None = None,
) -> LadderResult:
    """Run all registered baselines through the backtest engine.

    Args:
        df: Working dataset DataFrame.
        config: BacktestConfig (shared across all models).
        dataset_variant: For provenance.
        dataset_fingerprint: For provenance.
        source_fingerprint: For provenance.
        model_names: Optional subset of model names to run. None = all.

    Returns:
        LadderResult with entries sorted by nDCG@20 descending.
    """
    registry = get_model_registry()

    if model_names is not None:
        for name in model_names:
            if name not in registry:
                available = ", ".join(sorted(registry.keys()))
                raise KeyError(f"Unknown model {name!r}. Available: {available}")
        selected = {name: registry[name] for name in model_names}
    else:
        selected = registry

    entries: list[LadderEntry] = []

    for model_name, scoring_fn in sorted(selected.items()):
        logger.info("ladder_running_model", model=model_name)

        model_config = BacktestConfig(
            min_train_rows=config.min_train_rows,
            step=config.step,
            max_windows=config.max_windows,
            k=config.k,
            model_name=model_name,
        )

        result = run_backtest(
            df=df,
            scoring_fn=scoring_fn,
            config=model_config,
            dataset_variant=dataset_variant,
            dataset_fingerprint=dataset_fingerprint,
            source_fingerprint=source_fingerprint,
        )

        _, metric_summary = compute_backtest_metrics(result)

        entries.append(
            LadderEntry(
                model_name=model_name,
                metric_summary=metric_summary,
                backtest_result=result,
            )
        )

    # Sort by nDCG@20 descending (primary metric)
    entries.sort(key=lambda e: e.metric_summary.ndcg_20_mean, reverse=True)

    best_model = entries[0].model_name if entries else ""

    logger.info(
        "ladder_complete",
        models_run=len(entries),
        best_model=best_model,
    )

    return LadderResult(
        entries=entries,
        config=config,
        best_model=best_model,
    )


def write_ladder_artifacts(
    ladder_result: LadderResult,
    output_dir: Path,
) -> list[str]:
    """Write comparison artifacts for the ladder run.

    Produces:
    - comparison_results.json: Full machine-readable comparison
    - comparison_summary.csv: Tabular metrics per model
    - comparison_summary.md: Human-readable ranked table

    Args:
        ladder_result: Complete LadderResult.
        output_dir: Directory to write into (created if needed).

    Returns:
        List of artifact file paths as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = _write_json_comparison(ladder_result, output_dir)
    csv_path = _write_csv_comparison(ladder_result, output_dir)
    md_path = _write_markdown_comparison(ladder_result, output_dir)

    paths = [str(json_path), str(csv_path), str(md_path)]
    logger.info("ladder_artifacts_written", paths=paths)
    return paths


def _write_json_comparison(result: LadderResult, output_dir: Path) -> Path:
    path = output_dir / "comparison_results.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def _write_csv_comparison(result: LadderResult, output_dir: Path) -> Path:
    path = output_dir / "comparison_summary.csv"
    rows = []
    for i, entry in enumerate(result.entries):
        s = entry.metric_summary
        rows.append(
            {
                "rank": i + 1,
                "model": entry.model_name,
                "ndcg_20_mean": round(s.ndcg_20_mean, 6),
                "weighted_recall_20_mean": round(s.weighted_recall_20_mean, 6),
                "brier_score_mean": round(s.brier_score_mean, 6),
                "precision_20_mean": round(s.precision_20_mean, 6),
                "recall_20_mean": round(s.recall_20_mean, 6),
                "jaccard_20_mean": round(s.jaccard_20_mean, 6),
                "total_folds": s.total_folds,
            }
        )
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(path, index=False)
    return path


def _write_markdown_comparison(result: LadderResult, output_dir: Path) -> Path:
    path = output_dir / "comparison_summary.md"
    lines = [
        "# Baseline Ladder Comparison",
        "",
        f"**Best model:** {result.best_model}",
        f"**Models evaluated:** {len(result.entries)}",
        "",
        "## Configuration",
        "",
        f"- min_train_rows: {result.config.min_train_rows}",
        f"- step: {result.config.step}",
        f"- max_windows: {result.config.max_windows}",
        f"- k: {result.config.k}",
        "",
        "## Results (ranked by nDCG@20)",
        "",
        "| Rank | Model | nDCG@20 | WR@20 | Brier | P@20 | R@20 | Jaccard |",
        "|------|-------|---------|-------|-------|------|------|---------|",
    ]
    for i, entry in enumerate(result.entries):
        s = entry.metric_summary
        lines.append(
            f"| {i + 1} "
            f"| {entry.model_name} "
            f"| {s.ndcg_20_mean:.4f} "
            f"| {s.weighted_recall_20_mean:.4f} "
            f"| {s.brier_score_mean:.4f} "
            f"| {s.precision_20_mean:.4f} "
            f"| {s.recall_20_mean:.4f} "
            f"| {s.jaccard_20_mean:.4f} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- **nDCG@20** (primary): count-weighted ranking quality (higher is better)")
    lines.append(
        "- **WR@20**: weighted recall capturing coverage of high-count parts (higher is better)"
    )
    lines.append("- **Brier**: calibration quality (lower is better)")
    lines.append("- **P@20/R@20/Jaccard**: binary overlap metrics (higher is better)")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
