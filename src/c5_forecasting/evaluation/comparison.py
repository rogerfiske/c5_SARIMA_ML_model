"""Experiment comparison engine — compare ladder results against a champion.

Applies configurable minimum-delta thresholds to determine whether the
best model in a ladder run qualifies as a champion candidate.  Produces
structured comparison reports (JSON + Markdown) with per-model verdicts.

The ranking policy is explicit and deterministic:
  1. nDCG@20 mean  (higher is better)
  2. WR@20 mean    (higher is better)  — tie-break #1
  3. Brier mean    (lower is better)   — tie-break #2
  4. Model name    (alphabetical)      — tie-break #3
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from c5_forecasting.evaluation.champion import ChampionRecord
from c5_forecasting.evaluation.ladder import LadderEntry, LadderResult
from c5_forecasting.evaluation.metrics import MetricSummary

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class ComparisonConfig:
    """Configurable thresholds for champion-candidate gating.

    Only ``min_ndcg_delta`` is a blocking gate in v1.  WR and Brier
    deltas are computed and reported but do not block candidacy.
    """

    min_ndcg_delta: float = 0.01
    min_wr_delta: float = 0.01  # reported only
    max_brier_delta: float = 0.01  # reported only

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_ndcg_delta": self.min_ndcg_delta,
            "min_wr_delta": self.min_wr_delta,
            "max_brier_delta": self.max_brier_delta,
        }


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------
class CandidateVerdict(str, Enum):
    """Verdict for whether a model can be champion candidate."""

    ELIGIBLE = "eligible"
    BLOCKED_BELOW_DELTA = "blocked_below_delta"
    BLOCKED_TIED = "blocked_tied"
    NO_CHAMPION = "no_champion"  # first-ever promotion


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class ComparisonEntry:
    """Per-model entry in a comparison result."""

    model_name: str
    metric_summary: MetricSummary
    verdict: CandidateVerdict
    delta_vs_champion: dict[str, float]
    is_best_in_report: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "metrics": self.metric_summary.to_dict(),
            "verdict": self.verdict.value,
            "delta_vs_champion": {k: round(v, 6) for k, v in self.delta_vs_champion.items()},
            "is_best_in_report": self.is_best_in_report,
        }


@dataclass
class ComparisonResult:
    """Complete result of a champion-candidate comparison."""

    comparison_id: str
    comparison_timestamp: str
    entries: list[ComparisonEntry]
    best_in_report: str
    champion_candidate: str | None
    current_champion: dict[str, Any] | None
    config: ComparisonConfig
    backtest_config: dict[str, Any]
    dataset_variant: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison_id": self.comparison_id,
            "comparison_timestamp": self.comparison_timestamp,
            "entries": [e.to_dict() for e in self.entries],
            "best_in_report": self.best_in_report,
            "champion_candidate": self.champion_candidate,
            "current_champion": self.current_champion,
            "config": self.config.to_dict(),
            "backtest_config": self.backtest_config,
            "dataset_variant": self.dataset_variant,
        }


# ---------------------------------------------------------------------------
# Deterministic sort key
# ---------------------------------------------------------------------------
def _sort_key(entry: LadderEntry) -> tuple[float, float, float, str]:
    """Deterministic 4-tuple sort key for ranking models.

    Ordering (all produce ascending sort → best model first):
      1. -nDCG@20 mean   (higher nDCG is better → negate)
      2. -WR@20 mean     (higher WR is better → negate)
      3. +Brier mean     (lower Brier is better → keep)
      4. model name      (alphabetical final tie-break)
    """
    s = entry.metric_summary
    return (
        -s.ndcg_20_mean,
        -s.weighted_recall_20_mean,
        s.brier_score_mean,
        entry.model_name,
    )


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def _compute_verdict(
    *,
    is_best: bool,
    champion: ChampionRecord | None,
    ndcg_delta: float,
    config: ComparisonConfig,
) -> CandidateVerdict:
    """Compute the verdict for a single model.

    Only the best_in_report model can receive ELIGIBLE or NO_CHAMPION.
    Non-best models always receive a BLOCKED verdict.
    """
    if champion is None:
        return CandidateVerdict.NO_CHAMPION if is_best else CandidateVerdict.BLOCKED_BELOW_DELTA

    if not is_best:
        if ndcg_delta == 0.0:
            return CandidateVerdict.BLOCKED_TIED
        return CandidateVerdict.BLOCKED_BELOW_DELTA

    # Best-in-report model — check threshold
    if ndcg_delta == 0.0:
        return CandidateVerdict.BLOCKED_TIED
    if ndcg_delta >= config.min_ndcg_delta:
        return CandidateVerdict.ELIGIBLE
    return CandidateVerdict.BLOCKED_BELOW_DELTA


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------
def compare_to_champion(
    ladder_result: LadderResult,
    champion: ChampionRecord | None,
    config: ComparisonConfig | None = None,
    dataset_variant: str = "raw",
) -> ComparisonResult:
    """Compare ladder results against the current champion.

    Args:
        ladder_result: Completed LadderResult from run_ladder().
        champion: Current champion record, or None for first-ever.
        config: Comparison thresholds (defaults to ComparisonConfig()).
        dataset_variant: For provenance tracking.

    Returns:
        ComparisonResult with sorted entries, verdicts, and candidate.
    """
    if config is None:
        config = ComparisonConfig()

    comparison_id = str(uuid.uuid4())
    comparison_timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Sort using deterministic 4-tuple key
    sorted_ladder = sorted(ladder_result.entries, key=_sort_key)

    best_model_name = sorted_ladder[0].model_name if sorted_ladder else ""

    entries: list[ComparisonEntry] = []
    for ladder_entry in sorted_ladder:
        s = ladder_entry.metric_summary
        is_best = ladder_entry.model_name == best_model_name

        if champion is not None:
            delta_vs_champion = {
                "ndcg_20_mean": s.ndcg_20_mean - champion.ndcg_20_mean,
                "weighted_recall_20_mean": (
                    s.weighted_recall_20_mean - champion.weighted_recall_20_mean
                ),
                "brier_score_mean": (
                    champion.brier_score_mean - s.brier_score_mean
                ),  # positive = improvement
            }
        else:
            delta_vs_champion = {
                "ndcg_20_mean": 0.0,
                "weighted_recall_20_mean": 0.0,
                "brier_score_mean": 0.0,
            }

        verdict = _compute_verdict(
            is_best=is_best,
            champion=champion,
            ndcg_delta=delta_vs_champion["ndcg_20_mean"],
            config=config,
        )

        entries.append(
            ComparisonEntry(
                model_name=ladder_entry.model_name,
                metric_summary=ladder_entry.metric_summary,
                verdict=verdict,
                delta_vs_champion=delta_vs_champion,
                is_best_in_report=is_best,
            )
        )

    # Only best_in_report can be champion_candidate
    champion_candidate: str | None = None
    if entries:
        best_entry = entries[0]
        if best_entry.verdict in (
            CandidateVerdict.ELIGIBLE,
            CandidateVerdict.NO_CHAMPION,
        ):
            champion_candidate = best_entry.model_name

    logger.info(
        "comparison_complete",
        comparison_id=comparison_id,
        best_in_report=best_model_name,
        champion_candidate=champion_candidate,
        models_compared=len(entries),
    )

    return ComparisonResult(
        comparison_id=comparison_id,
        comparison_timestamp=comparison_timestamp,
        entries=entries,
        best_in_report=best_model_name,
        champion_candidate=champion_candidate,
        current_champion=champion.to_dict() if champion else None,
        config=config,
        backtest_config=ladder_result.config.to_dict(),
        dataset_variant=dataset_variant,
    )


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def write_comparison_report(
    comparison_result: ComparisonResult,
    output_dir: Path,
) -> list[str]:
    """Write comparison report artifacts (JSON + Markdown).

    Args:
        comparison_result: The completed ComparisonResult.
        output_dir: Directory to write into (created if needed).

    Returns:
        List of artifact file paths as strings.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = _write_comparison_json(comparison_result, output_dir)
    md_path = _write_comparison_markdown(comparison_result, output_dir)

    paths = [str(json_path), str(md_path)]
    logger.info("comparison_report_written", paths=paths)
    return paths


def _write_comparison_json(result: ComparisonResult, output_dir: Path) -> Path:
    path = output_dir / "comparison_report.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def _write_comparison_markdown(result: ComparisonResult, output_dir: Path) -> Path:
    path = output_dir / "comparison_report.md"

    lines = [
        "# Experiment Comparison Report",
        "",
        f"**Comparison ID:** {result.comparison_id}",
        f"**Timestamp:** {result.comparison_timestamp}",
        f"**Dataset variant:** {result.dataset_variant}",
        f"**Models compared:** {len(result.entries)}",
        "",
    ]

    # Current champion section
    if result.current_champion is not None:
        champ = result.current_champion
        lines.extend(
            [
                "## Current Champion",
                "",
                f"- **Model:** {champ['model_name']}",
                f"- **nDCG@20:** {champ['ndcg_20_mean']:.4f}",
                f"- **WR@20:** {champ['weighted_recall_20_mean']:.4f}",
                f"- **Brier:** {champ['brier_score_mean']:.4f}",
                f"- **Promoted at:** {champ['promoted_at']}",
                f"- **Approver:** {champ['approver']}",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Current Champion",
                "",
                "No champion set. First promotion will establish the baseline.",
                "",
            ]
        )

    # Ranked results table
    lines.extend(
        [
            "## Results (ranked by nDCG@20)",
            "",
            "| Rank | Model | nDCG@20 | WR@20 | Brier | Verdict |",
            "|------|-------|---------|-------|-------|---------|",
        ]
    )
    for i, entry in enumerate(result.entries):
        s = entry.metric_summary
        lines.append(
            f"| {i + 1} "
            f"| {entry.model_name} "
            f"| {s.ndcg_20_mean:.4f} "
            f"| {s.weighted_recall_20_mean:.4f} "
            f"| {s.brier_score_mean:.4f} "
            f"| {entry.verdict.value} |"
        )
    lines.append("")

    # Delta table (only if champion exists)
    if result.current_champion is not None:
        lines.extend(
            [
                "## Deltas vs Champion",
                "",
                "| Model | nDCG delta | WR delta | Brier delta |",
                "|-------|-----------|----------|-------------|",
            ]
        )
        for entry in result.entries:
            d = entry.delta_vs_champion
            lines.append(
                f"| {entry.model_name} "
                f"| {d['ndcg_20_mean']:+.4f} "
                f"| {d['weighted_recall_20_mean']:+.4f} "
                f"| {d['brier_score_mean']:+.4f} |"
            )
        lines.append("")

    # Candidate verdict
    lines.append("## Champion Candidate Decision")
    lines.append("")
    if result.champion_candidate is not None:
        lines.append(
            f"**Champion candidate: {result.champion_candidate}** "
            f"— ready for PO approval via `promote --confirm`."
        )
    else:
        lines.append(
            "**No champion candidate.** No model meets the minimum improvement threshold."
        )
    lines.append("")

    # Configuration
    lines.extend(
        [
            "## Comparison Configuration",
            "",
            f"- min_ndcg_delta: {result.config.min_ndcg_delta}",
            f"- min_wr_delta: {result.config.min_wr_delta} (reporting only)",
            f"- max_brier_delta: {result.config.max_brier_delta} (reporting only)",
            "",
        ]
    )

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path
