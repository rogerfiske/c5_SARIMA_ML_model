"""Champion model state — persistence and promotion gate.

The champion is the currently approved best model. Champion state is
filesystem-backed via a single ``champion.json`` file.  Promotion
requires an explicit CLI action (``promote --confirm``); no code path
writes champion.json without that explicit gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from c5_forecasting.evaluation.comparison import ComparisonResult

logger = structlog.get_logger(__name__)

CHAMPION_FILENAME = "champion.json"


@dataclass
class ChampionRecord:
    """Persisted record of the currently promoted champion model."""

    model_name: str
    ndcg_20_mean: float
    weighted_recall_20_mean: float
    brier_score_mean: float
    promoted_at: str  # ISO 8601 timestamp
    promoted_from_comparison: str  # comparison_id for audit trail
    backtest_config: dict[str, Any]
    dataset_variant: str
    approver: str  # e.g. "PO"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "ndcg_20_mean": round(self.ndcg_20_mean, 6),
            "weighted_recall_20_mean": round(self.weighted_recall_20_mean, 6),
            "brier_score_mean": round(self.brier_score_mean, 6),
            "promoted_at": self.promoted_at,
            "promoted_from_comparison": self.promoted_from_comparison,
            "backtest_config": self.backtest_config,
            "dataset_variant": self.dataset_variant,
            "approver": self.approver,
        }


def load_champion(artifacts_dir: Path) -> ChampionRecord | None:
    """Load the current champion from champion.json, or None if not set."""
    path = artifacts_dir / CHAMPION_FILENAME
    if not path.exists():
        logger.info("champion_not_found", path=str(path))
        return None
    with open(path) as f:
        data = json.load(f)
    logger.info(
        "champion_loaded",
        model=data["model_name"],
        promoted_at=data["promoted_at"],
    )
    return ChampionRecord(**data)


def save_champion(record: ChampionRecord, artifacts_dir: Path) -> Path:
    """Write champion.json.  ONLY called by promote_champion()."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / CHAMPION_FILENAME
    with open(path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)
    logger.info("champion_saved", model=record.model_name, path=str(path))
    return path


def promote_champion(
    comparison_result: ComparisonResult,
    approver: str,
    artifacts_dir: Path,
) -> ChampionRecord:
    """Promote the champion_candidate to actual champion.

    Reads the candidate's metrics from the ComparisonResult and persists
    a new ChampionRecord to champion.json.

    Args:
        comparison_result: Completed ComparisonResult with a candidate.
        approver: Name/role of the person approving (e.g. "PO").
        artifacts_dir: Directory to write champion.json into.

    Returns:
        The newly created ChampionRecord.

    Raises:
        ValueError: If comparison_result has no champion_candidate.
    """
    if comparison_result.champion_candidate is None:
        raise ValueError(
            "No champion candidate in this comparison. "
            "Cannot promote without an eligible or first-time candidate."
        )

    # Find the candidate's entry
    candidate_entry = next(
        e
        for e in comparison_result.entries
        if e.model_name == comparison_result.champion_candidate
    )

    s = candidate_entry.metric_summary
    record = ChampionRecord(
        model_name=candidate_entry.model_name,
        ndcg_20_mean=s.ndcg_20_mean,
        weighted_recall_20_mean=s.weighted_recall_20_mean,
        brier_score_mean=s.brier_score_mean,
        promoted_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        promoted_from_comparison=comparison_result.comparison_id,
        backtest_config=comparison_result.backtest_config,
        dataset_variant=comparison_result.dataset_variant,
        approver=approver,
    )

    save_champion(record, artifacts_dir)

    logger.info(
        "champion_promoted",
        model=record.model_name,
        ndcg=round(record.ndcg_20_mean, 4),
        approver=approver,
    )

    return record
