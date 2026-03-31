"""Ranking module — deterministic top-K selection with hard validation.

Sorts part scores in descending order, applies deterministic tie-breaking
(lower part ID wins), selects top K, and validates the output against
hard domain invariants.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from c5_forecasting.domain.constants import TOP_K, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore

logger = structlog.get_logger(__name__)


@dataclass
class RankedForecast:
    """Validated ranked forecast output."""

    rankings: list[RankedEntry]
    model_name: str
    k: int


@dataclass
class RankedEntry:
    """A single entry in the ranked forecast."""

    rank: int
    part_id: int
    score: float


class ForecastValidationError(Exception):
    """Raised when a ranked forecast fails hard validation."""


def rank_and_select(
    scores: list[PartScore],
    k: int = TOP_K,
    model_name: str = "frequency_baseline",
) -> RankedForecast:
    """Sort scores, apply deterministic tie-breaking, select top K, and validate.

    Tie-breaking rule: when two parts have equal scores, the part with the
    **lower part ID** is ranked higher. This ensures identical input always
    produces identical output.

    Args:
        scores: List of PartScore from a scoring model.
        k: Number of top entries to select (default: TOP_K = 20).
        model_name: Name of the scoring model for provenance.

    Returns:
        A validated :class:`RankedForecast`.

    Raises:
        ForecastValidationError: If the output fails hard validation.
    """
    # Sort: primary key = score descending, secondary key = part_id ascending
    sorted_scores = sorted(scores, key=lambda s: (-s.score, s.part_id))

    # Select top K
    top_k = sorted_scores[:k]

    rankings = [
        RankedEntry(rank=i + 1, part_id=s.part_id, score=s.score) for i, s in enumerate(top_k)
    ]

    forecast = RankedForecast(rankings=rankings, model_name=model_name, k=k)

    # Hard validation
    validate_forecast(forecast)

    logger.info(
        "forecast_ranked",
        model=model_name,
        k=k,
        top_3=[r.part_id for r in rankings[:3]],
    )

    return forecast


def validate_forecast(forecast: RankedForecast) -> None:
    """Hard-validate a ranked forecast against domain invariants.

    Checks:
    1. Exactly K entries
    2. All part IDs in VALID_PART_IDS (1..39)
    3. 0 never appears
    4. No duplicate part IDs

    Args:
        forecast: The forecast to validate.

    Raises:
        ForecastValidationError: On any violation.
    """
    ids = [r.part_id for r in forecast.rankings]

    # Check count
    if len(ids) != forecast.k:
        raise ForecastValidationError(f"Expected {forecast.k} ranked entries, got {len(ids)}")

    # Check for 0 (CRITICAL invariant)
    if 0 in ids:
        raise ForecastValidationError(
            "CRITICAL: Part ID 0 found in forecast output. 0 is never a valid predicted part ID."
        )

    # Check all IDs are valid
    invalid = [pid for pid in ids if pid not in VALID_PART_IDS]
    if invalid:
        raise ForecastValidationError(
            f"Invalid part IDs in forecast: {invalid}. Must be in 1..39."
        )

    # Check for duplicates
    if len(set(ids)) != len(ids):
        seen: set[int] = set()
        dupes = []
        for pid in ids:
            if pid in seen:
                dupes.append(pid)
            seen.add(pid)
        raise ForecastValidationError(f"Duplicate part IDs in forecast: {dupes}")
