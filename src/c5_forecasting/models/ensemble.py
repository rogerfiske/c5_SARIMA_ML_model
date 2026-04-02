"""Ensemble scoring methods — composing individual models into combined forecasts.

Three ensemble strategies are provided:
  1. ensemble_avg — arithmetic mean of scores from 4 diverse models
  2. ensemble_rank_avg — Borda-count style rank averaging from 4 models
  3. ensemble_weighted — fixed weighted average across all 7 models

Each ensemble conforms to the ScoringFunction protocol:
    (pd.DataFrame) -> list[PartScore]

Component models are resolved at call-time via get_scoring_function() to
avoid circular imports. Weights for ensemble_weighted are frozen constants
derived from Story 3.3 backtest results — they are NOT computed at scoring
time, which prevents data leakage during backtest evaluation.
"""

from __future__ import annotations

import pandas as pd
import structlog

from c5_forecasting.domain.constants import TOP_K
from c5_forecasting.models.baseline import PartScore

logger = structlog.get_logger(__name__)

# Component model lists for score-averaging and rank-averaging ensembles
_AVG_COMPONENTS: list[str] = [
    "frequency_baseline",
    "gbm_ranking",
    "negbinom_glm",
    "sarima",
]

_RANK_AVG_COMPONENTS: list[str] = [
    "frequency_baseline",
    "gbm_ranking",
    "negbinom_glm",
    "sarima",
]

# Frozen weights from Story 3.3 nDCG@20 results (sum = 1.00)
# These are source-code constants, NOT computed at scoring time.
_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "frequency_baseline": 0.30,
    "negbinom_glm": 0.25,
    "gbm_ranking": 0.20,
    "sarima": 0.15,
    "recency_weighted": 0.05,
    "rolling_window": 0.03,
    "uniform_baseline": 0.02,
}


def _get_component_scores(
    component_names: list[str],
    df: pd.DataFrame,
) -> dict[str, list[PartScore]]:
    """Call each component model and return their scores.

    Args:
        component_names: List of model names to call.
        df: Training slice DataFrame.

    Returns:
        Dict keyed by model name, values are list[PartScore] from each model.
    """
    from c5_forecasting.models.registry import get_scoring_function

    results: dict[str, list[PartScore]] = {}
    for name in component_names:
        fn = get_scoring_function(name)
        results[name] = fn(df)
    return results


def _normalize_min_max(scores_dict: dict[int, float]) -> dict[int, float]:
    """Min-max normalize scores to [0, 1].

    When all values are identical (max == min), returns a uniform score
    of ``TOP_K / len(scores_dict)`` for every part so that the downstream
    ranking treats all parts equally.

    Args:
        scores_dict: Dict keyed by part_id, values are raw scores.

    Returns:
        Dict with same keys, values normalized to [0, 1].
    """
    values = list(scores_dict.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        uniform = TOP_K / len(scores_dict)
        return {k: uniform for k in scores_dict}

    span = max_val - min_val
    return {k: (v - min_val) / span for k, v in scores_dict.items()}


def ensemble_avg_scoring(df: pd.DataFrame) -> list[PartScore]:
    """Score all 39 parts using arithmetic mean of 4 component models.

    Component models: frequency_baseline, gbm_ranking, negbinom_glm, sarima.
    For each part, computes the average score across all 4 components.

    Conforms to the ``ScoringFunction`` protocol::

        (pd.DataFrame) -> list[PartScore]

    Args:
        df: Training slice of the working dataset (date + P_1..P_39).

    Returns:
        39 :class:`PartScore` entries sorted by score descending, then by
        part_id ascending as a tie-breaker.
    """
    component_scores = _get_component_scores(_AVG_COMPONENTS, df)

    all_scores: dict[int, list[float]] = {}
    for _, scores in component_scores.items():
        for ps in scores:
            all_scores.setdefault(ps.part_id, []).append(ps.score)

    result: list[PartScore] = []
    for part_id, score_list in all_scores.items():
        avg = sum(score_list) / len(score_list)
        result.append(PartScore(part_id=part_id, score=avg))

    result.sort(key=lambda ps: (-ps.score, ps.part_id))

    logger.info(
        "ensemble_avg_scores_computed",
        n_components=len(_AVG_COMPONENTS),
        n_parts=len(result),
    )
    return result


def ensemble_rank_avg_scoring(df: pd.DataFrame) -> list[PartScore]:
    """Score all 39 parts using rank averaging (Borda-count).

    Component models: frequency_baseline, gbm_ranking, negbinom_glm, sarima.
    For each part, computes the average rank across all 4 components, then
    converts average rank to a score.

    Rank conversion: score = 1.0 - (avg_rank - 1) / 38
    Maps rank 1 → score 1.0, rank 39 → score 0.0.

    Conforms to the ``ScoringFunction`` protocol::

        (pd.DataFrame) -> list[PartScore]

    Args:
        df: Training slice of the working dataset (date + P_1..P_39).

    Returns:
        39 :class:`PartScore` entries sorted by score descending, then by
        part_id ascending as a tie-breaker.
    """
    component_scores = _get_component_scores(_RANK_AVG_COMPONENTS, df)

    all_ranks: dict[int, list[int]] = {}
    for _, scores in component_scores.items():
        # Sort by (-score, part_id) to get ranking
        sorted_scores = sorted(scores, key=lambda ps: (-ps.score, ps.part_id))
        for rank, ps in enumerate(sorted_scores, 1):
            all_ranks.setdefault(ps.part_id, []).append(rank)

    result: list[PartScore] = []
    for part_id, rank_list in all_ranks.items():
        avg_rank = sum(rank_list) / len(rank_list)
        # Convert average rank to score: lower rank = higher score
        score = 1.0 - (avg_rank - 1) / 38.0
        result.append(PartScore(part_id=part_id, score=score))

    result.sort(key=lambda ps: (-ps.score, ps.part_id))

    logger.info(
        "ensemble_rank_avg_scores_computed",
        n_components=len(_RANK_AVG_COMPONENTS),
        n_parts=len(result),
    )
    return result


def ensemble_weighted_scoring(df: pd.DataFrame) -> list[PartScore]:
    """Score all 39 parts using fixed weighted average across all 7 models.

    Component models: all 7 individual models weighted by frozen constants
    derived from Story 3.3 nDCG@20 results. Weights are NOT computed at
    scoring time, which prevents data leakage.

    Conforms to the ``ScoringFunction`` protocol::

        (pd.DataFrame) -> list[PartScore]

    Args:
        df: Training slice of the working dataset (date + P_1..P_39).

    Returns:
        39 :class:`PartScore` entries sorted by score descending, then by
        part_id ascending as a tie-breaker.
    """
    component_names = list(_ENSEMBLE_WEIGHTS.keys())
    component_scores = _get_component_scores(component_names, df)

    weighted_sums: dict[int, float] = {}
    for model_name, scores in component_scores.items():
        weight = _ENSEMBLE_WEIGHTS[model_name]
        for ps in scores:
            weighted_sums[ps.part_id] = weighted_sums.get(ps.part_id, 0.0) + ps.score * weight

    # Apply min-max normalization
    normalized = _normalize_min_max(weighted_sums)

    result: list[PartScore] = []
    for part_id, score in normalized.items():
        result.append(PartScore(part_id=part_id, score=score))

    result.sort(key=lambda ps: (-ps.score, ps.part_id))

    logger.info(
        "ensemble_weighted_scores_computed",
        n_components=len(_ENSEMBLE_WEIGHTS),
        n_parts=len(result),
    )
    return result
