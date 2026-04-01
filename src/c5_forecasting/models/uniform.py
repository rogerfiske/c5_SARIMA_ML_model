"""Uniform baseline — assigns the same score to all parts.

The trivial uninformative baseline. Score = TOP_K / MAX_PART_ID = 20/39
for all 39 parts. This represents the prior probability that any given
part would be selected by random top-20 selection from 39 candidates.

With all scores equal, deterministic tie-breaking (by part_id ascending)
always selects parts 1-20.
"""

from __future__ import annotations

import pandas as pd
import structlog

from c5_forecasting.domain.constants import MAX_PART_ID, TOP_K, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore

logger = structlog.get_logger(__name__)

MODEL_NAME = "uniform_baseline"
UNIFORM_SCORE = TOP_K / MAX_PART_ID  # 20/39 ~ 0.5128


def compute_uniform_scores(df: pd.DataFrame) -> list[PartScore]:
    """Return a constant score for all 39 parts.

    Args:
        df: Working dataset DataFrame (used only for validation; data is ignored).

    Returns:
        List of 39 PartScore objects, all with score = 20/39.

    Raises:
        ValueError: If DataFrame is empty.
    """
    if len(df) == 0:
        raise ValueError("DataFrame is empty — cannot compute uniform scores")

    scores = [PartScore(part_id=pid, score=UNIFORM_SCORE) for pid in sorted(VALID_PART_IDS)]

    logger.info(
        "uniform_scores_computed",
        n_parts=len(scores),
        n_rows=len(df),
        score=round(UNIFORM_SCORE, 6),
    )

    return scores
