"""Recency-weighted frequency baseline — scores parts by exponentially
decay-weighted occurrence rate.

For each part P_i, the score is the weighted fraction of rows where
P_i > 0, with more recent rows receiving higher weight via exponential
decay: w_j = decay^(n-1-j) for the j-th row (0-indexed, chronological).

Default decay=0.995 gives a half-life of ~138 days (about 5 months).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from c5_forecasting.domain.constants import PART_COLUMNS, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore

logger = structlog.get_logger(__name__)

MODEL_NAME = "recency_weighted"
DEFAULT_DECAY = 0.995


def compute_recency_weighted_scores(
    df: pd.DataFrame,
    *,
    decay: float = DEFAULT_DECAY,
) -> list[PartScore]:
    """Compute recency-weighted occurrence-rate scores for all 39 parts.

    Args:
        df: Working dataset DataFrame with P_1..P_39 columns (chronological).
        decay: Exponential decay factor per row. Default 0.995 (~138-day half-life).

    Returns:
        List of PartScore for all 39 parts, unsorted.

    Raises:
        ValueError: If no part columns present or DataFrame empty.
    """
    part_cols = [c for c in PART_COLUMNS if c in df.columns]
    if not part_cols:
        raise ValueError("No part columns found in DataFrame")

    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("DataFrame is empty — cannot compute recency-weighted scores")

    # Compute weights: w[j] = decay^(n-1-j) for j in 0..n-1
    # Most recent row (j=n-1) gets weight decay^0 = 1.0
    exponents = np.arange(n_rows - 1, -1, -1, dtype=np.float64)
    weights = np.float_power(decay, exponents)
    total_weight = float(weights.sum())

    scores: list[PartScore] = []
    for col in part_cols:
        part_id = int(col.split("_")[1])
        if part_id not in VALID_PART_IDS:
            continue
        indicators = (df[col] > 0).to_numpy(dtype=np.float64)
        weighted_sum = float((indicators * weights).sum())
        rate = weighted_sum / total_weight
        scores.append(PartScore(part_id=part_id, score=rate))

    logger.info(
        "recency_weighted_scores_computed",
        n_parts=len(scores),
        n_rows=n_rows,
        decay=decay,
    )

    return scores
