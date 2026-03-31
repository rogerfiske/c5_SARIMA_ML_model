"""Frequency baseline model — scores parts by historical occurrence rate.

This is the canary baseline for Story 1.5. It computes, for each part P_i,
the fraction of days in the dataset where P_i > 0 (the historical occurrence
rate). Higher rate = higher predicted likelihood of appearing next.

This is deliberately simple and transparent — it is a vertical-slice baseline,
not the champion model.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import structlog

from c5_forecasting.domain.constants import PART_COLUMNS, VALID_PART_IDS

logger = structlog.get_logger(__name__)

MODEL_NAME = "frequency_baseline"


@dataclass
class PartScore:
    """Score for a single part ID."""

    part_id: int
    score: float


def compute_frequency_scores(df: pd.DataFrame) -> list[PartScore]:
    """Compute historical occurrence-rate scores for all 39 parts.

    For each part column P_i, the score is the fraction of rows where the
    value is > 0. Part columns must be numeric (Int64 or int).

    Args:
        df: Working dataset DataFrame with P_1..P_39 columns.

    Returns:
        List of :class:`PartScore` for all 39 parts, unsorted.

    Raises:
        ValueError: If no part columns are present.
    """
    part_cols = [c for c in PART_COLUMNS if c in df.columns]
    if not part_cols:
        raise ValueError("No part columns found in DataFrame")

    n_rows = len(df)
    if n_rows == 0:
        raise ValueError("DataFrame is empty — cannot compute frequency scores")

    scores: list[PartScore] = []
    for col in part_cols:
        # Extract the numeric part ID from column name "P_{id}"
        part_id = int(col.split("_")[1])
        if part_id not in VALID_PART_IDS:
            continue
        occurrence_count = int((df[col] > 0).sum())
        rate = occurrence_count / n_rows
        scores.append(PartScore(part_id=part_id, score=rate))

    logger.info(
        "frequency_scores_computed",
        n_parts=len(scores),
        n_rows=n_rows,
    )

    return scores
