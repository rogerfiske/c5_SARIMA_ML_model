"""Rolling-window frequency baseline — scores parts by occurrence rate
over only the most recent W rows.

For each part P_i, the score is the fraction of the last W rows where
P_i > 0. If the dataset has fewer than W rows, all available rows are used.
"""

from __future__ import annotations

import pandas as pd
import structlog

from c5_forecasting.domain.constants import PART_COLUMNS, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore

logger = structlog.get_logger(__name__)

MODEL_NAME = "rolling_window"
DEFAULT_WINDOW_SIZE = 365


def compute_rolling_window_scores(
    df: pd.DataFrame,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> list[PartScore]:
    """Compute occurrence-rate scores over the last W rows for all 39 parts.

    Args:
        df: Working dataset DataFrame with P_1..P_39 columns.
        window_size: Number of most-recent rows to use. Default 365.

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
        raise ValueError("DataFrame is empty — cannot compute rolling-window scores")

    # Use only the last window_size rows (or all rows if fewer)
    effective_window = min(window_size, n_rows)
    window_df = df.iloc[-effective_window:]
    w_rows = len(window_df)

    scores: list[PartScore] = []
    for col in part_cols:
        part_id = int(col.split("_")[1])
        if part_id not in VALID_PART_IDS:
            continue
        occurrence_count = int((window_df[col] > 0).sum())
        rate = occurrence_count / w_rows
        scores.append(PartScore(part_id=part_id, score=rate))

    logger.info(
        "rolling_window_scores_computed",
        n_parts=len(scores),
        n_rows=n_rows,
        window_size=window_size,
        effective_window=effective_window,
    )

    return scores
