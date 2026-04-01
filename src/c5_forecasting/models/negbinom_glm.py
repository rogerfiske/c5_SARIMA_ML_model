"""Negative Binomial GLM challenger model — per-part count regression.

Fits 39 independent Negative Binomial GLMs (one per P_1..P_39 count series)
using statsmodels GLM with a NegativeBinomial family, then min-max normalises
the 1-step-ahead forecasts into [0, 1] scores.

Features per part: lag-1 (yesterday's count) and lag-7 (same weekday last week).

Fallback cascade per part:
  1. NegBinom GLM with lag1 + lag7
  2. NegBinom GLM with lag1 only
  3. Series mean
  4. 0.0

All negative forecasts are clamped to 0 before normalisation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import structlog

from c5_forecasting.domain.constants import PART_COLUMNS, TOP_K
from c5_forecasting.models.baseline import PartScore

logger = structlog.get_logger(__name__)

MODEL_NAME = "negbinom_glm"

_MAX_ITER = 100


def _build_lag_features(
    series: np.ndarray,
    include_lag7: bool = True,
) -> np.ndarray:
    """Build a feature matrix of lagged values from a 1-D count series.

    Args:
        series: 1-D array of historical counts.
        include_lag7: Whether to include the lag-7 feature.

    Returns:
        2-D array of shape ``(n, 2)`` or ``(n, 1)`` with columns
        ``[lag1]`` or ``[lag1, lag7]``.  First rows with unavailable
        lags are filled with 0.
    """
    n = len(series)
    lag1 = np.empty(n, dtype=np.float64)
    lag1[0] = 0.0
    lag1[1:] = series[:-1]

    if not include_lag7:
        return lag1.reshape(-1, 1)

    lag7 = np.empty(n, dtype=np.float64)
    lag7[: min(7, n)] = 0.0
    if n > 7:
        lag7[7:] = series[:-7]

    return np.column_stack([lag1, lag7])


def _fit_and_forecast_glm(series: np.ndarray) -> float:
    """Fit a NegBinom GLM on *series* and return a 1-step-ahead forecast.

    Tries lag1+lag7 features first, then lag1 only, then the series mean.
    Returns 0.0 only when the series is empty or entirely non-finite.
    """
    from statsmodels.genmod.families import NegativeBinomial
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.tools import add_constant

    if len(series) < 2:
        if len(series) > 0 and np.isfinite(series[0]):
            return max(float(series[0]), 0.0)
        return 0.0

    for include_lag7 in [True, False]:
        try:
            features = _build_lag_features(series, include_lag7=include_lag7)
            y = series.astype(np.float64)
            x_const = add_constant(features)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = GLM(y, x_const, family=NegativeBinomial())
                result = model.fit(maxiter=_MAX_ITER, disp=False)

            # Build prediction features for next step
            last_lag1 = float(series[-1])
            if include_lag7:
                last_lag7 = float(series[-7]) if len(series) >= 7 else 0.0
                new_x = np.array([[1.0, last_lag1, last_lag7]])
            else:
                new_x = np.array([[1.0, last_lag1]])

            forecast = float(result.predict(new_x)[0])
            if np.isfinite(forecast):
                return max(forecast, 0.0)
        except Exception:
            continue

    # Fallback: series mean
    mean_val = float(np.nanmean(series))
    if np.isfinite(mean_val):
        return max(mean_val, 0.0)
    return 0.0


def _normalize_scores(raw_forecasts: dict[str, float]) -> dict[str, float]:
    """Min-max normalise forecasts to [0, 1].

    When all values are identical (max == min), returns a uniform score
    of ``TOP_K / len(forecasts)`` for every part so that the downstream
    ranking treats all parts equally.
    """
    values = list(raw_forecasts.values())
    min_val = min(values)
    max_val = max(values)

    if max_val == min_val:
        uniform = TOP_K / len(values)
        return {k: uniform for k in raw_forecasts}

    span = max_val - min_val
    return {k: (v - min_val) / span for k, v in raw_forecasts.items()}


def negbinom_glm_scoring(df: pd.DataFrame) -> list[PartScore]:
    """Score all 39 parts using per-part Negative Binomial GLM.

    Conforms to the ``ScoringFunction`` protocol::

        (pd.DataFrame) -> list[PartScore]

    Args:
        df: Training slice of the working dataset (date + P_1..P_39).

    Returns:
        39 :class:`PartScore` entries sorted by score descending, then by
        part_id ascending as a tie-breaker.
    """
    raw_forecasts: dict[str, float] = {}

    for col in PART_COLUMNS:
        series = df[col].to_numpy(dtype=float, na_value=0.0)
        forecast = _fit_and_forecast_glm(series)
        raw_forecasts[col] = forecast

    normalized = _normalize_scores(raw_forecasts)

    scores: list[PartScore] = []
    for col, score in normalized.items():
        part_id = int(col.replace("P_", ""))
        scores.append(PartScore(part_id=part_id, score=score))

    scores.sort(key=lambda ps: (-ps.score, ps.part_id))
    return scores
