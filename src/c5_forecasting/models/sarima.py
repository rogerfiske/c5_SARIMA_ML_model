"""SARIMA challenger model — per-part ARIMA(1,1,1) scoring.

Fits 39 independent ARIMA(1,1,1) models (one per P_1..P_39 count series)
using statsmodels SARIMAX, then min-max normalises the 1-step-ahead
forecasts into [0, 1] scores.

Fallback cascade per part:
  1. ARIMA(1,1,1)
  2. ARIMA(0,1,0)  (random walk)
  3. Last observed value
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

MODEL_NAME = "sarima"

_DEFAULT_ORDER: tuple[int, int, int] = (1, 1, 1)
_FALLBACK_ORDER: tuple[int, int, int] = (0, 1, 0)


def _fit_and_forecast(
    series: np.ndarray,
    order: tuple[int, int, int] = _DEFAULT_ORDER,
) -> float:
    """Fit ARIMA on *series* and return a 1-step-ahead forecast.

    Tries *order* first, then ``_FALLBACK_ORDER``, then the last
    observed value.  Returns 0.0 only when the series is empty or
    entirely non-finite.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    for attempt_order in [order, _FALLBACK_ORDER]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    series,
                    order=attempt_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(disp=False, maxiter=50)
                forecast = float(result.forecast(steps=1)[0])
                if np.isfinite(forecast):
                    return max(forecast, 0.0)
        except Exception:
            continue

    # Ultimate fallback: last observed value
    if len(series) > 0 and np.isfinite(series[-1]):
        return max(float(series[-1]), 0.0)
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


def sarima_scoring(df: pd.DataFrame) -> list[PartScore]:
    """Score all 39 parts using per-part ARIMA(1,1,1).

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
        forecast = _fit_and_forecast(series, _DEFAULT_ORDER)
        raw_forecasts[col] = forecast

    normalized = _normalize_scores(raw_forecasts)

    scores: list[PartScore] = []
    for col, score in normalized.items():
        part_id = int(col.replace("P_", ""))
        scores.append(PartScore(part_id=part_id, score=score))

    scores.sort(key=lambda ps: (-ps.score, ps.part_id))
    return scores
