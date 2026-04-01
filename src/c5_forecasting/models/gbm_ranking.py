"""Gradient-boosted tree ranking challenger — per-part GBM regression.

Fits 39 independent HistGradientBoostingRegressor models (one per P_1..P_39
count series) using scikit-learn, then min-max normalises the 1-step-ahead
forecasts into [0, 1] scores.

Features per part (all derived from historical data only):
  1. lag_1          — yesterday's count
  2. lag_7          — same weekday last week
  3. lag_14         — same weekday 2 weeks ago
  4. rolling_mean_7 — 7-day rolling mean (shifted to avoid leakage)
  5. rolling_mean_30— 30-day rolling mean (shifted to avoid leakage)
  6. day_of_week    — 0–6 from date column

Training approach: X = features[0..n-2], y = counts[1..n-1]
(predict next day's count from today's features). Prediction uses the
last row's features.

Fallback cascade per part:
  1. HistGradientBoostingRegressor with all 6 features
  2. Series mean
  3. 0.0

All negative forecasts are clamped to 0 before normalisation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from c5_forecasting.domain.constants import PART_COLUMNS, TOP_K
from c5_forecasting.models.baseline import PartScore

logger = structlog.get_logger(__name__)

MODEL_NAME = "gbm_ranking"

_RANDOM_STATE = 42
_MAX_ITER = 100
_MAX_DEPTH = 4
_MIN_SAMPLES_LEAF = 20


def _build_features(
    series: np.ndarray,
    dates: np.ndarray,
) -> np.ndarray:
    """Build a feature matrix from a 1-D count series and date array.

    Args:
        series: 1-D array of historical counts (float64).
        dates: 1-D array of datetime64 values corresponding to *series*.

    Returns:
        2-D array of shape ``(n, 6)`` with columns
        ``[lag_1, lag_7, lag_14, rolling_mean_7, rolling_mean_30, day_of_week]``.
        Early rows with unavailable lags/windows are filled with 0.
    """
    s = pd.Series(series, dtype=np.float64)

    lag_1 = s.shift(1).fillna(0.0).to_numpy()
    lag_7 = s.shift(7).fillna(0.0).to_numpy()
    lag_14 = s.shift(14).fillna(0.0).to_numpy()

    # Rolling means of *past* values — shift(1) ensures we exclude the
    # current row's count, preventing any same-row leakage.
    rolling_mean_7 = s.shift(1).rolling(window=7, min_periods=1).mean().fillna(0.0).to_numpy()
    rolling_mean_30 = s.shift(1).rolling(window=30, min_periods=1).mean().fillna(0.0).to_numpy()

    day_of_week = pd.to_datetime(dates).dayofweek.to_numpy(dtype=np.float64)

    return np.column_stack([lag_1, lag_7, lag_14, rolling_mean_7, rolling_mean_30, day_of_week])


def _fit_and_forecast_gbm(
    series: np.ndarray,
    dates: np.ndarray,
) -> float:
    """Fit a HistGradientBoostingRegressor and return a 1-step-ahead forecast.

    Training: X = features[0..n-2], y = series[1..n-1].
    Prediction: features[-1] → forecast of the next day's count.

    Falls back to the series mean if the model fails, then to 0.0.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    if len(series) < 2:
        if len(series) > 0 and np.isfinite(series[0]):
            return max(float(series[0]), 0.0)
        return 0.0

    try:
        features = _build_features(series, dates)

        # Predict next-day count from today's features.
        x_train = features[:-1]  # rows 0..n-2
        y_train = series[1:]  # rows 1..n-1

        model = HistGradientBoostingRegressor(
            max_iter=_MAX_ITER,
            max_depth=_MAX_DEPTH,
            min_samples_leaf=_MIN_SAMPLES_LEAF,
            random_state=_RANDOM_STATE,
        )
        model.fit(x_train, y_train)

        x_pred = features[-1:].reshape(1, -1)
        forecast = float(model.predict(x_pred)[0])

        if np.isfinite(forecast):
            return max(forecast, 0.0)
    except Exception:
        logger.debug("gbm_fit_failed", exc_info=True)

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


def gbm_ranking_scoring(df: pd.DataFrame) -> list[PartScore]:
    """Score all 39 parts using per-part gradient-boosted tree regression.

    Conforms to the ``ScoringFunction`` protocol::

        (pd.DataFrame) -> list[PartScore]

    Args:
        df: Training slice of the working dataset (date + P_1..P_39).

    Returns:
        39 :class:`PartScore` entries sorted by score descending, then by
        part_id ascending as a tie-breaker.
    """
    dates = df["date"].to_numpy()
    raw_forecasts: dict[str, float] = {}

    for col in PART_COLUMNS:
        series = df[col].to_numpy(dtype=float, na_value=0.0)
        forecast = _fit_and_forecast_gbm(series, dates)
        raw_forecasts[col] = forecast

    normalized = _normalize_scores(raw_forecasts)

    scores: list[PartScore] = []
    for col, score in normalized.items():
        part_id = int(col.replace("P_", ""))
        scores.append(PartScore(part_id=part_id, score=score))

    scores.sort(key=lambda ps: (-ps.score, ps.part_id))
    return scores
