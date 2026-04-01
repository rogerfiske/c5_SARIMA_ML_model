"""Model registry — maps model names to ScoringFunction callables.

The registry is the single source of truth for all available baseline
models. It is used by the CLI and ladder runner to look up scoring
functions by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from c5_forecasting.evaluation.backtest import ScoringFunction

logger = structlog.get_logger(__name__)


def get_model_registry() -> dict[str, ScoringFunction]:
    """Return a dict mapping model names to their scoring functions.

    Imports are done inside the function to avoid circular dependencies
    and keep startup fast.
    """
    from c5_forecasting.models.baseline import compute_frequency_scores
    from c5_forecasting.models.recency_weighted import compute_recency_weighted_scores
    from c5_forecasting.models.rolling_window import compute_rolling_window_scores
    from c5_forecasting.models.uniform import compute_uniform_scores

    return {
        "frequency_baseline": compute_frequency_scores,
        "recency_weighted": compute_recency_weighted_scores,
        "rolling_window": compute_rolling_window_scores,
        "uniform_baseline": compute_uniform_scores,
    }


def get_model_names() -> list[str]:
    """Return sorted list of all registered model names."""
    return sorted(get_model_registry().keys())


def get_scoring_function(name: str) -> ScoringFunction:
    """Look up a scoring function by model name.

    Args:
        name: Registered model name.

    Returns:
        The scoring function callable.

    Raises:
        KeyError: If the name is not registered.
    """
    registry = get_model_registry()
    if name not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown model {name!r}. Available: {available}")
    return registry[name]
