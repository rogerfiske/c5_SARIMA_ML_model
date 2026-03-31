"""Rolling-origin backtesting engine for next-event ranking forecasts.

Evaluates forecasting methods over sequential cutoff dates using an
expanding-window approach: for each cutoff, train on all historical data
up to (and including) the cutoff, then predict and compare against the
realized next event.

Leakage prevention:
- DataFrame is sorted by date before windowing.
- Training folds use positional slicing (df.iloc[0:cutoff+1]).
- Each fold asserts train_max_date < target_date.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

import pandas as pd
import structlog

from c5_forecasting.domain.constants import DATE_COLUMN, PART_COLUMNS, TOP_K, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore
from c5_forecasting.ranking.ranker import rank_and_select

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Scoring function protocol
# ---------------------------------------------------------------------------
class ScoringFunction(Protocol):
    """Protocol for any scoring function usable by the backtest engine."""

    def __call__(self, df: pd.DataFrame) -> list[PartScore]: ...


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BacktestConfig:
    """Parameters controlling the rolling-origin backtest."""

    min_train_rows: int = 365
    step: int = 1
    max_windows: int | None = None
    k: int = TOP_K
    model_name: str = "frequency_baseline"

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_train_rows": self.min_train_rows,
            "step": self.step,
            "max_windows": self.max_windows,
            "k": self.k,
            "model_name": self.model_name,
        }


@dataclass
class BacktestFold:
    """Result of a single backtest evaluation fold."""

    fold_index: int
    cutoff_date: str
    target_date: str
    train_rows: int
    predicted_ranking: list[dict[str, Any]]
    actual_active_parts: list[int]
    actual_row_total: int
    hit_count: int
    hit_parts: list[int]
    miss_parts: list[int]
    actual_part_counts: dict[int, int] = field(default_factory=dict)
    all_scores: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "cutoff_date": self.cutoff_date,
            "target_date": self.target_date,
            "train_rows": self.train_rows,
            "predicted_ranking": self.predicted_ranking,
            "actual_active_parts": self.actual_active_parts,
            "actual_row_total": self.actual_row_total,
            "hit_count": self.hit_count,
            "hit_parts": self.hit_parts,
            "miss_parts": self.miss_parts,
            "actual_part_counts": self.actual_part_counts,
            "all_scores": self.all_scores,
        }


@dataclass
class BacktestSummary:
    """Aggregate summary statistics across all folds."""

    total_folds: int
    mean_hit_count: float
    min_hit_count: int
    max_hit_count: int
    mean_actual_row_total: float
    first_cutoff_date: str
    last_cutoff_date: str
    first_target_date: str
    last_target_date: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_folds": self.total_folds,
            "mean_hit_count": round(self.mean_hit_count, 4),
            "min_hit_count": self.min_hit_count,
            "max_hit_count": self.max_hit_count,
            "mean_actual_row_total": round(self.mean_actual_row_total, 4),
            "first_cutoff_date": self.first_cutoff_date,
            "last_cutoff_date": self.last_cutoff_date,
            "first_target_date": self.first_target_date,
            "last_target_date": self.last_target_date,
        }


@dataclass
class BacktestProvenance:
    """Provenance metadata for a backtest run."""

    run_id: str
    run_timestamp: str
    model_name: str
    dataset_variant: str
    dataset_fingerprint: str
    source_fingerprint: str
    config: dict[str, Any]
    dataset_row_count: int
    dataset_date_min: str
    dataset_date_max: str
    total_folds: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "model_name": self.model_name,
            "dataset_variant": self.dataset_variant,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "config": self.config,
            "dataset_row_count": self.dataset_row_count,
            "dataset_date_min": self.dataset_date_min,
            "dataset_date_max": self.dataset_date_max,
            "total_folds": self.total_folds,
        }


@dataclass
class BacktestResult:
    """Complete result of a rolling-origin backtest run."""

    provenance: BacktestProvenance
    folds: list[BacktestFold]
    summary: BacktestSummary
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "summary": self.summary.to_dict(),
            "folds": [f.to_dict() for f in self.folds],
            "artifacts": self.artifacts,
        }


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------
def generate_backtest_windows(
    n_rows: int,
    config: BacktestConfig,
) -> list[tuple[int, int]]:
    """Generate (cutoff_idx, target_idx) pairs for rolling-origin evaluation.

    Args:
        n_rows: Total number of rows in the dataset.
        config: Backtest configuration with min_train_rows, step, max_windows.

    Returns:
        List of (cutoff_index, target_index) tuples.

    Raises:
        ValueError: If dataset has fewer than min_train_rows + 1 rows.
    """
    min_needed = config.min_train_rows + 1
    if n_rows < min_needed:
        raise ValueError(
            f"Dataset has {n_rows} rows, need at least {min_needed} "
            f"(min_train_rows={config.min_train_rows} + 1 target row)"
        )

    first_cutoff = config.min_train_rows - 1
    last_cutoff = n_rows - 2

    windows = []
    for cutoff_idx in range(first_cutoff, last_cutoff + 1, config.step):
        windows.append((cutoff_idx, cutoff_idx + 1))

    if config.max_windows is not None and len(windows) > config.max_windows:
        windows = windows[-config.max_windows :]

    return windows


# ---------------------------------------------------------------------------
# Actual-parts extraction
# ---------------------------------------------------------------------------
def extract_actual_parts(row: pd.Series) -> tuple[list[int], int, dict[int, int]]:
    """Extract active part IDs, row total, and per-part counts from a target row.

    Args:
        row: A single row (pd.Series) from the dataset.

    Returns:
        Tuple of (sorted active part IDs, row_total, counts_dict).
        counts_dict maps part_id → count for parts with count > 0.
    """
    active: list[int] = []
    counts: dict[int, int] = {}
    total = 0
    for col in PART_COLUMNS:
        if col not in row.index:
            continue
        part_id = int(col.split("_")[1])
        val = int(row[col]) if pd.notna(row[col]) else 0
        total += val
        if val > 0 and part_id in VALID_PART_IDS:
            active.append(part_id)
            counts[part_id] = val
    return sorted(active), total, counts


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    scoring_fn: ScoringFunction,
    config: BacktestConfig,
    dataset_variant: str = "raw",
    dataset_fingerprint: str = "",
    source_fingerprint: str = "",
) -> BacktestResult:
    """Execute a rolling-origin backtest across the dataset.

    For each window (cutoff_idx, target_idx):
      1. Slice df[0:cutoff_idx+1] as the training fold
      2. Call scoring_fn(train_fold) to get PartScores
      3. Call rank_and_select(scores, k, model_name) to get RankedForecast
      4. Extract actual active parts from df[target_idx]
      5. Compare predicted vs actual, record BacktestFold

    Args:
        df: Working dataset DataFrame (will be sorted by date).
        scoring_fn: A callable matching the ScoringFunction protocol.
        config: Backtest parameters.
        dataset_variant: For provenance.
        dataset_fingerprint: For provenance.
        source_fingerprint: For provenance.

    Returns:
        BacktestResult with all folds, summary, and provenance.

    Raises:
        ValueError: If dates are not monotonically non-decreasing after sort.
    """
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

    # Defensive sort by date
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    # Verify monotonicity
    dates = df[DATE_COLUMN]
    if not dates.is_monotonic_increasing:
        raise ValueError("Dataset dates are not monotonically increasing after sort")

    # Generate windows
    windows = generate_backtest_windows(len(df), config)

    logger.info(
        "backtest_started",
        run_id=run_id,
        model=config.model_name,
        total_windows=len(windows),
        min_train_rows=config.min_train_rows,
        step=config.step,
    )

    folds: list[BacktestFold] = []
    for i, (cutoff_idx, target_idx) in enumerate(windows):
        # Slice training fold — no future data
        train_fold = df.iloc[: cutoff_idx + 1]
        target_row = df.iloc[target_idx]

        # Leakage guard: assert strict chronological separation
        train_max_date = train_fold[DATE_COLUMN].max()
        target_date = target_row[DATE_COLUMN]
        if train_max_date >= target_date:
            raise ValueError(
                f"Leakage detected at fold {i}: train_max_date={train_max_date} "
                f">= target_date={target_date}"
            )

        # Score and rank
        scores = scoring_fn(train_fold)
        forecast = rank_and_select(scores, k=config.k, model_name=config.model_name)

        # Extract actuals (with per-part counts for metrics)
        actual_parts, actual_total, actual_counts = extract_actual_parts(target_row)

        # Compare predicted vs actual
        predicted_ids = [r.part_id for r in forecast.rankings]
        predicted_set = set(predicted_ids)
        actual_set = set(actual_parts)
        hits = sorted(predicted_set & actual_set)
        misses = sorted(predicted_set - actual_set)

        fold = BacktestFold(
            fold_index=i,
            cutoff_date=str(train_max_date.date()),
            target_date=str(target_date.date()),
            train_rows=len(train_fold),
            predicted_ranking=[
                {"rank": r.rank, "part_id": r.part_id, "score": round(r.score, 6)}
                for r in forecast.rankings
            ],
            actual_active_parts=actual_parts,
            actual_row_total=actual_total,
            hit_count=len(hits),
            hit_parts=hits,
            miss_parts=misses,
            actual_part_counts=actual_counts,
            all_scores=[{"part_id": s.part_id, "score": round(s.score, 6)} for s in scores],
        )
        folds.append(fold)

        if (i + 1) % 500 == 0 or i == len(windows) - 1:
            logger.info(
                "backtest_progress",
                fold=i + 1,
                total=len(windows),
                latest_cutoff=fold.cutoff_date,
                latest_hits=fold.hit_count,
            )

    # Compute summary
    hit_counts = [f.hit_count for f in folds]
    row_totals = [f.actual_row_total for f in folds]
    summary = BacktestSummary(
        total_folds=len(folds),
        mean_hit_count=sum(hit_counts) / len(hit_counts),
        min_hit_count=min(hit_counts),
        max_hit_count=max(hit_counts),
        mean_actual_row_total=sum(row_totals) / len(row_totals),
        first_cutoff_date=folds[0].cutoff_date,
        last_cutoff_date=folds[-1].cutoff_date,
        first_target_date=folds[0].target_date,
        last_target_date=folds[-1].target_date,
    )

    # Build provenance
    provenance = BacktestProvenance(
        run_id=run_id,
        run_timestamp=run_timestamp,
        model_name=config.model_name,
        dataset_variant=dataset_variant,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
        config=config.to_dict(),
        dataset_row_count=len(df),
        dataset_date_min=str(dates.min().date()),
        dataset_date_max=str(dates.max().date()),
        total_folds=len(folds),
    )

    logger.info(
        "backtest_complete",
        run_id=run_id,
        model=config.model_name,
        total_folds=len(folds),
        mean_hit_count=round(summary.mean_hit_count, 2),
    )

    return BacktestResult(provenance=provenance, folds=folds, summary=summary)
