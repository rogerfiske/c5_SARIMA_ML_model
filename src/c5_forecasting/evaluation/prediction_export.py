"""Export historical daily predictions to CSV format."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import structlog

from c5_forecasting.evaluation.backtest import BacktestResult
from c5_forecasting.evaluation.metrics import FoldMetrics

logger = structlog.get_logger()


def _format_actual_parts(actual_part_counts: dict[int, int]) -> str:
    """Convert actual part counts dict to pipe-delimited string of IDs.

    Args:
        actual_part_counts: Mapping of part ID to count for active parts.

    Returns:
        Pipe-delimited string of part IDs sorted ascending (e.g., "5|12|27|30").
    """
    if not actual_part_counts:
        return ""
    sorted_parts = sorted(actual_part_counts.keys())
    return "|".join(str(p) for p in sorted_parts)


def write_daily_predictions_csv(
    result: BacktestResult,
    fold_metrics: list[FoldMetrics],
    output_path: Path,
) -> Path:
    """Write daily predictions to wide-format CSV.

    Schema (55 columns):
      - Metadata (5): target_date, cutoff_date, model_name, dataset_variant, train_rows
      - Predictions (20): pred_01 through pred_20
      - Scores (20): score_01 through score_20
      - Actuals (2): actual_nonzero_parts, actual_hit_count_top20
      - Metrics (5): ndcg_20, weighted_recall_20, precision_20, recall_20, jaccard_20

    Predictions are generated using strict rolling-origin backtest logic with no future
    leakage. Each row represents one target date.

    Args:
        result: Backtest result containing all fold predictions.
        fold_metrics: Per-fold metrics computed from backtest result.
        output_path: Destination path for CSV file.

    Returns:
        Path to written CSV file.

    Raises:
        ValueError: If any prediction contains part ID 0 (invalid invariant).
    """
    if len(result.folds) != len(fold_metrics):
        raise ValueError(
            f"Fold count mismatch: {len(result.folds)} folds vs {len(fold_metrics)} metrics"
        )

    # Build rows: one row per target date
    rows = []
    for fold, fold_metric in zip(result.folds, fold_metrics, strict=True):
        # Extract predictions and scores
        pred_ids = []
        pred_scores = []
        for entry in fold.predicted_ranking:
            part_id = entry["part_id"]
            score = entry["score"]

            # CRITICAL: Validate no zeros (domain invariant)
            if part_id == 0:
                raise ValueError(
                    f"Invalid prediction: part_id=0 in fold {fold.fold_index} "
                    f"(target_date={fold.target_date}). Part ID 0 is forbidden."
                )

            pred_ids.append(part_id)
            pred_scores.append(score)

        # Ensure exactly 20 predictions
        if len(pred_ids) != 20:
            raise ValueError(
                f"Expected 20 predictions, got {len(pred_ids)} "
                f"in fold {fold.fold_index} (target_date={fold.target_date})"
            )

        # Build row dict
        row = {
            # Metadata
            "target_date": fold.target_date,
            "cutoff_date": fold.cutoff_date,
            "model_name": result.provenance.model_name,
            "dataset_variant": result.provenance.dataset_variant,
            "train_rows": fold.train_rows,
        }

        # Predictions (pred_01 through pred_20)
        for i, part_id in enumerate(pred_ids, 1):
            row[f"pred_{i:02d}"] = part_id

        # Scores (score_01 through score_20)
        for i, score in enumerate(pred_scores, 1):
            row[f"score_{i:02d}"] = score

        # Actuals
        row["actual_nonzero_parts"] = _format_actual_parts(fold.actual_part_counts)
        row["actual_hit_count_top20"] = fold.hit_count

        # Metrics
        row["ndcg_20"] = fold_metric.ndcg_20
        row["weighted_recall_20"] = fold_metric.weighted_recall_20
        row["precision_20"] = fold_metric.precision_20
        row["recall_20"] = fold_metric.recall_20
        row["jaccard_20"] = fold_metric.jaccard_20

        rows.append(row)

    # Create DataFrame with explicit column ordering
    columns = (
        # Metadata
        ["target_date", "cutoff_date", "model_name", "dataset_variant", "train_rows"]
        # Predictions
        + [f"pred_{i:02d}" for i in range(1, 21)]
        # Scores
        + [f"score_{i:02d}" for i in range(1, 21)]
        # Actuals
        + ["actual_nonzero_parts", "actual_hit_count_top20"]
        # Metrics
        + ["ndcg_20", "weighted_recall_20", "precision_20", "recall_20", "jaccard_20"]
    )

    df = pd.DataFrame(rows, columns=columns)

    # Write to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(
        "daily_predictions_csv_written",
        path=str(output_path),
        row_count=len(df),
        first_target_date=df["target_date"].iloc[0] if len(df) > 0 else None,
        last_target_date=df["target_date"].iloc[-1] if len(df) > 0 else None,
    )

    return output_path


def write_timestamped_export(
    df: pd.DataFrame,
    artifacts_dir: Path,
    basename: str,
) -> Path:
    """Write timestamped export to artifacts/exports/ directory.

    Args:
        df: DataFrame to export.
        artifacts_dir: Base artifacts directory.
        basename: Base filename (e.g., "c5_predictions.csv").

    Returns:
        Path to written timestamped file.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    export_dir = artifacts_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{timestamp}_{basename}"
    path = export_dir / filename
    df.to_csv(path, index=False)

    logger.info("timestamped_export_written", path=str(path), row_count=len(df))
    return path
