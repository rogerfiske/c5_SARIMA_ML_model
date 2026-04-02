"""Analyze temporal diversity of prediction exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def analyze_model_diversity(csv_path: Path, model_name: str) -> dict:
    """Analyze temporal diversity of predictions for a single model.

    Returns:
        Dictionary with diversity metrics.
    """
    df = pd.read_csv(csv_path)

    # Extract rank-1 predictions
    rank1_col = "pred-1" if "pred-1" in df.columns else "pred_01"
    rank1_preds = df[rank1_col].tolist()

    # Count unique parts in rank-1 position
    unique_rank1 = len(set(rank1_preds))

    # Count how many times rank-1 changes
    rank1_changes = sum(
        1 for i in range(1, len(rank1_preds)) if rank1_preds[i] != rank1_preds[i - 1]
    )
    rank1_change_rate = rank1_changes / (len(rank1_preds) - 1) if len(rank1_preds) > 1 else 0

    # Get all 20 predictions per row
    if "pred-1" in df.columns:
        pred_cols = [f"pred-{i}" for i in range(1, 21)]
    else:
        pred_cols = [f"pred_{i:02d}" for i in range(1, 21)]

    # Count how many times the full top-20 list changes
    full_changes = 0
    for i in range(1, len(df)):
        prev_set = set(df.iloc[i - 1][pred_cols].tolist())
        curr_set = set(df.iloc[i][pred_cols].tolist())
        if prev_set != curr_set:
            full_changes += 1

    full_change_rate = full_changes / (len(df) - 1) if len(df) > 1 else 0

    # Most common rank-1 prediction
    from collections import Counter

    rank1_counts = Counter(rank1_preds)
    most_common_rank1 = rank1_counts.most_common(1)[0] if rank1_counts else (None, 0)

    # Calculate concentration (what % of predictions are the most common part)
    concentration = (most_common_rank1[1] / len(rank1_preds)) * 100 if rank1_preds else 0

    return {
        "model": model_name,
        "total_predictions": len(df),
        "unique_rank1_parts": unique_rank1,
        "rank1_change_count": rank1_changes,
        "rank1_change_rate": rank1_change_rate,
        "full_top20_change_count": full_changes,
        "full_top20_change_rate": full_change_rate,
        "most_common_rank1_part": most_common_rank1[0],
        "most_common_rank1_count": most_common_rank1[1],
        "rank1_concentration_pct": concentration,
        "first_date": (
            df.iloc[0]["M/D/YYYY"] if "M/D/YYYY" in df.columns else df.iloc[0]["target_date"]
        ),
        "last_date": (
            df.iloc[-1]["M/D/YYYY"] if "M/D/YYYY" in df.columns else df.iloc[-1]["target_date"]
        ),
    }


def main():
    """Run diversity analysis on all prediction exports."""
    base_path = Path("data/raw")

    models = [
        ("frequency_baseline", "c5_predictions.csv"),
        ("recency_weighted", "c5_predictions_recency_weighted_simple.csv"),
        ("rolling_window", "c5_predictions_rolling_window_simple.csv"),
    ]

    # Check for ensemble models (might not be ready yet)
    if (base_path / "c5_predictions_ensemble_rank_avg_simple.csv").exists():
        models.append(("ensemble_rank_avg", "c5_predictions_ensemble_rank_avg_simple.csv"))
    if (base_path / "c5_predictions_ensemble_weighted_simple.csv").exists():
        models.append(("ensemble_weighted", "c5_predictions_ensemble_weighted_simple.csv"))

    results = []
    for model_name, filename in models:
        csv_path = base_path / filename
        if csv_path.exists():
            print(f"\nAnalyzing {model_name}...")
            result = analyze_model_diversity(csv_path, model_name)
            results.append(result)

            # Print summary
            print(f"  Total predictions: {result['total_predictions']}")
            print(f"  Unique parts in rank-1: {result['unique_rank1_parts']}")
            print(
                f"  Rank-1 changes: {result['rank1_change_count']} "
                f"({result['rank1_change_rate']:.1%})"
            )
            print(
                f"  Full top-20 changes: {result['full_top20_change_count']} "
                f"({result['full_top20_change_rate']:.1%})"
            )
            print(
                f"  Most common rank-1: Part {result['most_common_rank1_part']} "
                f"({result['most_common_rank1_count']} times, "
                f"{result['rank1_concentration_pct']:.1f}%)"
            )
        else:
            print(f"\nSkipping {model_name} (file not found: {csv_path})")

    # Create comparative summary
    print("\n" + "=" * 80)
    print("COMPARATIVE SUMMARY")
    print("=" * 80)
    print(
        f"\n{'Model':<25} {'Unique R1':<12} {'R1 Chg %':<12} {'Top20 Chg %':<14} {'R1 Conc %':<12}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['model']:<25} {r['unique_rank1_parts']:<12} "
            f"{r['rank1_change_rate']:>10.1%}  "
            f"{r['full_top20_change_rate']:>12.1%}  "
            f"{r['rank1_concentration_pct']:>10.1f}%"
        )

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("- Unique R1: Number of distinct parts that appeared in rank-1 position")
    print("- R1 Chg %: How often the rank-1 prediction changes from one day to the next")
    print("- Top20 Chg %: How often the full top-20 set changes from one day to the next")
    print("- R1 Conc %: Percentage of predictions where the most common part was ranked #1")
    print("\nHigher change rates = more temporally responsive/dynamic model")
    print("Lower concentration = less dominated by a single part")


if __name__ == "__main__":
    main()
