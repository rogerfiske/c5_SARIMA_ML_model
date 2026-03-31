"""Canary forecast pipeline — end-to-end from dataset to ranked artifacts.

This module orchestrates the full canary forecast flow for Story 1.5:
1. Load a working dataset (Parquet)
2. Score all parts using the frequency baseline
3. Rank and select top-K with deterministic tie-breaking
4. Validate the output
5. Write forecast artifacts (CSV, JSON, Markdown)
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from c5_forecasting.domain.constants import TOP_K
from c5_forecasting.models.baseline import MODEL_NAME, compute_frequency_scores
from c5_forecasting.ranking.ranker import RankedForecast, rank_and_select

logger = structlog.get_logger(__name__)


@dataclass
class ForecastProvenance:
    """Provenance metadata for a forecast run."""

    run_id: str
    run_timestamp: str
    model_name: str
    dataset_variant: str
    dataset_fingerprint: str
    source_fingerprint: str
    config_fingerprint: str
    k: int
    dataset_row_count: int
    dataset_date_min: str
    dataset_date_max: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "model_name": self.model_name,
            "dataset_variant": self.dataset_variant,
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "k": self.k,
            "dataset_row_count": self.dataset_row_count,
            "dataset_date_min": self.dataset_date_min,
            "dataset_date_max": self.dataset_date_max,
        }


@dataclass
class ForecastResult:
    """Complete result of a forecast run."""

    forecast: RankedForecast
    provenance: ForecastProvenance
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "rankings": [
                {"rank": r.rank, "part_id": r.part_id, "score": round(r.score, 6)}
                for r in self.forecast.rankings
            ],
            "artifacts": self.artifacts,
        }


def run_canary_forecast(
    dataset_path: Path,
    dataset_variant: str,
    dataset_fingerprint: str,
    source_fingerprint: str,
    output_dir: Path,
) -> ForecastResult:
    """Run the full canary forecast pipeline.

    Args:
        dataset_path: Path to the working dataset Parquet file.
        dataset_variant: Name of the dataset variant (e.g. "raw").
        dataset_fingerprint: SHA-256 of the dataset Parquet file.
        source_fingerprint: SHA-256 of the original raw CSV.
        output_dir: Directory to write forecast artifacts.

    Returns:
        A :class:`ForecastResult` with the ranked forecast and provenance.
    """
    run_id = str(uuid.uuid4())
    run_timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")

    # Load dataset
    df = pd.read_parquet(dataset_path)

    # Compute config fingerprint (hash of variant + model name)
    config_str = f"variant={dataset_variant};model={MODEL_NAME};k={TOP_K}"
    config_fingerprint = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    # Score and rank
    scores = compute_frequency_scores(df)
    forecast = rank_and_select(scores, k=TOP_K, model_name=MODEL_NAME)

    # Build provenance
    date_col = df["date"]
    provenance = ForecastProvenance(
        run_id=run_id,
        run_timestamp=run_timestamp,
        model_name=MODEL_NAME,
        dataset_variant=dataset_variant,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
        config_fingerprint=config_fingerprint,
        k=TOP_K,
        dataset_row_count=len(df),
        dataset_date_min=str(date_col.min().date()) if len(df) > 0 else "",
        dataset_date_max=str(date_col.max().date()) if len(df) > 0 else "",
    )

    # Write artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    result = ForecastResult(forecast=forecast, provenance=provenance)

    csv_path = _write_csv_artifact(result, output_dir)
    json_path = _write_json_artifact(result, output_dir)
    md_path = _write_markdown_artifact(result, output_dir)
    result.artifacts = [str(csv_path), str(json_path), str(md_path)]

    logger.info(
        "canary_forecast_complete",
        run_id=run_id,
        model=MODEL_NAME,
        top_3=[r.part_id for r in forecast.rankings[:3]],
        artifacts=len(result.artifacts),
    )

    return result


def _write_csv_artifact(result: ForecastResult, output_dir: Path) -> Path:
    """Write the ranked forecast as a CSV file."""
    path = output_dir / "forecast.csv"
    rows = [
        {"rank": r.rank, "part_id": r.part_id, "score": round(r.score, 6)}
        for r in result.forecast.rankings
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    logger.info("artifact_written", format="csv", path=str(path))
    return path


def _write_json_artifact(result: ForecastResult, output_dir: Path) -> Path:
    """Write the full forecast package as JSON."""
    path = output_dir / "forecast.json"
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("artifact_written", format="json", path=str(path))
    return path


def _write_markdown_artifact(result: ForecastResult, output_dir: Path) -> Path:
    """Write a human-readable Markdown summary."""
    path = output_dir / "forecast.md"
    prov = result.provenance
    lines = [
        f"# Forecast: {prov.model_name}",
        "",
        f"**Run ID:** {prov.run_id}",
        f"**Timestamp:** {prov.run_timestamp}",
        f"**Dataset variant:** {prov.dataset_variant}",
        f"**Dataset fingerprint:** {prov.dataset_fingerprint[:16]}...",
        f"**Source fingerprint:** {prov.source_fingerprint[:16]}...",
        f"**Config fingerprint:** {prov.config_fingerprint}",
        f"**Dataset rows:** {prov.dataset_row_count}",
        f"**Date range:** {prov.dataset_date_min} to {prov.dataset_date_max}",
        "",
        f"## Top {prov.k} Ranked Parts",
        "",
        "| Rank | Part ID | Score |",
        "|------|---------|-------|",
    ]
    for r in result.forecast.rankings:
        lines.append(f"| {r.rank} | {r.part_id} | {r.score:.6f} |")

    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info("artifact_written", format="markdown", path=str(path))
    return path
