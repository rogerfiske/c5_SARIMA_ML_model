"""Validation report generation and persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import structlog

from c5_forecasting.data.validation import ValidationResult

logger = structlog.get_logger(__name__)


def write_validation_report(
    result: ValidationResult,
    output_dir: Path,
) -> Path:
    """Write a validation result as a JSON report to the artifacts directory.

    The report filename includes a timestamp so that successive runs do not
    overwrite each other (per NFR12).

    Args:
        result: The completed validation result.
        output_dir: Directory to write the report into.

    Returns:
        Path to the written JSON report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    filename = f"validation_report_{timestamp}.json"
    output_path = output_dir / filename

    report = {
        "report_type": "raw_dataset_validation",
        "generated_at": timestamp,
        **result.to_dict(),
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("validation_report_written", path=str(output_path))
    return output_path
