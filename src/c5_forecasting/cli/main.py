"""CLI entry point for the c5_forecasting platform.

Usage:
    python -m c5_forecasting --help
    forecasting --help          (if installed via Poetry)
"""

from __future__ import annotations

from pathlib import Path

import typer

from c5_forecasting import __version__
from c5_forecasting.config.logging import configure_logging
from c5_forecasting.config.settings import get_settings

app = typer.Typer(
    name="forecasting",
    help="c5_forecasting: next-event ranking forecast platform.",
    no_args_is_help=True,
)


@app.callback()
def main(
    log_level: str = typer.Option("INFO", envvar="C5_LOG_LEVEL", help="Logging level"),
) -> None:
    """Initialize logging for all subcommands."""
    configure_logging(log_level)


@app.command()
def version() -> None:
    """Print the current package version."""
    typer.echo(f"c5_forecasting {__version__}")


@app.command(name="health-check")
def health_check() -> None:
    """Verify that the platform is installed and configured correctly."""
    settings = get_settings()
    typer.echo(f"c5_forecasting {__version__}")
    typer.echo(f"  data_dir:      {settings.data_dir}")
    typer.echo(f"  artifacts_dir: {settings.artifacts_dir}")
    typer.echo(f"  configs_dir:   {settings.configs_dir}")
    typer.echo(f"  log_level:     {settings.log_level}")
    typer.echo("Health check passed.")


_VALIDATE_RAW_CSV_OPTION = typer.Option(
    None,
    help="Path to raw CSV. Defaults to data/raw/c5_aggregated_matrix.csv",
)


@app.command(name="validate-raw")
def validate_raw(
    csv_path: Path = _VALIDATE_RAW_CSV_OPTION,
) -> None:
    """Validate the raw aggregated matrix CSV and emit a structured report."""
    from c5_forecasting.data.report import write_validation_report
    from c5_forecasting.data.validation import validate_raw_dataset

    settings = get_settings()

    if csv_path is None:
        csv_path = settings.raw_data_dir / "c5_aggregated_matrix.csv"

    typer.echo(f"Validating: {csv_path}")
    result = validate_raw_dataset(csv_path)

    report_dir = settings.artifacts_dir / "manifests"
    report_path = write_validation_report(result, report_dir)

    typer.echo(f"  Source SHA-256: {result.source_sha256}")
    typer.echo(f"  Rows:          {result.row_count}")
    typer.echo(f"  Date range:    {result.date_min} to {result.date_max}")
    typer.echo(f"  Missing dates: {result.missing_date_count}")
    typer.echo(f"  Duplicates:    {result.duplicate_date_count}")
    typer.echo(f"  Errors:        {len(result.errors)}")
    typer.echo(f"  Warnings:      {len(result.warnings)}")
    typer.echo(f"  Report:        {report_path}")

    if result.is_valid:
        typer.echo("Validation PASSED.")
    else:
        typer.echo("Validation FAILED.")
        for err in result.errors:
            typer.echo(f"  ERROR: {err}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
