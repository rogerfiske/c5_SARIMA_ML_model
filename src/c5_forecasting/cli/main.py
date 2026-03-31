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


_ANNOTATE_CSV_OPTION = typer.Option(
    None,
    help="Path to raw CSV. Defaults to data/raw/c5_aggregated_matrix.csv",
)

_ANNOTATE_CONFIG_OPTION = typer.Option(
    None,
    help="Path to event annotations YAML. Defaults to configs/datasets/event_annotations.yaml",
)


@app.command(name="annotate-dataset")
def annotate_dataset_cmd(
    csv_path: Path = _ANNOTATE_CSV_OPTION,
    config_path: Path = _ANNOTATE_CONFIG_OPTION,
) -> None:
    """Annotate the raw dataset with event labels and anomaly flags."""
    from c5_forecasting.data.annotation import annotate_dataset, load_annotation_config
    from c5_forecasting.data.loader import load_raw_csv
    from c5_forecasting.data.validation import validate_raw_dataset

    settings = get_settings()

    if csv_path is None:
        csv_path = settings.raw_data_dir / "c5_aggregated_matrix.csv"
    if config_path is None:
        config_path = settings.configs_dir / "datasets" / "event_annotations.yaml"

    # Validate first
    typer.echo(f"Validating: {csv_path}")
    val_result = validate_raw_dataset(csv_path)
    if not val_result.is_valid:
        typer.echo("Validation FAILED — cannot annotate invalid dataset.")
        for err in val_result.errors:
            typer.echo(f"  ERROR: {err}")
        raise typer.Exit(code=1)

    # Load and annotate
    typer.echo(f"Loading annotation config: {config_path}")
    config = load_annotation_config(config_path)

    df = load_raw_csv(csv_path)
    df_annotated, ann_result = annotate_dataset(df, config)

    typer.echo(f"  Rows:                {ann_result.row_count}")
    typer.echo(f"  Standard days:       {ann_result.standard_count}")
    typer.echo(f"  Reviewed exceptions: {ann_result.reviewed_exception_count}")
    typer.echo(f"  Unreviewed exceptions: {ann_result.unreviewed_exception_count}")
    typer.echo(f"  Columns added:       {ann_result.annotation_columns_added}")

    if ann_result.warnings:
        typer.echo(f"  Warnings:            {len(ann_result.warnings)}")
        for w in ann_result.warnings[:10]:
            typer.echo(f"    WARN: {w}")

    typer.echo("Annotation PASSED.")


_BUILD_CSV_OPTION = typer.Option(
    None,
    help="Path to raw CSV. Defaults to data/raw/c5_aggregated_matrix.csv",
)

_BUILD_CONFIG_OPTION = typer.Option(
    None,
    help="Path to event annotations YAML. Defaults to configs/datasets/event_annotations.yaml",
)

_BUILD_VARIANT_OPTION = typer.Option(
    None,
    help="Dataset variant: 'raw' or 'curated'. Defaults to C5_DATASET_VARIANT setting.",
)


@app.command(name="build-dataset")
def build_dataset_cmd(
    csv_path: Path = _BUILD_CSV_OPTION,
    config_path: Path = _BUILD_CONFIG_OPTION,
    variant: str = _BUILD_VARIANT_OPTION,
) -> None:
    """Build a working dataset (Parquet) from the validated raw CSV."""
    from c5_forecasting.data.annotation import load_annotation_config
    from c5_forecasting.data.dataset_builder import (
        VALID_VARIANTS,
        build_curated_dataset,
        build_raw_dataset,
        write_manifest,
    )

    settings = get_settings()

    if csv_path is None:
        csv_path = settings.raw_data_dir / "c5_aggregated_matrix.csv"
    if config_path is None:
        config_path = settings.configs_dir / "datasets" / "event_annotations.yaml"
    if variant is None:
        variant = settings.dataset_variant

    if variant not in VALID_VARIANTS:
        typer.echo(f"Invalid variant {variant!r}. Must be one of: {sorted(VALID_VARIANTS)}")
        raise typer.Exit(code=1)

    typer.echo(f"Building {variant!r} dataset from: {csv_path}")
    annotation_config = load_annotation_config(config_path)

    output_dir = settings.processed_data_dir
    manifest_dir = settings.artifacts_dir / "manifests"

    if variant == "raw":
        df, manifest = build_raw_dataset(csv_path, annotation_config, output_dir)
    else:
        df, manifest = build_curated_dataset(csv_path, annotation_config, output_dir)

    manifest_path = write_manifest(manifest, manifest_dir)

    typer.echo(f"  Variant:       {manifest.variant_name}")
    typer.echo(f"  Rows:          {manifest.row_count}")
    typer.echo(f"  Columns:       {manifest.column_count}")
    typer.echo(f"  Date range:    {manifest.date_min} to {manifest.date_max}")
    typer.echo(f"  Source SHA-256: {manifest.source_sha256}")
    typer.echo(f"  Output SHA-256: {manifest.output_sha256}")
    typer.echo(f"  Output:        {manifest.output_path}")
    typer.echo(f"  Manifest:      {manifest_path}")
    typer.echo(f"  Transforms:    {manifest.transform_steps}")
    typer.echo("Build PASSED.")


_FORECAST_VARIANT_OPTION = typer.Option(
    None,
    help="Dataset variant: 'raw' or 'curated'. Defaults to C5_DATASET_VARIANT setting.",
)


@app.command(name="forecast-next-day")
def forecast_next_day_cmd(
    variant: str = _FORECAST_VARIANT_OPTION,
) -> None:
    """Run the canary next-day top-20 forecast using the frequency baseline."""
    from c5_forecasting.data.dataset_builder import VALID_VARIANTS
    from c5_forecasting.pipelines.forecast import run_canary_forecast

    settings = get_settings()

    if variant is None:
        variant = settings.dataset_variant
    if variant not in VALID_VARIANTS:
        typer.echo(f"Invalid variant {variant!r}. Must be one of: {sorted(VALID_VARIANTS)}")
        raise typer.Exit(code=1)

    # Resolve dataset path
    parquet_path = settings.processed_data_dir / f"{variant}_v1.parquet"
    if not parquet_path.exists():
        typer.echo(f"Dataset not found: {parquet_path}")
        typer.echo("Run 'build-dataset' first to create the working dataset.")
        raise typer.Exit(code=1)

    # Read manifest for fingerprints
    manifest_path = settings.artifacts_dir / "manifests" / f"{variant}_v1_manifest.json"
    dataset_fingerprint = ""
    source_fingerprint = ""
    if manifest_path.exists():
        import json

        with open(manifest_path) as f:
            manifest_data = json.load(f)
        dataset_fingerprint = manifest_data.get("output_sha256", "")
        source_fingerprint = manifest_data.get("source_sha256", "")

    # Run forecast
    output_dir = settings.artifacts_dir / "runs" / "latest"
    typer.echo(f"Running canary forecast (variant={variant!r})...")

    result = run_canary_forecast(
        dataset_path=parquet_path,
        dataset_variant=variant,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
        output_dir=output_dir,
    )

    prov = result.provenance
    typer.echo(f"  Run ID:        {prov.run_id}")
    typer.echo(f"  Model:         {prov.model_name}")
    typer.echo(f"  Variant:       {prov.dataset_variant}")
    typer.echo(f"  Dataset rows:  {prov.dataset_row_count}")
    typer.echo("  Top 5:")
    for r in result.forecast.rankings[:5]:
        typer.echo(f"    #{r.rank}: P_{r.part_id} (score={r.score:.6f})")
    typer.echo(f"  Artifacts:     {result.artifacts}")
    typer.echo("Forecast PASSED.")


_BACKTEST_VARIANT_OPTION = typer.Option(
    None,
    help="Dataset variant: 'raw' or 'curated'. Defaults to C5_DATASET_VARIANT setting.",
)

_BACKTEST_MIN_TRAIN_OPTION = typer.Option(
    365,
    help="Minimum number of training rows before first cutoff.",
)

_BACKTEST_STEP_OPTION = typer.Option(
    1,
    help="Evaluate every Nth eligible cutoff (1=every day, 7=weekly).",
)

_BACKTEST_MAX_WINDOWS_OPTION = typer.Option(
    None,
    help="Maximum number of evaluation windows (None=unlimited).",
)


@app.command(name="backtest")
def backtest_cmd(
    variant: str = _BACKTEST_VARIANT_OPTION,
    min_train_rows: int = _BACKTEST_MIN_TRAIN_OPTION,
    step: int = _BACKTEST_STEP_OPTION,
    max_windows: int | None = _BACKTEST_MAX_WINDOWS_OPTION,
) -> None:
    """Run rolling-origin backtesting with the frequency baseline."""
    import json

    import pandas as pd

    from c5_forecasting.data.dataset_builder import VALID_VARIANTS
    from c5_forecasting.evaluation.artifacts import write_backtest_artifacts
    from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
    from c5_forecasting.models.baseline import MODEL_NAME, compute_frequency_scores

    settings = get_settings()

    if variant is None:
        variant = settings.dataset_variant
    if variant not in VALID_VARIANTS:
        typer.echo(f"Invalid variant {variant!r}. Must be one of: {sorted(VALID_VARIANTS)}")
        raise typer.Exit(code=1)

    parquet_path = settings.processed_data_dir / f"{variant}_v1.parquet"
    if not parquet_path.exists():
        typer.echo(f"Dataset not found: {parquet_path}")
        typer.echo("Run 'build-dataset' first to create the working dataset.")
        raise typer.Exit(code=1)

    # Load dataset
    df = pd.read_parquet(parquet_path)

    # Read manifest for fingerprints
    manifest_path = settings.artifacts_dir / "manifests" / f"{variant}_v1_manifest.json"
    dataset_fingerprint = ""
    source_fingerprint = ""
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        dataset_fingerprint = manifest_data.get("output_sha256", "")
        source_fingerprint = manifest_data.get("source_sha256", "")

    config = BacktestConfig(
        min_train_rows=min_train_rows,
        step=step,
        max_windows=max_windows,
        model_name=MODEL_NAME,
    )

    typer.echo(
        f"Running backtest (variant={variant!r}, step={step}, "
        f"min_train={min_train_rows}, max_windows={max_windows})..."
    )

    result = run_backtest(
        df=df,
        scoring_fn=compute_frequency_scores,
        config=config,
        dataset_variant=variant,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
    )

    # Write artifacts
    output_dir = settings.artifacts_dir / "backtests" / "latest"
    artifact_paths = write_backtest_artifacts(result, output_dir)
    result.artifacts = artifact_paths

    # Print summary
    s = result.summary
    typer.echo(f"  Run ID:           {result.provenance.run_id}")
    typer.echo(f"  Model:            {result.provenance.model_name}")
    typer.echo(f"  Total folds:      {s.total_folds}")
    typer.echo(f"  Mean hit count:   {s.mean_hit_count:.2f}")
    typer.echo(f"  Min hit count:    {s.min_hit_count}")
    typer.echo(f"  Max hit count:    {s.max_hit_count}")
    typer.echo(f"  Cutoff range:     {s.first_cutoff_date} to {s.last_cutoff_date}")
    typer.echo(f"  Target range:     {s.first_target_date} to {s.last_target_date}")
    typer.echo(f"  Artifacts:        {artifact_paths}")
    typer.echo("Backtest PASSED.")


if __name__ == "__main__":
    app()
