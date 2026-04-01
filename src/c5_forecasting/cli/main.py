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


_BACKTEST_MODEL_OPTION = typer.Option(
    "frequency_baseline",
    help="Model name to run. Use 'ladder' command to run all models.",
)


@app.command(name="backtest")
def backtest_cmd(
    variant: str = _BACKTEST_VARIANT_OPTION,
    min_train_rows: int = _BACKTEST_MIN_TRAIN_OPTION,
    step: int = _BACKTEST_STEP_OPTION,
    max_windows: int | None = _BACKTEST_MAX_WINDOWS_OPTION,
    model: str = _BACKTEST_MODEL_OPTION,
) -> None:
    """Run rolling-origin backtesting with a specified baseline model."""
    import json

    import pandas as pd

    from c5_forecasting.data.dataset_builder import VALID_VARIANTS
    from c5_forecasting.evaluation.artifacts import write_backtest_artifacts
    from c5_forecasting.evaluation.backtest import BacktestConfig, run_backtest
    from c5_forecasting.evaluation.metric_report import write_metric_report
    from c5_forecasting.evaluation.metrics import compute_backtest_metrics
    from c5_forecasting.models.registry import get_scoring_function

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

    # Look up scoring function
    try:
        scoring_fn = get_scoring_function(model)
    except KeyError as e:
        typer.echo(str(e))
        raise typer.Exit(code=1) from None

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
        model_name=model,
    )

    typer.echo(
        f"Running backtest (model={model!r}, variant={variant!r}, step={step}, "
        f"min_train={min_train_rows}, max_windows={max_windows})..."
    )

    result = run_backtest(
        df=df,
        scoring_fn=scoring_fn,
        config=config,
        dataset_variant=variant,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
    )

    # Compute metrics
    fold_metrics, metric_summary = compute_backtest_metrics(result)

    # Write artifacts (with metrics)
    output_dir = settings.artifacts_dir / "backtests" / "latest"
    artifact_paths = write_backtest_artifacts(
        result, output_dir, fold_metrics=fold_metrics, metric_summary=metric_summary
    )
    metric_paths = write_metric_report(fold_metrics, metric_summary, result.provenance, output_dir)
    artifact_paths.extend(metric_paths)
    result.artifacts = artifact_paths

    # Print summary
    s = result.summary
    typer.echo(f"  Run ID:           {result.provenance.run_id}")
    typer.echo(f"  Model:            {result.provenance.model_name}")
    typer.echo(f"  Total folds:      {s.total_folds}")
    typer.echo(f"  Mean hit count:   {s.mean_hit_count:.2f}")
    typer.echo(f"  Min hit count:    {s.min_hit_count}")
    typer.echo(f"  Max hit count:    {s.max_hit_count}")
    typer.echo(f"  nDCG@20 mean:     {metric_summary.ndcg_20_mean:.4f}")
    typer.echo(f"  WR@20 mean:       {metric_summary.weighted_recall_20_mean:.4f}")
    typer.echo(f"  Brier mean:       {metric_summary.brier_score_mean:.4f}")
    typer.echo(f"  Cutoff range:     {s.first_cutoff_date} to {s.last_cutoff_date}")
    typer.echo(f"  Target range:     {s.first_target_date} to {s.last_target_date}")
    typer.echo(f"  Artifacts:        {artifact_paths}")
    typer.echo("Backtest PASSED.")


_LADDER_VARIANT_OPTION = typer.Option(
    None,
    help="Dataset variant: 'raw' or 'curated'. Defaults to C5_DATASET_VARIANT setting.",
)

_LADDER_MIN_TRAIN_OPTION = typer.Option(
    365,
    help="Minimum number of training rows before first cutoff.",
)

_LADDER_STEP_OPTION = typer.Option(
    1,
    help="Evaluate every Nth eligible cutoff (1=every day, 7=weekly).",
)

_LADDER_MAX_WINDOWS_OPTION = typer.Option(
    None,
    help="Maximum number of evaluation windows (None=unlimited).",
)


@app.command(name="ladder")
def ladder_cmd(
    variant: str = _LADDER_VARIANT_OPTION,
    min_train_rows: int = _LADDER_MIN_TRAIN_OPTION,
    step: int = _LADDER_STEP_OPTION,
    max_windows: int | None = _LADDER_MAX_WINDOWS_OPTION,
) -> None:
    """Run all baseline models and produce a ranked comparison."""
    import json

    import pandas as pd

    from c5_forecasting.data.dataset_builder import VALID_VARIANTS
    from c5_forecasting.evaluation.backtest import BacktestConfig
    from c5_forecasting.evaluation.ladder import run_ladder, write_ladder_artifacts

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
    )

    typer.echo(
        f"Running baseline ladder (variant={variant!r}, step={step}, "
        f"min_train={min_train_rows}, max_windows={max_windows})..."
    )

    ladder_result = run_ladder(
        df=df,
        config=config,
        dataset_variant=variant,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
    )

    # Write comparison artifacts
    output_dir = settings.artifacts_dir / "backtests" / "ladder"
    artifact_paths = write_ladder_artifacts(ladder_result, output_dir)
    ladder_result.artifacts = artifact_paths

    # Print ranked summary
    typer.echo("")
    typer.echo(f"  Best model:       {ladder_result.best_model}")
    typer.echo(f"  Models evaluated: {len(ladder_result.entries)}")
    typer.echo("")
    typer.echo("  Rank | Model                | nDCG@20 | WR@20  | Brier")
    typer.echo("  -----|----------------------|---------|--------|------")
    for i, entry in enumerate(ladder_result.entries):
        s = entry.metric_summary
        typer.echo(
            f"  {i + 1:4d} | {entry.model_name:<20s} "
            f"| {s.ndcg_20_mean:.4f}  | {s.weighted_recall_20_mean:.4f} "
            f"| {s.brier_score_mean:.4f}"
        )
    typer.echo("")
    typer.echo(f"  Artifacts:        {artifact_paths}")
    typer.echo("Ladder PASSED.")


_COMPARE_VARIANT_OPTION = typer.Option(
    None,
    help="Dataset variant: 'raw' or 'curated'. Defaults to C5_DATASET_VARIANT setting.",
)

_COMPARE_MIN_TRAIN_OPTION = typer.Option(
    365,
    help="Minimum number of training rows before first cutoff.",
)

_COMPARE_STEP_OPTION = typer.Option(
    1,
    help="Evaluate every Nth eligible cutoff (1=every day, 7=weekly).",
)

_COMPARE_MAX_WINDOWS_OPTION = typer.Option(
    None,
    help="Maximum number of evaluation windows (None=unlimited).",
)

_COMPARE_MIN_NDCG_DELTA_OPTION = typer.Option(
    0.01,
    help="Minimum nDCG@20 improvement over champion to be eligible.",
)

_COMPARE_MIN_WR_DELTA_OPTION = typer.Option(
    0.01,
    help="Minimum WR@20 improvement (reporting only, not gating).",
)

_COMPARE_MAX_BRIER_DELTA_OPTION = typer.Option(
    0.01,
    help="Maximum Brier improvement (reporting only, not gating).",
)


@app.command(name="compare")
def compare_cmd(
    variant: str = _COMPARE_VARIANT_OPTION,
    min_train_rows: int = _COMPARE_MIN_TRAIN_OPTION,
    step: int = _COMPARE_STEP_OPTION,
    max_windows: int | None = _COMPARE_MAX_WINDOWS_OPTION,
    min_ndcg_delta: float = _COMPARE_MIN_NDCG_DELTA_OPTION,
    min_wr_delta: float = _COMPARE_MIN_WR_DELTA_OPTION,
    max_brier_delta: float = _COMPARE_MAX_BRIER_DELTA_OPTION,
) -> None:
    """Run baseline ladder, compare against current champion, produce decision report."""
    import json

    import pandas as pd

    from c5_forecasting.data.dataset_builder import VALID_VARIANTS
    from c5_forecasting.evaluation.backtest import BacktestConfig
    from c5_forecasting.evaluation.champion import load_champion
    from c5_forecasting.evaluation.comparison import (
        ComparisonConfig,
        compare_to_champion,
        write_comparison_report,
    )
    from c5_forecasting.evaluation.ladder import run_ladder, write_ladder_artifacts

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
    )

    typer.echo(
        f"Running compare (variant={variant!r}, step={step}, "
        f"min_train={min_train_rows}, max_windows={max_windows})..."
    )

    # Run ladder
    ladder_result = run_ladder(
        df=df,
        config=config,
        dataset_variant=variant,
        dataset_fingerprint=dataset_fingerprint,
        source_fingerprint=source_fingerprint,
    )

    # Write ladder artifacts
    ladder_dir = settings.artifacts_dir / "backtests" / "ladder"
    write_ladder_artifacts(ladder_result, ladder_dir)

    # Load current champion
    champion = load_champion(settings.artifacts_dir)

    # Run comparison
    comp_config = ComparisonConfig(
        min_ndcg_delta=min_ndcg_delta,
        min_wr_delta=min_wr_delta,
        max_brier_delta=max_brier_delta,
    )
    comparison = compare_to_champion(ladder_result, champion, comp_config, dataset_variant=variant)

    # Write comparison report
    comp_dir = settings.artifacts_dir / "comparisons" / "latest"
    artifact_paths = write_comparison_report(comparison, comp_dir)

    # Print summary
    typer.echo("")
    if champion is not None:
        typer.echo(f"  Current champion: {champion.model_name} (nDCG={champion.ndcg_20_mean:.4f})")
    else:
        typer.echo("  Current champion: (none)")
    typer.echo("")
    typer.echo("  Rank | Model                | nDCG@20 | WR@20  | Brier  | Verdict")
    typer.echo("  -----|----------------------|---------|--------|--------|--------")
    for i, entry in enumerate(comparison.entries):
        s = entry.metric_summary
        typer.echo(
            f"  {i + 1:4d} | {entry.model_name:<20s} "
            f"| {s.ndcg_20_mean:.4f}  | {s.weighted_recall_20_mean:.4f} "
            f"| {s.brier_score_mean:.4f} | {entry.verdict.value}"
        )
    typer.echo("")
    if comparison.champion_candidate is not None:
        typer.echo(
            f"  Champion candidate: {comparison.champion_candidate} "
            f"— run 'promote --confirm' to approve."
        )
    else:
        typer.echo("  No champion candidate meets the threshold.")
    typer.echo(f"  Artifacts:        {artifact_paths}")
    typer.echo("Compare PASSED.")


_PROMOTE_COMPARISON_DIR_OPTION = typer.Option(
    None,
    help=("Path to comparison artifacts dir. Defaults to artifacts/comparisons/latest."),
)

_PROMOTE_APPROVER_OPTION = typer.Option(
    "PO",
    help="Name/role of approver.",
)

_PROMOTE_CONFIRM_OPTION = typer.Option(
    False,
    "--confirm",
    help="Required flag to actually write champion.json.",
)


@app.command(name="promote")
def promote_cmd(
    comparison_dir: Path = _PROMOTE_COMPARISON_DIR_OPTION,
    approver: str = _PROMOTE_APPROVER_OPTION,
    confirm: bool = _PROMOTE_CONFIRM_OPTION,
) -> None:
    """Promote the champion candidate from a comparison report."""
    import json

    from c5_forecasting.evaluation.champion import (
        ChampionRecord,
        save_champion,
    )

    settings = get_settings()

    if comparison_dir is None:
        comparison_dir = settings.artifacts_dir / "comparisons" / "latest"

    report_path = comparison_dir / "comparison_report.json"
    if not report_path.exists():
        typer.echo(f"Comparison report not found: {report_path}")
        typer.echo("Run 'compare' first to produce a comparison report.")
        raise typer.Exit(code=1)

    with open(report_path) as f:
        report_data = json.load(f)

    candidate_name = report_data.get("champion_candidate")
    if candidate_name is None:
        typer.echo("No champion candidate in this comparison report.")
        typer.echo("No model meets the minimum improvement threshold.")
        raise typer.Exit(code=1)

    # Find the candidate's metrics from entries
    candidate_metrics = None
    for entry in report_data["entries"]:
        if entry["model_name"] == candidate_name:
            candidate_metrics = entry["metrics"]
            break

    if candidate_metrics is None:
        typer.echo(f"Candidate {candidate_name!r} not found in report entries.")
        raise typer.Exit(code=1)

    typer.echo(f"  Candidate:   {candidate_name}")
    primary = candidate_metrics.get("primary", {})
    ndcg_mean = primary.get("ndcg_20", {}).get("mean", 0.0)
    wr_mean = primary.get("weighted_recall_20", {}).get("mean", 0.0)
    brier_mean = primary.get("brier_score", {}).get("mean", 0.0)
    typer.echo(f"  nDCG@20:     {ndcg_mean:.4f}")
    typer.echo(f"  WR@20:       {wr_mean:.4f}")
    typer.echo(f"  Brier:       {brier_mean:.4f}")

    if not confirm:
        typer.echo("")
        typer.echo("  Dry run — pass --confirm to actually promote.")
        typer.echo("Promote DRY-RUN.")
        return

    # Build and save ChampionRecord
    from datetime import UTC, datetime

    record = ChampionRecord(
        model_name=candidate_name,
        ndcg_20_mean=ndcg_mean,
        weighted_recall_20_mean=wr_mean,
        brier_score_mean=brier_mean,
        promoted_at=datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        promoted_from_comparison=report_data.get("comparison_id", ""),
        backtest_config=report_data.get("backtest_config", {}),
        dataset_variant=report_data.get("dataset_variant", ""),
        approver=approver,
    )
    champion_path = save_champion(record, settings.artifacts_dir)

    typer.echo("")
    typer.echo(f"  Champion promoted: {candidate_name}")
    typer.echo(f"  Approver:          {approver}")
    typer.echo(f"  Written to:        {champion_path}")
    typer.echo("Promote PASSED.")


@app.command(name="champion")
def champion_cmd() -> None:
    """Show the current champion model, or report that none is set."""
    from c5_forecasting.evaluation.champion import load_champion

    settings = get_settings()
    record = load_champion(settings.artifacts_dir)

    if record is None:
        typer.echo("No champion set.")
        typer.echo("Run 'compare' then 'promote --confirm' to establish one.")
        return

    typer.echo(f"  Model:       {record.model_name}")
    typer.echo(f"  nDCG@20:     {record.ndcg_20_mean:.4f}")
    typer.echo(f"  WR@20:       {record.weighted_recall_20_mean:.4f}")
    typer.echo(f"  Brier:       {record.brier_score_mean:.4f}")
    typer.echo(f"  Promoted at: {record.promoted_at}")
    typer.echo(f"  Approver:    {record.approver}")
    typer.echo(f"  Variant:     {record.dataset_variant}")
    typer.echo("Champion PASSED.")


if __name__ == "__main__":
    app()
