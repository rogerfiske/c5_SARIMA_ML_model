"""CLI entry point for the c5_forecasting platform.

Usage:
    python -m c5_forecasting --help
    forecasting --help          (if installed via Poetry)
"""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
