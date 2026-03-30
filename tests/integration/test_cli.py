"""Integration tests for the CLI entry point."""

import subprocess
import sys


class TestCliHelp:
    """Tests that the CLI is reachable and prints help."""

    def test_module_help_exits_zero(self) -> None:
        """Running ``python -m c5_forecasting --help`` must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "c5_forecasting" in result.stdout

    def test_health_check_exits_zero(self) -> None:
        """Running the health-check subcommand must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "health-check"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "Health check passed" in result.stdout

    def test_version_exits_zero(self) -> None:
        """Running the version subcommand must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout
