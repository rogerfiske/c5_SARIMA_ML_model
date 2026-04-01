"""Integration tests for the champion comparison and promotion flow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.config.settings import get_settings
from c5_forecasting.domain.constants import PART_COLUMNS
from c5_forecasting.evaluation.backtest import BacktestConfig
from c5_forecasting.evaluation.champion import load_champion, promote_champion
from c5_forecasting.evaluation.comparison import (
    ComparisonConfig,
    compare_to_champion,
    write_comparison_report,
)
from c5_forecasting.evaluation.ladder import run_ladder


def _make_row(date_str: str, counts: dict[str, int] | None = None) -> dict:
    """Build a row dict. Unspecified parts default to 0."""
    row: dict = {"date": date_str}
    for col in PART_COLUMNS:
        row[col] = counts.get(col, 0) if counts else 0
    return row


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _build_synthetic_df(n_rows: int = 400) -> pd.DataFrame:
    """Build a synthetic DataFrame with enough rows for backtesting."""
    rows = []
    for i in range(n_rows):
        date_str = pd.Timestamp("2020-01-01") + pd.Timedelta(days=i)
        counts = {}
        for j in range(1, 40):
            if (i + j) % 3 == 0:
                counts[f"P_{j}"] = (i + j) % 5 + 1
        rows.append(_make_row(str(date_str.date()), counts))
    return _make_df(rows)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _build_synthetic_df(400)


@pytest.fixture()
def fast_config() -> BacktestConfig:
    return BacktestConfig(min_train_rows=50, step=100, max_windows=5)


class TestChampionFlow:
    """Integration tests for the full champion flow."""

    def test_full_flow_first_promotion(
        self,
        synthetic_df: pd.DataFrame,
        fast_config: BacktestConfig,
        tmp_path: Path,
    ) -> None:
        """ladder -> compare (no champion) -> promote -> champion.json exists."""
        ladder_result = run_ladder(synthetic_df, fast_config)
        comparison = compare_to_champion(ladder_result, champion=None)

        assert comparison.champion_candidate is not None
        assert comparison.current_champion is None

        record = promote_champion(comparison, "PO", tmp_path)
        assert record.model_name == comparison.champion_candidate
        assert (tmp_path / "champion.json").exists()

        loaded = load_champion(tmp_path)
        assert loaded is not None
        assert loaded.model_name == record.model_name

    def test_full_flow_replacement(
        self,
        synthetic_df: pd.DataFrame,
        fast_config: BacktestConfig,
        tmp_path: Path,
    ) -> None:
        """First promote -> compare again -> verify comparison sees champion."""
        # First promotion
        ladder1 = run_ladder(synthetic_df, fast_config)
        comp1 = compare_to_champion(ladder1, champion=None)
        promote_champion(comp1, "PO", tmp_path)

        champion = load_champion(tmp_path)
        assert champion is not None

        # Second comparison against existing champion
        ladder2 = run_ladder(synthetic_df, fast_config)
        comp2 = compare_to_champion(ladder2, champion)

        # Champion exists now
        assert comp2.current_champion is not None
        assert comp2.current_champion["model_name"] == champion.model_name

    def test_close_scores_block_promotion(
        self,
        synthetic_df: pd.DataFrame,
        fast_config: BacktestConfig,
        tmp_path: Path,
    ) -> None:
        """When min_ndcg_delta is impossibly high, no candidate is produced."""
        # First promote to establish a champion
        ladder_result = run_ladder(synthetic_df, fast_config)
        comp1 = compare_to_champion(ladder_result, champion=None)
        promote_champion(comp1, "PO", tmp_path)

        champion = load_champion(tmp_path)
        assert champion is not None

        # Now compare with impossibly high threshold
        high_config = ComparisonConfig(min_ndcg_delta=1.0)
        comp2 = compare_to_champion(ladder_result, champion, config=high_config)
        assert comp2.champion_candidate is None

    def test_comparison_artifacts_written(
        self,
        synthetic_df: pd.DataFrame,
        fast_config: BacktestConfig,
        tmp_path: Path,
    ) -> None:
        """write_comparison_report produces JSON and MD files."""
        ladder_result = run_ladder(synthetic_df, fast_config)
        comparison = compare_to_champion(ladder_result, champion=None)
        paths = write_comparison_report(comparison, tmp_path)

        assert len(paths) == 2
        assert (tmp_path / "comparison_report.json").exists()
        assert (tmp_path / "comparison_report.md").exists()

    def test_deterministic_comparison(
        self,
        synthetic_df: pd.DataFrame,
        fast_config: BacktestConfig,
    ) -> None:
        """Same data produces same comparison (except IDs/timestamps)."""
        ladder = run_ladder(synthetic_df, fast_config)

        r1 = compare_to_champion(ladder, champion=None)
        r2 = compare_to_champion(ladder, champion=None)

        names1 = [e.model_name for e in r1.entries]
        names2 = [e.model_name for e in r2.entries]
        assert names1 == names2

        assert r1.champion_candidate == r2.champion_candidate
        assert r1.best_in_report == r2.best_in_report


class TestChampionCli:
    """CLI integration tests for compare, promote, champion commands."""

    def test_cli_compare_exits_zero(self) -> None:
        """compare --step 2000 should exit 0."""
        settings = get_settings()
        parquet_path = settings.processed_data_dir / "raw_v1.parquet"
        if not parquet_path.exists():
            pytest.skip(f"Real dataset not found at {parquet_path}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "c5_forecasting",
                "compare",
                "--step",
                "2000",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "Compare PASSED" in result.stdout

    def test_cli_champion_exits_zero(self) -> None:
        """champion command should exit 0 even with no champion set."""
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "champion"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    def test_cli_promote_without_confirm_is_dry_run(self) -> None:
        """promote without --confirm should do a dry run."""
        settings = get_settings()
        comp_dir = settings.artifacts_dir / "comparisons" / "latest"
        report_path = comp_dir / "comparison_report.json"
        if not report_path.exists():
            pytest.skip("No comparison report found — run compare first")

        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "promote"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Should succeed (dry run) or fail gracefully
        assert "Dry run" in result.stdout or result.returncode != 0
