"""Integration tests for the canary forecast pipeline against the real dataset."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from c5_forecasting.data.annotation import load_annotation_config
from c5_forecasting.data.dataset_builder import build_raw_dataset
from c5_forecasting.domain.constants import TOP_K, VALID_PART_IDS
from c5_forecasting.pipelines.forecast import run_canary_forecast

_RAW_CSV = Path("data/raw/c5_aggregated_matrix.csv")
_CONFIG_YAML = Path("configs/datasets/event_annotations.yaml")


@pytest.fixture()
def real_dataset(tmp_path: Path) -> tuple[Path, str, str]:
    """Build the real raw dataset and return (parquet_path, output_sha, source_sha)."""
    if not _RAW_CSV.exists():
        pytest.skip("Real CSV not available")
    if not _CONFIG_YAML.exists():
        pytest.skip("Annotation config not available")

    config = load_annotation_config(_CONFIG_YAML)
    _, manifest = build_raw_dataset(_RAW_CSV, config, tmp_path)
    parquet_path = tmp_path / "raw_v1.parquet"
    return parquet_path, manifest.output_sha256, manifest.source_sha256


class TestRealForecast:
    """Integration tests for the canary forecast on the real dataset."""

    def test_forecast_produces_20_parts(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """Forecast must produce exactly 20 ranked parts."""
        pq, out_sha, src_sha = real_dataset
        result = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out")
        assert len(result.forecast.rankings) == TOP_K

    def test_all_ids_in_valid_range(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """All predicted part IDs must be in 1..39."""
        pq, out_sha, src_sha = real_dataset
        result = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out")
        for r in result.forecast.rankings:
            assert r.part_id in VALID_PART_IDS, f"Invalid part ID: {r.part_id}"

    def test_zero_never_in_output(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """Part ID 0 must NEVER appear in the forecast."""
        pq, out_sha, src_sha = real_dataset
        result = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out")
        ids = [r.part_id for r in result.forecast.rankings]
        assert 0 not in ids, "CRITICAL: Part ID 0 found in forecast output"

    def test_no_duplicate_ids(self, real_dataset: tuple[Path, str, str], tmp_path: Path) -> None:
        """No duplicate part IDs in the forecast."""
        pq, out_sha, src_sha = real_dataset
        result = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out")
        ids = [r.part_id for r in result.forecast.rankings]
        assert len(set(ids)) == len(ids)

    def test_deterministic_output(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """Running twice on same data must produce identical output."""
        pq, out_sha, src_sha = real_dataset
        r1 = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out1")
        r2 = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out2")
        ids1 = [r.part_id for r in r1.forecast.rankings]
        ids2 = [r.part_id for r in r2.forecast.rankings]
        assert ids1 == ids2

    def test_csv_artifact_valid(self, real_dataset: tuple[Path, str, str], tmp_path: Path) -> None:
        """CSV artifact must be readable with correct schema."""
        pq, out_sha, src_sha = real_dataset
        out_dir = tmp_path / "out"
        run_canary_forecast(pq, "raw", out_sha, src_sha, out_dir)
        df = pd.read_csv(out_dir / "forecast.csv")
        assert list(df.columns) == ["rank", "part_id", "score"]
        assert len(df) == TOP_K
        assert 0 not in df["part_id"].values

    def test_json_artifact_valid(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """JSON artifact must be valid and contain provenance + rankings."""
        pq, out_sha, src_sha = real_dataset
        out_dir = tmp_path / "out"
        run_canary_forecast(pq, "raw", out_sha, src_sha, out_dir)
        data = json.loads((out_dir / "forecast.json").read_text())
        assert "provenance" in data
        assert "rankings" in data
        assert len(data["rankings"]) == TOP_K
        prov = data["provenance"]
        assert prov["model_name"] == "frequency_baseline"
        assert prov["dataset_variant"] == "raw"
        assert prov["run_id"]
        assert prov["run_timestamp"]

    def test_markdown_artifact_valid(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """Markdown artifact must contain the ranking table."""
        pq, out_sha, src_sha = real_dataset
        out_dir = tmp_path / "out"
        run_canary_forecast(pq, "raw", out_sha, src_sha, out_dir)
        content = (out_dir / "forecast.md").read_text()
        assert "frequency_baseline" in content
        assert "Rank" in content
        assert "Part ID" in content

    def test_provenance_has_dataset_fingerprint(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """Provenance must include the dataset fingerprint."""
        pq, out_sha, src_sha = real_dataset
        result = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out")
        assert result.provenance.dataset_fingerprint == out_sha
        assert result.provenance.source_fingerprint == src_sha
        assert len(result.provenance.source_fingerprint) == 64

    def test_scores_are_between_0_and_1(
        self, real_dataset: tuple[Path, str, str], tmp_path: Path
    ) -> None:
        """All frequency scores must be in [0, 1]."""
        pq, out_sha, src_sha = real_dataset
        result = run_canary_forecast(pq, "raw", out_sha, src_sha, tmp_path / "out")
        for r in result.forecast.rankings:
            assert 0.0 <= r.score <= 1.0


class TestForecastCli:
    """Integration tests for the forecast-next-day CLI command."""

    def _ensure_dataset_built(self) -> None:
        """Ensure the raw dataset is built before running CLI."""
        parquet = Path("data/processed/raw_v1.parquet")
        if not parquet.exists():
            subprocess.run(
                [sys.executable, "-m", "c5_forecasting", "build-dataset", "--variant", "raw"],
                capture_output=True,
                timeout=120,
            )

    def test_cli_forecast_exits_zero(self) -> None:
        """Running forecast-next-day must exit 0."""
        self._ensure_dataset_built()
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "forecast-next-day"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
        assert "Forecast PASSED" in result.stdout

    def test_cli_forecast_shows_top_5(self) -> None:
        """CLI output should show top-5 entries."""
        self._ensure_dataset_built()
        result = subprocess.run(
            [sys.executable, "-m", "c5_forecasting", "forecast-next-day"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "#1:" in result.stdout
        assert "#2:" in result.stdout
