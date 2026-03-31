"""Unit tests for the forecast pipeline and artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from c5_forecasting.domain.constants import PART_COLUMNS, TOP_K, VALID_PART_IDS
from c5_forecasting.pipelines.forecast import run_canary_forecast


def _make_parquet(tmp_path: Path, n_rows: int = 10) -> Path:
    """Write a minimal working-dataset Parquet for testing."""
    rows = []
    for i in range(n_rows):
        row: dict = {"date": pd.Timestamp(f"2020-01-{i + 1:02d}")}
        for j, col in enumerate(PART_COLUMNS):
            row[col] = 1 if j < 30 else 0
        # Annotation columns
        row["row_total"] = 30
        row["total_class"] = "standard_output"
        row["is_exception_day"] = False
        row["domain_event_label"] = ""
        row["quality_flags"] = ""
        rows.append(row)

    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        df[col] = df[col].astype("Int64")

    path = tmp_path / "test_v1.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


class TestRunCanaryForecast:
    """Tests for run_canary_forecast."""

    def test_produces_20_rankings(self, tmp_path: Path) -> None:
        """Forecast should produce exactly 20 ranked entries."""
        pq = _make_parquet(tmp_path)
        result = run_canary_forecast(pq, "raw", "abc", "def", tmp_path / "out")
        assert len(result.forecast.rankings) == TOP_K

    def test_all_ids_valid(self, tmp_path: Path) -> None:
        """All ranked IDs must be in 1..39."""
        pq = _make_parquet(tmp_path)
        result = run_canary_forecast(pq, "raw", "abc", "def", tmp_path / "out")
        for r in result.forecast.rankings:
            assert r.part_id in VALID_PART_IDS

    def test_no_zero_in_output(self, tmp_path: Path) -> None:
        """0 must never appear in forecast output."""
        pq = _make_parquet(tmp_path)
        result = run_canary_forecast(pq, "raw", "abc", "def", tmp_path / "out")
        ids = [r.part_id for r in result.forecast.rankings]
        assert 0 not in ids

    def test_csv_artifact_written(self, tmp_path: Path) -> None:
        """CSV forecast artifact must be written."""
        pq = _make_parquet(tmp_path)
        out_dir = tmp_path / "out"
        run_canary_forecast(pq, "raw", "abc", "def", out_dir)
        assert (out_dir / "forecast.csv").exists()

    def test_json_artifact_written(self, tmp_path: Path) -> None:
        """JSON forecast artifact must be written."""
        pq = _make_parquet(tmp_path)
        out_dir = tmp_path / "out"
        run_canary_forecast(pq, "raw", "abc", "def", out_dir)
        path = out_dir / "forecast.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "provenance" in data
        assert "rankings" in data
        assert len(data["rankings"]) == TOP_K

    def test_markdown_artifact_written(self, tmp_path: Path) -> None:
        """Markdown forecast artifact must be written."""
        pq = _make_parquet(tmp_path)
        out_dir = tmp_path / "out"
        run_canary_forecast(pq, "raw", "abc", "def", out_dir)
        md_path = out_dir / "forecast.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Rank" in content
        assert "Part ID" in content

    def test_provenance_fields_populated(self, tmp_path: Path) -> None:
        """All provenance fields must be non-empty."""
        pq = _make_parquet(tmp_path)
        result = run_canary_forecast(pq, "raw", "fp123", "src456", tmp_path / "out")
        prov = result.provenance
        assert prov.run_id
        assert prov.run_timestamp
        assert prov.model_name == "frequency_baseline"
        assert prov.dataset_variant == "raw"
        assert prov.dataset_fingerprint == "fp123"
        assert prov.source_fingerprint == "src456"
        assert prov.config_fingerprint
        assert prov.k == TOP_K
        assert prov.dataset_row_count == 10

    def test_three_artifacts_listed(self, tmp_path: Path) -> None:
        """Result should list exactly 3 artifact paths."""
        pq = _make_parquet(tmp_path)
        result = run_canary_forecast(pq, "raw", "abc", "def", tmp_path / "out")
        assert len(result.artifacts) == 3

    def test_csv_has_correct_columns(self, tmp_path: Path) -> None:
        """CSV should have rank, part_id, score columns."""
        pq = _make_parquet(tmp_path)
        out_dir = tmp_path / "out"
        run_canary_forecast(pq, "raw", "abc", "def", out_dir)
        df = pd.read_csv(out_dir / "forecast.csv")
        assert list(df.columns) == ["rank", "part_id", "score"]
        assert len(df) == TOP_K

    def test_result_to_dict(self, tmp_path: Path) -> None:
        """ForecastResult.to_dict should be JSON-serializable."""
        pq = _make_parquet(tmp_path)
        result = run_canary_forecast(pq, "raw", "abc", "def", tmp_path / "out")
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
