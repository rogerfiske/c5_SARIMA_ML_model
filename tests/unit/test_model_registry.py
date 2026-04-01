"""Unit tests for the model registry."""

from __future__ import annotations

import pandas as pd
import pytest

from c5_forecasting.domain.constants import PART_COLUMNS, VALID_PART_IDS
from c5_forecasting.models.baseline import PartScore
from c5_forecasting.models.registry import (
    get_model_names,
    get_model_registry,
    get_scoring_function,
)


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from row dicts with Int64 typed part columns."""
    df = pd.DataFrame(rows)
    for col in PART_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def _make_row(date_str: str, counts: dict[str, int] | None = None) -> dict:
    """Build a row dict. Unspecified parts default to 0."""
    row: dict = {"date": pd.Timestamp(date_str)}
    for col in PART_COLUMNS:
        row[col] = counts.get(col, 0) if counts else 0
    return row


EXPECTED_MODELS = {
    "frequency_baseline",
    "recency_weighted",
    "rolling_window",
    "sarima",
    "uniform_baseline",
}


class TestModelRegistry:
    """Tests for the model registry."""

    def test_registry_has_five_models(self) -> None:
        """Registry should contain exactly 5 models."""
        registry = get_model_registry()
        assert len(registry) == 5

    def test_all_names_present(self) -> None:
        """All expected model names should be registered."""
        registry = get_model_registry()
        assert set(registry.keys()) == EXPECTED_MODELS

    def test_get_model_names_sorted(self) -> None:
        """get_model_names() should return a sorted list."""
        names = get_model_names()
        assert names == sorted(names)
        assert set(names) == EXPECTED_MODELS

    def test_get_scoring_function_returns_callable(self) -> None:
        """Each model should return a callable scoring function."""
        for name in EXPECTED_MODELS:
            fn = get_scoring_function(name)
            assert callable(fn)

    def test_unknown_model_raises(self) -> None:
        """Looking up a nonexistent model should raise KeyError."""
        with pytest.raises(KeyError, match="Unknown model"):
            get_scoring_function("nonexistent_model")

    def test_each_model_satisfies_protocol(self) -> None:
        """Each scoring function should accept a DataFrame and return list[PartScore]."""
        counts = {col: 1 for col in PART_COLUMNS[:10]}
        df = _make_df([_make_row("2020-01-01", counts), _make_row("2020-01-02")])

        for name in EXPECTED_MODELS:
            fn = get_scoring_function(name)
            result = fn(df)
            assert isinstance(result, list)
            assert all(isinstance(s, PartScore) for s in result)

    def test_each_model_returns_39_scores(self) -> None:
        """Each scoring function should return exactly 39 scores."""
        counts = {col: 1 for col in PART_COLUMNS[:10]}
        df = _make_df([_make_row("2020-01-01", counts), _make_row("2020-01-02")])

        for name in EXPECTED_MODELS:
            fn = get_scoring_function(name)
            result = fn(df)
            assert len(result) == 39, f"{name} returned {len(result)} scores, expected 39"

    def test_all_scores_in_valid_range(self) -> None:
        """All models should produce scores in [0, 1] with valid part IDs."""
        counts = {col: 1 for col in PART_COLUMNS[:10]}
        df = _make_df([_make_row("2020-01-01", counts), _make_row("2020-01-02")])

        for name in EXPECTED_MODELS:
            fn = get_scoring_function(name)
            result = fn(df)
            for s in result:
                assert 0.0 <= s.score <= 1.0, f"{name}: score {s.score} out of [0,1]"
                assert s.part_id in VALID_PART_IDS, f"{name}: part_id {s.part_id} invalid"
