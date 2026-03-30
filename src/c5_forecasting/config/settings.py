"""Application settings loaded from environment variables and .env files."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Top-level application settings.

    Values can be overridden via environment variables prefixed with ``C5_``.
    """

    model_config = {"env_prefix": "C5_"}

    log_level: str = "INFO"
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    configs_dir: Path = Path("configs")

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_data_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"


def get_settings() -> AppSettings:
    """Load settings from environment."""
    return AppSettings()
