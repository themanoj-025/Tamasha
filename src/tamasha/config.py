"""Centralized configuration using Pydantic Settings.

All paths, thresholds, model-selection criteria, and decay-rate
constants live here — never hardcoded inline in source modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Paths ──────────────────────────────────────────────────────────
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    FIGURES_DIR: Path = REPORTS_DIR / "figures"

    # ── Fuzzy-join thresholds ──────────────────────────────────────────
    FUZZY_JOIN_SCORE_CUTOFF: float = 60.0  # minimum rapidfuzz score
    FUZZY_JOIN_YEAR_TOLERANCE: int = 2  # ± years for year match

    # ── Model selection ────────────────────────────────────────────────
    MODEL_SELECTION_METRIC: Literal["MAE", "RMSE", "R2"] = "MAE"
    """Metric used to auto-select the best model (lowest for MAE/RMSE, highest for R2)."""
    CV_FOLDS: int = 5
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42

    # ── Bankability Score ──────────────────────────────────────────────
    BANKABILITY_DECAY_HALFLIFE_YEARS: float = 3.0
    """Time-decay half-life in years for the Bankability Score.
    A film's contribution to an actor's Bankability Score halves
    after this many years."""

    # ── Text / NLP ─────────────────────────────────────────────────────
    PLOT_SENTIMENT_MODEL: str = "vader"
    """Which sentiment model to use: 'vader' or a Hugging Face model id."""

    # ── CV / Poster ────────────────────────────────────────────────────
    POSTER_IMAGE_SIZE: tuple[int, int] = (224, 224)
    POSTER_SAMPLE_SIZE: int = 200

    # ── Release timing ─────────────────────────────────────────────────
    FESTIVAL_CLASH_WINDOW_DAYS: int = 7
    """Number of days before/after a major release to consider a 'clash'."""

    # ── Logging ────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"


settings = Settings()

# Ensure required directories exist
for _dir in [
    settings.DATA_RAW,
    settings.DATA_PROCESSED,
    settings.MODELS_DIR,
    settings.REPORTS_DIR,
    settings.FIGURES_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

logger.info("Configuration loaded: PROJECT_ROOT=%s", settings.PROJECT_ROOT)
