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

    # ── Festival / release-window multipliers ───────────────────────────
    FESTIVAL_CLASH_WINDOW_DAYS: int = 7
    """Number of days before/after a major release to consider a 'clash'."""
    FESTIVAL_MULTIPLIERS: dict[str, float] = {
        "Normal": 1.0,
        "Diwali": 1.25,
        "Eid": 1.18,
        "Christmas": 1.12,
        "Independence Day": 1.08,
        "Republic Day": 1.05,
        "New Year": 1.10,
    }
    """
    Estimated box-office multipliers for different release windows.

    .. note::

       These are **domain-expert priors**, not derived from historical
       data.  In a production system these should be periodically
       re-estimated from opening-weekend performance grouped by festival
       window with appropriate controls for selection bias (studios
       release their best films in festival windows, so the multiplier
       captures both the genuine uplift and the quality-selection effect).

       Current estimates are based on industry heuristics:
       - Diwali (1.25×): strongest release window
       - Eid (1.18×): strong, especially for action/family films
       - Christmas (1.12×): solid family window
       - Others: moderate (1.05-1.10×)
       - Normal (1.0×): baseline non-festival release
    """

    # ── Auth ───────────────────────────────────────────────────────────
    API_KEY: str = "tamasha-dev-key-2026"
    """API key for authenticating requests (X-API-Key header).

    Default dev key is documented; set a strong random value via
    the ``API_KEY`` env var in production.
    """
    ALLOWED_ORIGINS: str = "http://localhost:8501,http://localhost:8000"
    """Comma-separated list of allowed CORS origins.
    Set via env var for production deployments.
    """

    # ── Rate limiting ───────────────────────────────────────────────────
    RATE_LIMIT: str = "60/minute"
    """Global rate limit string (slowapi format)."""

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
