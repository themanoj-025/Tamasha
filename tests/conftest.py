"""Shared pytest fixtures for Tamasha tests.

Fixtures that require dummy model files use explicit ``scope="function"``
or ``scope="session"`` but are **never** ``autouse=True`` — tests must
explicitly request them to avoid side effects on unrelated tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import LabelEncoder

from tamasha.config import settings


# ── Dummy artifacts for PredictionService / API tests ────────────────

_FAKE_RATING_COLS = [
    "genre_Action", "genre_Drama", "genre_Romance",
    "director_encoded", "cast_size", "runtime_minutes",
    "budget_inr", "decade_2020",
]

_FAKE_BOX_COLS = _FAKE_RATING_COLS + ["avg_bankability_score"]


def install_dummy_artifacts(
    models_dir: Path,
    reports_dir: Path | None = None,
) -> None:
    """Write dummy model files, feature JSONs, and bankability CSV.

    Idempotent — safe to call in setup even if files already exist.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # Rating model
    _fit_dummy(_FAKE_RATING_COLS, 7.0, models_dir / "best_rating_model.pkl")
    (models_dir / "rating_features.json").write_text(json.dumps(_FAKE_RATING_COLS))

    # Box office model
    _fit_dummy(_FAKE_BOX_COLS, 500_000_000.0, models_dir / "best_boxoffice_model.pkl")
    (models_dir / "boxoffice_features.json").write_text(json.dumps(_FAKE_BOX_COLS))

    # Director encoder
    le = LabelEncoder()
    le.fit(["Director A", "Director B", "Director C"])
    joblib.dump(le, models_dir / "director_encoder.pkl")

    # Bankability scores (optional, for family of tests that need it)
    if reports_dir is not None:
        reports_dir.mkdir(parents=True, exist_ok=True)
        bank_df = pd.DataFrame({
            "actor": ["Actor Known", "Actor AlsoKnown", "Shah Rukh Khan"],
            "type": ["actor", "actor", "actor"],
            "bankability_score": [1.5, 0.8, 1.2],
            "film_count": [5, 3, 10],
        })
        bank_df.to_csv(reports_dir / "bankability_scores.csv", index=False)

        chem_df = pd.DataFrame({
            "actor_1": ["Actor Known", "Actor Known"],
            "actor_2": ["Actor AlsoKnown", "Shah Rukh Khan"],
            "joint_films": [2, 3],
            "uplift": [0.05, 0.08],
        })
        chem_df.to_csv(reports_dir / "chemistry_pairs.csv", index=False)


def _fit_dummy(cols: list[str], constant: float, path: Path) -> None:
    """Create and save a DummyRegressor."""
    model = DummyRegressor(strategy="constant", constant=constant)
    X = np.zeros((1, len(cols)))
    model.fit(X, np.array([constant]))
    joblib.dump(model, path)


# ── Auto-setup: install dummy artifacts for API contract tests ──────
# This hook runs once when pytest loads conftest.py, ensuring that
# ``TestClient(app)``-based tests (test_api_contract.py, test_api.py)
# always find valid model files in the default settings paths.

def pytest_configure() -> None:
    """One-time hook: install dummy artifacts in the default models/reports dirs.

    This is intentionally NOT a fixture because ``TestClient(app)``
    creates its own ``PredictionService`` from ``settings.MODELS_DIR``
    and we can't inject a custom path via Depends() for module-level
    ``client`` objects.
    """
    install_dummy_artifacts(settings.MODELS_DIR, settings.REPORTS_DIR)


# ── Function-scoped fixtures for PredictionService tests ─────────────

@pytest.fixture
def dummy_models_dir(tmp_path: Path) -> Path:
    """Fresh dummy artifacts per test (isolated, no cross-test pollution)."""
    d = tmp_path / "artifacts"
    install_dummy_artifacts(d, d)
    return d


@pytest.fixture
def dummy_svc(dummy_models_dir: Path):
    """PredictionService loaded with fresh dummy artifacts."""
    from tamasha.predict import PredictionService

    svc = PredictionService(models_dir=dummy_models_dir, reports_dir=dummy_models_dir)
    svc.load()
    return svc


# ── Plain-data fixtures (no disk I/O) ────────────────────────────────

@pytest.fixture
def sample_movie_df() -> pd.DataFrame:
    """A small synthetic movie DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
            "year": [2020, 2021, 2022, 2020, 2023],
            "genre": ["Action, Drama", "Comedy, Romance", "Drama, Thriller",
                      "Action, Comedy", "Romance, Drama"],
            "cast": ["Actor X, Actor Y", "Actor Y, Actor Z", "Actor X, Actor Z",
                     "Actor W, Actor X", "Actor Y, Actor W"],
            "director": ["Director A", "Director B", "Director A",
                         "Director C", "Director B"],
            "rating": [7.5, 6.8, 8.2, 6.0, 7.9],
            "budget_inr": [50e6, 30e6, 80e6, 20e6, 60e6],
            "collection_inr": [150e6, 80e6, 300e6, 50e6, 200e6],
            "runtime": [150, 120, 160, 100, 140],
        }
    )


@pytest.fixture
def sample_imdb_df() -> pd.DataFrame:
    """Synthetic IMDB-like DataFrame for join testing."""
    return pd.DataFrame(
        {
            "Title": ["Movie A", "Movie B", "Movie Extra", "Movie D"],
            "Year": [2020, 2021, 2019, 2020],
            "Rating": [7.5, 6.8, 5.0, 6.0],
        }
    )


@pytest.fixture
def sample_boxoffice_df() -> pd.DataFrame:
    """Synthetic Box Office-like DataFrame for join testing."""
    return pd.DataFrame(
        {
            "title": ["Movie A", "Movie B", "Movie C", "Movie D"],
            "year": [2020, 2021, 2022, 2020],
            "box_office": ["₹150 Cr", "₹80 Cr", "₹300 Cr", "₹50 Cr"],
        }
    )


@pytest.fixture
def sample_bankability_scores() -> pd.DataFrame:
    """Synthetic Bankability Scores for testing."""
    return pd.DataFrame(
        {
            "actor": ["Actor X", "Actor Y", "Actor Z", "Actor W"],
            "type": ["actor", "actor", "actor", "actor"],
            "bankability_score": [0.85, 0.65, 0.72, 0.45],
            "film_count": [3, 3, 2, 2],
        }
    )
