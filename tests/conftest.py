"""Shared pytest fixtures for Tamasha tests."""

from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tamasha.config import settings


@pytest.fixture(autouse=True, scope="session")
def setup_dummy_models_if_missing() -> None:
    """Ensure dummy model files exist for API testing if not already trained."""
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rating_model_path = settings.MODELS_DIR / "best_rating_model.pkl"
    rating_features_path = settings.MODELS_DIR / "rating_features.json"
    if not rating_model_path.exists() or not rating_features_path.exists():
        rating_cols = [
            "genre_Action",
            "genre_Drama",
            "genre_Romance",
            "cast_size",
            "runtime_minutes",
            "budget_inr",
        ]
        model = DummyRegressor(strategy="constant", constant=7.0)
        X = np.zeros((1, len(rating_cols)))
        y = np.array([7.0])
        model.fit(X, y)
        joblib.dump(model, rating_model_path)
        rating_features_path.write_text(json.dumps(rating_cols))

    box_model_path = settings.MODELS_DIR / "best_boxoffice_model.pkl"
    box_features_path = settings.MODELS_DIR / "boxoffice_features.json"
    if not box_model_path.exists() or not box_features_path.exists():
        box_cols = [
            "genre_Action",
            "genre_Drama",
            "cast_size",
            "runtime_minutes",
            "budget_inr",
            "avg_bankability_score",
        ]
        model = DummyRegressor(strategy="constant", constant=500000000.0)
        X = np.zeros((1, len(box_cols)))
        y = np.array([500000000.0])
        model.fit(X, y)
        joblib.dump(model, box_model_path)
        box_features_path.write_text(json.dumps(box_cols))


@pytest.fixture
def sample_movie_df() -> pd.DataFrame:
    """A small synthetic movie DataFrame for testing."""
    return pd.DataFrame(
        {
            "title": [
                "Movie A",
                "Movie B",
                "Movie C",
                "Movie D",
                "Movie E",
            ],
            "year": [2020, 2021, 2022, 2020, 2023],
            "genre": [
                "Action, Drama",
                "Comedy, Romance",
                "Drama, Thriller",
                "Action, Comedy",
                "Romance, Drama",
            ],
            "cast": [
                "Actor X, Actor Y",
                "Actor Y, Actor Z",
                "Actor X, Actor Z",
                "Actor W, Actor X",
                "Actor Y, Actor W",
            ],
            "director": [
                "Director A",
                "Director B",
                "Director A",
                "Director C",
                "Director B",
            ],
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
