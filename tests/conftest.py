"""Shared pytest fixtures for Tamasha tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


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
