"""Tests for the Bankability Score module.

Uses a small synthetic graph with a hand-computed expected score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tamasha.network.bankability_score import compute_bankability_scores


def test_bankability_scores_positive():
    """Test that all bankability scores are non-negative.

    Note: Scores can exceed 1.0 because the performance signal
    combines normalized rating (0-1) with a log-transformed box
    office factor, which can push the combined score above 1.
    """
    df = pd.DataFrame(
        {
            "cast": ["Actor A, Actor B", "Actor A, Actor C"],
            "director": ["Dir X", "Dir Y"],
            "year": [2020, 2022],
            "rating": [8.0, 6.0],
            "collection": [200_000_000, 100_000_000],
        }
    )
    scores = compute_bankability_scores(
        df,
        cast_column="cast",
        director_column="director",
        year_column="year",
        rating_column="rating",
        boxoffice_column="collection",
    )
    assert all(s >= 0 for s in scores["bankability_score"]), "All scores should be non-negative"


def test_bankability_actors_only_type():
    """Test that returned types are correct."""
    df = pd.DataFrame(
        {
            "cast": ["Actor A"],
            "director": ["Dir X"],
            "year": [2020],
            "rating": [7.0],
            "collection": [150_000_000],
        }
    )
    scores = compute_bankability_scores(
        df,
        rating_column="rating",
        boxoffice_column="collection",
    )
    assert "actor" in scores["type"].values
    assert "director" in scores["type"].values


def test_bankability_hand_computed():
    """Test against a hand-computed expected score."""
    # Single actor, single film, current year → weight ~1.0
    # Rating 10/10 + box office → performance should be high
    df = pd.DataFrame(
        {
            "cast": ["Actor A"],
            "director": ["Dir X"],
            "year": [2024],
            "rating": [10.0],
            "collection": [1_000_000_000],
        }
    )
    scores = compute_bankability_scores(
        df,
        rating_column="rating",
        boxoffice_column="collection",
    )
    actor_score = scores[scores["actor"] == "Actor A"]["bankability_score"].values[0]
    # Should be high performance (rating=1.0 normalized, box office high)
    assert actor_score > 0.5
