"""Tests for the chemistry pairs module.

Uses a synthetic graph with a known "obvious chemistry pair" planted
in the test data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tamasha.network.chemistry_pairs import detect_chemistry_pairs


def test_known_chemistry_pair_detected():
    """Plant an obvious chemistry pair and verify it's detected.

    Actor X and Actor Y appear together 3 times with high ratings,
    while their solo films have lower ratings.
    """
    df = pd.DataFrame(
        {
            "title": [
                "Joint Hit 1", "Joint Hit 2", "Joint Hit 3",
                "Solo X", "Solo Y",
            ],
            "cast": [
                "Actor X, Actor Y",
                "Actor X, Actor Y, Actor Z",
                "Actor X, Actor Y",
                "Actor X",
                "Actor Y",
            ],
            "rating": [9.0, 8.5, 9.5, 6.0, 5.5],
            "collection": [500e6, 400e6, 600e6, 100e6, 80e6],
        }
    )
    pairs = detect_chemistry_pairs(
        df,
        cast_column="cast",
        rating_column="rating",
        boxoffice_column="collection",
        min_joint_films=2,
    )
    assert len(pairs) > 0
    # The Actor X / Actor Y pair should be in the results
    pair_found = any(
        (row["actor_1"].lower() == "actor x" and row["actor_2"].lower() == "actor y")
        or (row["actor_1"].lower() == "actor y" and row["actor_2"].lower() == "actor x")
        for _, row in pairs.iterrows()
    )
    assert pair_found, "Known chemistry pair (Actor X, Actor Y) not detected"


def test_min_joint_films_filter():
    """Test that pairs below min_joint_films are excluded."""
    df = pd.DataFrame(
        {
            "title": ["Joint 1", "Solo A", "Solo B"],
            "cast": [
                "Actor A, Actor B",
                "Actor A",
                "Actor B",
            ],
            "rating": [7.0, 6.0, 5.0],
            "collection": [100e6, 50e6, 30e6],
        }
    )
    pairs = detect_chemistry_pairs(
        df,
        min_joint_films=3,
    )
    # Only 1 joint film, so should be excluded with min=3
    assert len(pairs) == 0


def test_no_pairs_below_threshold():
    """Test that an empty DataFrame returns empty."""
    df = pd.DataFrame()
    pairs = detect_chemistry_pairs(df)
    assert len(pairs) == 0
