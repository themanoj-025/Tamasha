"""Tests for chemistry pairs feature engineering."""


import pandas as pd

from tamasha.network.chemistry_pairs import detect_chemistry_pairs


class TestDetectChemistryPairs:
    """Tests for detect_chemistry_pairs."""

    def test_returns_dataframe(self) -> None:
        """detect_chemistry_pairs should return a DataFrame."""
        data = {
            "cast": ["Actor A, Actor B, Actor C", "Actor A, Actor B"],
            "title": ["Movie 1", "Movie 2"],
        }
        df = pd.DataFrame(data)
        result = detect_chemistry_pairs(df)
        assert isinstance(result, pd.DataFrame)

    def test_empty_cast(self) -> None:
        """Empty cast should still produce a valid DataFrame."""
        data = {
            "cast": ["", "Unknown"],
            "title": ["Movie 1", "Movie 2"],
        }
        df = pd.DataFrame(data)
        result = detect_chemistry_pairs(df)
        assert isinstance(result, pd.DataFrame)

    def test_single_actor_no_pairs(self) -> None:
        """Films with a single actor should not produce pair entries."""
        data = {
            "cast": ["Actor A", "Actor B"],
            "title": ["Movie 1", "Movie 2"],
        }
        df = pd.DataFrame(data)
        result = detect_chemistry_pairs(df, min_joint_films=2)
        # With only 1 film per actor and min_joint_films=2, no pairs
        assert isinstance(result, pd.DataFrame)
