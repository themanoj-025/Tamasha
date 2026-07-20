"""Regression test for vectorized bankability computation.

Verifies the current vectorized implementation produces the same output
as a hand-computed reference for a small, deterministic input set.

Since the pre-vectorization (iterrows) implementation has been deleted
from the codebase, we derive the expected output manually for a small
input with known actors and time-decay behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


from tamasha.network.bankability_score import compute_bankability_scores


def _hand_compute_reference() -> pd.DataFrame:
    """Manually compute expected bankability scores for a small input.

    Three movies with known cast/director and ratings.
    We can verify the time-decay weights by hand.

    Movies:
    - Movie A (2024): cast [ActorX, ActorY], dir DirZ, rating=8.0
    - Movie B (2023): cast [ActorX, ActorZ], dir DirZ, rating=6.0
    - Movie C (2022): cast [ActorY, ActorZ], dir DirY, rating=7.0

    Current year = 2024, half_life = 3 years.
    weights: 2024→1.0, 2023→0.7937, 2022→0.62996
    performance = rating / 10 (since no box office column)

    ActorX:
      Movie A: w=1.0, perf=0.8 → weighted_sum=0.8, total_weight=1.0
      Movie B: w=0.7937, perf=0.6 → weighted_sum=0.47622, total_weight=0.7937
      Total: sum=1.27622, weight=1.7937 → score=0.7116

    ActorY:
      Movie A: w=1.0, perf=0.8 → weighted_sum=0.8, total_weight=1.0
      Movie C: w=0.62996, perf=0.7 → weighted_sum=0.44097, total_weight=0.62996
      Total: sum=1.24097, weight=1.62996 → score=0.7614

    ActorZ:
      Movie B: w=0.7937, perf=0.6 → weighted_sum=0.47622, total_weight=0.7937
      Movie C: w=0.62996, perf=0.7 → weighted_sum=0.44097, total_weight=0.62996
      Total: sum=0.91719, weight=1.42366 → score=0.6442

    DirZ:
      Movie A: w=1.0, perf=0.8 → weighted_sum=0.8, total_weight=1.0
      Movie B: w=0.7937, perf=0.6 → weighted_sum=0.47622, total_weight=0.7937
      Total: sum=1.27622, weight=1.7937 → score=0.7116

    DirY:
      Movie C: w=0.62996, perf=0.7 → weighted_sum=0.44097, total_weight=0.62996
      Total: sum=0.44097, weight=0.62996 → score=0.7000
    """
    return pd.DataFrame({
        "actor": ["ActorY", "ActorX", "DirZ", "ActorZ", "DirY"],
        "type": ["actor", "actor", "director", "actor", "director"],
        "bankability_score": [0.7614, 0.7116, 0.7116, 0.6442, 0.7000],
        "film_count": [2, 2, 2, 2, 1],
    })


class TestBankabilityRegression:
    """Verify vectorized bankability matches hand-computed reference."""

    def test_matches_hand_computed_reference(self) -> None:
        """Vectorized output matches manually derived expectations."""
        df = pd.DataFrame({
            "title": ["Movie A", "Movie B", "Movie C"],
            "cast": ["ActorX, ActorY", "ActorX, ActorZ", "ActorY, ActorZ"],
            "director": ["DirZ", "DirZ", "DirY"],
            "year": [2024, 2023, 2022],
            "rating": [8.0, 6.0, 7.0],
        })

        result = compute_bankability_scores(
            df,
            cast_column="cast",
            director_column="director",
            year_column="year",
            rating_column="rating",
            half_life=3.0,
        )

        expected = _hand_compute_reference()

        # Merge on actor+type for comparison
        merged = result.merge(expected, on=["actor", "type"], suffixes=("_actual", "_expected"))

        # Compare bankability scores with tolerance
        for _, row in merged.iterrows():
            assert abs(row["bankability_score_actual"] - row["bankability_score_expected"]) < 0.01, (
                f"{row['actor']} ({row['type']}): "
                f"actual={row['bankability_score_actual']:.4f}, "
                f"expected={row['bankability_score_expected']:.4f}"
            )

    def test_film_count_matches_expected(self) -> None:
        """Film counts should match the expected values."""
        df = pd.DataFrame({
            "title": ["Movie A", "Movie B", "Movie C"],
            "cast": ["ActorX, ActorY", "ActorX, ActorZ", "ActorY, ActorZ"],
            "director": ["DirZ", "DirZ", "DirY"],
            "year": [2024, 2023, 2022],
            "rating": [8.0, 6.0, 7.0],
        })

        result = compute_bankability_scores(
            df,
            cast_column="cast",
            director_column="director",
            year_column="year",
            rating_column="rating",
            half_life=3.0,
        )

        expected = _hand_compute_reference()
        merged = result.merge(expected, on=["actor", "type"], suffixes=("_actual", "_expected"))

        for _, row in merged.iterrows():
            assert row["film_count_actual"] == row["film_count_expected"], (
                f"{row['actor']}: got {row['film_count_actual']}, "
                f"expected {row['film_count_expected']}"
            )
