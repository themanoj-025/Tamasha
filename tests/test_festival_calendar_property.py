"""Property-based tests for the festival calendar module.

Uses Hypothesis to verify invariants of ``compute_clash_feature``:
- Symmetry: if A clashes with B, B clashes with A
- Order-invariance: shuffling rows doesn't change clash results
- Threshold behavior: exactly at window boundary
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from hypothesis import assume, given
from hypothesis import strategies as st

from tamasha.timing.festival_calendar import compute_clash_feature

# ── Strategy: sorted dates with gaps ──────────────────────────────────

_date_strategy = st.dates(
    min_value=date(2020, 1, 1),
    max_value=date(2025, 12, 31),
)


# ── Properties for compute_clash_feature ─────────────────────────────


class TestClashDetectionProperties:
    """Property-based tests for the O(n²) clash detection."""

    @given(
        dates=st.lists(
            _date_strategy,
            min_size=2,
            max_size=20,
            unique=True,
        ),
        window_days=st.integers(min_value=1, max_value=14),
    )
    def test_clash_is_symmetric(self, dates: list[date], window_days: int) -> None:
        """If movie A clashes with movie B, movie B must also clash with movie A."""
        df = pd.DataFrame(
            {
                "title": [f"Movie {i}" for i in range(len(dates))],
                "release_date": [d.isoformat() for d in dates],
            }
        )
        result = compute_clash_feature(df, clash_window_days=window_days)
        clashes = result[result["has_clash"]].index.tolist()

        # Manually verify: for every pair within window_days, both should be flagged
        for i in range(len(dates)):
            for j in range(i + 1, len(dates)):
                diff = abs((dates[i] - dates[j]).days)
                if 0 < diff <= window_days:
                    assert (
                        i in clashes
                    ), f"{dates[i]} and {dates[j]} are {diff} days apart but movie {i} not flagged"
                    assert (
                        j in clashes
                    ), f"{dates[i]} and {dates[j]} are {diff} days apart but movie {j} not flagged"

    @given(
        dates=st.lists(
            _date_strategy,
            min_size=2,
            max_size=15,
            unique=True,
        ),
        window_days=st.integers(min_value=1, max_value=14),
    )
    def test_clash_is_order_invariant(self, dates: list[date], window_days: int) -> None:
        """Shuffling rows must produce the same clash flags (by movie identity)."""
        import random

        df = pd.DataFrame(
            {
                "title": [f"Movie {i}" for i in range(len(dates))],
                "release_date": [d.isoformat() for d in dates],
            }
        )
        result_original = compute_clash_feature(df, clash_window_days=window_days)

        # Shuffle and recompute
        shuffled_indices = list(range(len(dates)))
        random.shuffle(shuffled_indices)
        df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)
        result_shuffled = compute_clash_feature(df_shuffled, clash_window_days=window_days)

        # Map back to original indices via movie title
        original_clashes = set(
            df.iloc[i]["title"] for i in result_original[result_original["has_clash"]].index
        )
        shuffled_clashes = set(
            df_shuffled.iloc[i]["title"]
            for i in result_shuffled[result_shuffled["has_clash"]].index
        )

        assert original_clashes == shuffled_clashes, (
            f"Clash sets differ after shuffle.\n"
            f"Original: {original_clashes}\n"
            f"Shuffled: {shuffled_clashes}"
        )

    @given(
        base_date=_date_strategy,
        window_days=st.integers(min_value=1, max_value=14),
    )
    def test_clash_at_exact_boundary(self, base_date: date, window_days: int) -> None:
        """Movies exactly window_days apart should clash."""
        d1 = base_date
        d2 = base_date + timedelta(days=window_days)

        df = pd.DataFrame(
            {
                "title": ["Movie A", "Movie B"],
                "release_date": [d1.isoformat(), d2.isoformat()],
            }
        )
        result = compute_clash_feature(df, clash_window_days=window_days)

        assert result["has_clash"].all(), (
            f"Movies {d1} and {d2} ({window_days} days apart) should clash "
            f"at window={window_days}"
        )

    @given(
        base_date=_date_strategy,
        window_days=st.integers(min_value=1, max_value=14),
    )
    def test_no_clash_beyond_window(self, base_date: date, window_days: int) -> None:
        """Movies more than window_days apart should NOT clash."""
        d1 = base_date
        d2 = base_date + timedelta(days=window_days + 1)

        assume(d2.year <= 2025)  # stay within valid range

        df = pd.DataFrame(
            {
                "title": ["Movie A", "Movie B"],
                "release_date": [d1.isoformat(), d2.isoformat()],
            }
        )
        result = compute_clash_feature(df, clash_window_days=window_days)

        assert not result["has_clash"].any(), (
            f"Movies {d1} and {d2} ({window_days + 1} days apart) should NOT clash "
            f"at window={window_days}"
        )

    @given(
        dates=st.lists(
            _date_strategy,
            min_size=3,
            max_size=10,
            unique=True,
        ),
    )
    def test_movies_far_apart_have_no_clashes(self, dates: list[date]) -> None:
        """When all movies are >14 days apart, no clashes should be detected."""
        # Ensure all dates are at least 15 days apart
        sorted_dates = sorted(dates)
        for i in range(len(sorted_dates) - 1):
            diff = (sorted_dates[i + 1] - sorted_dates[i]).days
            assume(diff > 14)

        df = pd.DataFrame(
            {
                "title": [f"Movie {i}" for i in range(len(dates))],
                "release_date": [d.isoformat() for d in sorted_dates],
            }
        )
        result = compute_clash_feature(df, clash_window_days=7)

        assert not result[
            "has_clash"
        ].any(), f"All {len(dates)} movies are >14 days apart — no clashes expected at window=7"

    def test_single_movie_no_clash(self) -> None:
        """Single movie should never clash."""
        df = pd.DataFrame(
            {
                "title": ["Only Movie"],
                "release_date": ["2024-01-01"],
            }
        )
        result = compute_clash_feature(df, clash_window_days=7)
        assert not result["has_clash"].iloc[0]

    def test_empty_dataframe(self) -> None:
        """Empty DataFrame should not crash."""
        df = pd.DataFrame({"title": [], "release_date": []})
        result = compute_clash_feature(df, clash_window_days=7)
        assert "has_clash" in result.columns
        assert len(result) == 0

    @given(
        dates=st.lists(
            _date_strategy,
            min_size=2,
            max_size=8,
            unique=True,
        ),
        window_days=st.integers(min_value=1, max_value=30),
    )
    def test_clash_count_within_bounds(self, dates: list[date], window_days: int) -> None:
        """Number of clashing movies can't exceed total movies."""
        df = pd.DataFrame(
            {
                "title": [f"Movie {i}" for i in range(len(dates))],
                "release_date": [d.isoformat() for d in dates],
            }
        )
        result = compute_clash_feature(df, clash_window_days=window_days)
        n_clash = result["has_clash"].sum()
        assert 0 <= n_clash <= len(dates)
