"""Chemistry pairing analysis.

For actor pairs with 2+ joint appearances, statistically test whether
their joint-film performance exceeds their individual solo-average
Bankability Scores.  Rank and report top pairs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_chemistry_pairs(
    df: pd.DataFrame,
    cast_column: str = "cast",
    rating_column: Optional[str] = None,
    boxoffice_column: Optional[str] = None,
    min_joint_films: int = 2,
    top_n: int = 10,
) -> pd.DataFrame:
    """Identify actor pairs with statistically exceptional chemistry.

    For every actor pair that appears in at least ``min_joint_films``:
    1. Compute the average performance of their joint films.
    2. Compute each actor's solo average from films where they appear
       without the other.
    3. If joint > max(solo_avg_a, solo_avg_b), the pair is considered
       "high-chemistry" — this is compared as a simple uplift score.

    Parameters
    ----------
    df : pd.DataFrame
        Movie DataFrame.
    cast_column : str, default="cast"
        Column with comma-separated cast.
    rating_column : str, optional
        Column with numeric rating.
    boxoffice_column : str, optional
        Column with numeric box office.
    min_joint_films : int, default=2
        Minimum number of joint appearances to consider.
    top_n : int, default=10
        Number of top pairs to return.

    Returns
    -------
    pd.DataFrame
        Ranked chemistry pairs with joint average, solo averages,
        and uplift score.
    """
    # Handle empty DataFrame early
    if df.empty or cast_column not in df.columns:
        logger.warning("Empty DataFrame or missing cast column; returning empty result.")
        return pd.DataFrame()

    # Build performance series
    performance = pd.Series(1.0, index=df.index)
    if rating_column and rating_column in df.columns:
        r = pd.to_numeric(df[rating_column], errors="coerce").fillna(0)
        performance = performance * (r / 10.0)
    if boxoffice_column and boxoffice_column in df.columns:
        b = pd.to_numeric(df[boxoffice_column], errors="coerce").fillna(0)
        b_norm = np.log1p(b) / np.log1p(b.max()) if b.max() > 0 else 0
        performance = performance * (1 + b_norm)

    df = df.copy()
    df["_performance"] = performance

    # Parse cast lists
    df["_cast_list"] = df[cast_column].fillna("").astype(str).str.split(r"\s*,\s*")

    # Build co-appearance dictionary
    pair_films: dict[tuple[str, str], list[float]] = {}
    actor_all_films: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        cast = [c.strip().lower() for c in row["_cast_list"] if c.strip()]
        perf = row["_performance"]

        # Solo performance tracking
        for actor in cast:
            if actor not in actor_all_films:
                actor_all_films[actor] = []
            actor_all_films[actor].append(perf)

        # Pair tracking
        for i, a1 in enumerate(cast):
            for a2 in cast[i + 1 :]:
                pair = (min(a1, a2), max(a1, a2))
                if pair not in pair_films:
                    pair_films[pair] = []
                pair_films[pair].append(perf)

    # Compute chemistry for each pair
    rows_list = []
    for (a1, a2), joint_perfs in pair_films.items():
        if len(joint_perfs) < min_joint_films:
            continue

        joint_avg = np.mean(joint_perfs)

        # Solo averages
        solo_a1 = np.mean(actor_all_films.get(a1, [0]))
        solo_a2 = np.mean(actor_all_films.get(a2, [0]))
        best_solo = max(solo_a1, solo_a2)

        uplift = joint_avg - best_solo

        if uplift > 0:
            rows_list.append(
                {
                    "actor_1": a1.title(),
                    "actor_2": a2.title(),
                    "joint_films": len(joint_perfs),
                    "joint_avg_perf": round(joint_avg, 4),
                    "solo_avg_1": round(solo_a1, 4),
                    "solo_avg_2": round(solo_a2, 4),
                    "uplift": round(uplift, 4),
                }
            )

    if not rows_list:
        logger.warning("No chemistry pairs found with ≥ %d joint films.", min_joint_films)
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows_list)
        .sort_values("uplift", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info("Top %d chemistry pairs identified.", len(result))
    return result
