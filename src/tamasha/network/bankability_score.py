"""Bankability Score computation.

The Bankability Score of an actor/director is a time-decay-weighted
historical average of their films' performance.

Decay function
--------------
We use an exponential decay with a half-life of ``H`` years
(configurable in ``settings.BANKABILITY_DECAY_HALFLIFE_YEARS``):

    w(t) = 2^{-(current_year - t) / H}

where ``t`` is the film's release year.  This gives more weight to
recent films while still acknowledging past work.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from tamasha.config import settings

logger = logging.getLogger(__name__)


def _decay_weight(
    release_year: float,
    current_year: float,
    half_life: float,
) -> float:
    """Exponential time-decay weight.

    Parameters
    ----------
    release_year : float
        Film release year.
    current_year : float
        Reference year (usually the latest year in the dataset).
    half_life : float
        Half-life in years.

    Returns
    -------
    float
        Weight in (0, 1].
    """
    return float(2.0 ** (-(current_year - release_year) / half_life))


def compute_bankability_scores(
    df: pd.DataFrame,
    cast_column: str = "cast",
    director_column: str = "director",
    year_column: str = "year",
    rating_column: Optional[str] = None,
    boxoffice_column: Optional[str] = None,
    half_life: Optional[float] = None,
) -> pd.DataFrame:
    """Compute Bankability Scores for all actors and directors.

    Parameters
    ----------
    df : pd.DataFrame
        Movie DataFrame.  Each row is a movie with cast and
        performance metrics.
    cast_column : str, default="cast"
        Column with comma-separated cast names.
    director_column : str, default="director"
        Column with director name.
    year_column : str, default="year"
        Column with release year.
    rating_column : str, optional
        Column with numeric rating.  If provided, used as part of
        the performance signal.
    boxoffice_column : str, optional
        Column with numeric box office.  If provided, used.
    half_life : float, optional
        Decay half-life in years.  Defaults to
        ``settings.BANKABILITY_DECAY_HALFLIFE_YEARS``.

    Returns
    -------
    pd.DataFrame
        Columns: ``actor``, ``type`` (actor/director),
        ``bankability_score``, ``film_count``, ``weighted_avg_rating``.
    """
    half_life = half_life or settings.BANKABILITY_DECAY_HALFLIFE_YEARS
    current_year = pd.to_numeric(df[year_column], errors="coerce").max()
    if pd.isna(current_year):
        current_year = 2024.0

    # Performance signal: combine rating and box office
    performance = pd.Series(1.0, index=df.index)
    if rating_column and rating_column in df.columns:
        r = pd.to_numeric(df[rating_column], errors="coerce").fillna(0)
        performance = performance * (r / 10.0)  # Normalize 0-10 → 0-1
    if boxoffice_column and boxoffice_column in df.columns:
        b = pd.to_numeric(df[boxoffice_column], errors="coerce").fillna(0)
        # Normalize: log transform to handle wide range
        b_norm = np.log1p(b) / np.log1p(b.max()) if b.max() > 0 else 0
        performance = performance * (1 + b_norm)

    scores: dict[str, dict] = {}

    def _add_film(person: str, person_type: str, weight: float, perf: float):
        if person.lower() == "nan" or not person.strip():
            return
        key = person.strip().lower()
        if key not in scores:
            scores[key] = {
                "name": person.strip(),
                "type": person_type,
                "weighted_sum": 0.0,
                "total_weight": 0.0,
                "film_count": 0,
            }
        scores[key]["weighted_sum"] += weight * perf
        scores[key]["total_weight"] += weight
        scores[key]["film_count"] += 1

    for _, row in df.iterrows():
        year = pd.to_numeric(row.get(year_column), errors="coerce")
        if pd.isna(year):
            continue
        w = _decay_weight(year, current_year, half_life)
        p = performance.loc[row.name]

        # Process cast
        cast_str = str(row.get(cast_column, ""))
        for actor in cast_str.split(","):
            _add_film(actor.strip(), "actor", w, p)

        # Process director
        director = str(row.get(director_column, ""))
        if director and director.lower() != "nan":
            _add_film(director.strip(), "director", w, p)

    # Compile results
    rows_list = []
    for key, data in scores.items():
        bankability = data["weighted_sum"] / data["total_weight"] if data["total_weight"] > 0 else 0.0
        rows_list.append(
            {
                "actor": data["name"],
                "type": data["type"],
                "bankability_score": round(bankability, 4),
                "film_count": data["film_count"],
            }
        )

    result = pd.DataFrame(rows_list).sort_values("bankability_score", ascending=False)
    logger.info(
        "Bankability scores computed for %d individuals.",
        len(result),
    )
    return result
