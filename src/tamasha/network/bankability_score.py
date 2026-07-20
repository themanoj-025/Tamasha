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

    # ── Vectorized implementation (O(n) instead of iterrows O(n²)) ──
    # Explode cast into individual rows, compute decay weights, then groupby.
    df_work = df[[year_column, cast_column, director_column]].copy()
    df_work["_perf"] = performance
    df_work["_year"] = pd.to_numeric(df[year_column], errors="coerce")
    df_work = df_work.dropna(subset=["_year"])

    # Compute decay weight for each row (vectorized)
    df_work["_weight"] = 2.0 ** (-(current_year - df_work["_year"]) / half_life)
    df_work["_weighted_perf"] = df_work["_weight"] * df_work["_perf"]

    # Explode cast column — each actor gets a row per film
    cast_exploded = (
        df_work[cast_column]
        .astype(str)
        .str.split(",")
        .apply(pd.Series)
        .stack()
        .reset_index(level=1, drop=True)
        .to_frame("person")
    )
    cast_exploded["type"] = "actor"
    # Join back to get weight/perf
    cast_exploded = cast_exploded.join(df_work[["_weight", "_weighted_perf"]])
    cast_exploded = cast_exploded[~cast_exploded["person"].str.lower().isin(["nan", "none", ""])]
    cast_exploded["person"] = cast_exploded["person"].str.strip().str.lower()
    cast_exploded = cast_exploded[cast_exploded["person"] != ""]

    # Director rows
    dir_df = df_work[[director_column, "_weight", "_weighted_perf"]].copy()
    dir_df["person"] = dir_df[director_column].astype(str).str.strip().str.lower()
    dir_df["type"] = "director"
    dir_df = dir_df[~dir_df["person"].isin(["nan", "none", ""])]
    dir_df = dir_df[dir_df["person"] != ""]
    dir_df = dir_df.rename(columns={"_weight": "_weight", "_weighted_perf": "_weighted_perf"})

    # Combine actor + director
    combined = pd.concat(
        [
            cast_exploded[["person", "type", "_weight", "_weighted_perf"]],
            dir_df[["person", "type", "_weight", "_weighted_perf"]],
        ],
        ignore_index=True,
    )

    # Groupby person + type, compute weighted average
    grouped = (
        combined.groupby(["person", "type"], sort=False)
        .agg(
            weighted_sum=("_weighted_perf", "sum"),
            total_weight=("_weight", "sum"),
            film_count=("_weight", "count"),
        )
        .reset_index()
    )

    grouped["bankability_score"] = (
        (grouped["weighted_sum"] / grouped["total_weight"]).fillna(0.0).round(4)
    )

    # Map original-case names (use first occurrence from cast column)
    name_map: dict[str, str] = {}
    for _, row in df[[cast_column, director_column]].iterrows():
        for val in [str(row.get(cast_column, "")), str(row.get(director_column, ""))]:
            for part in val.split(","):
                stripped = part.strip()
                if stripped and stripped.lower() not in ("nan", "none", ""):
                    name_map[stripped.lower()] = stripped

    grouped["actor"] = grouped["person"].map(name_map).fillna(grouped["person"])

    result = (
        grouped[["actor", "type", "bankability_score", "film_count"]]
        .sort_values("bankability_score", ascending=False)
        .reset_index(drop=True)
    )

    logger.info("Bankability scores computed for %d individuals (vectorized).", len(result))
    return result
