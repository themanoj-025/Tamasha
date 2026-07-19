"""Feature engineering for movie-level prediction.

Transforms the cleaned joined DataFrame into a feature matrix suitable
for rating and box-office models.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from tamasha.config import settings

logger = logging.getLogger(__name__)


def extract_genre_features(
    df: pd.DataFrame,
    genre_column: str = "genre",
    max_genres: Optional[int] = None,
) -> pd.DataFrame:
    """One-hot encode the genre column (assumed comma-separated).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    genre_column : str, default="genre"
        Column with comma-separated genres.
    max_genres : int, optional
        If set, keep only the top-N most frequent genres.

    Returns
    -------
    pd.DataFrame
        Genre dummies (bool).
    """
    genres = df[genre_column].fillna("").astype(str).str.split(r"\s*,\s*")
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(genres)
    genre_df = pd.DataFrame(
        encoded, index=df.index, columns=[f"genre_{g}" for g in mlb.classes_]
    )

    if max_genres and genre_df.shape[1] > max_genres:
        # Keep only top-N by frequency
        col_sums = genre_df.sum().sort_values(ascending=False)
        keep = col_sums.head(max_genres).index
        genre_df = genre_df[keep]
        logger.info("Kept top %d genres: %s", max_genres, list(keep))

    logger.info("Genre features: %s columns", genre_df.shape[1])
    return genre_df


def extract_cast_features(
    df: pd.DataFrame,
    cast_column: str = "cast",
    director_column: str = "director",
) -> pd.DataFrame:
    """Extract cast-related features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cast_column : str, default="cast"
        Column with cast list (comma-separated or list-like).
    director_column : str, default="director"
        Column with director name.

    Returns
    -------
    pd.DataFrame
        Features: cast_size, director (label-encoded).
    """
    result = pd.DataFrame(index=df.index)

    # Cast size
    cast = df[cast_column].fillna("").astype(str).str.split(r"\s*,\s*")
    result["cast_size"] = cast.apply(len)

    # Director label encoding
    le = LabelEncoder()
    directors = df[director_column].fillna("Unknown Director").astype(str)
    result["director_encoded"] = le.fit_transform(directors)

    logger.info("Cast features extracted.")
    return result


def extract_runtime_feature(
    df: pd.DataFrame,
    runtime_column: str = "runtime",
) -> pd.DataFrame:
    """Extract runtime feature (fill NaNs with median).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    runtime_column : str, default="runtime"
        Column with runtime in minutes.

    Returns
    -------
    pd.DataFrame
        Single-column DataFrame with runtime.
    """
    result = pd.DataFrame(index=df.index)
    runtime = pd.to_numeric(df[runtime_column], errors="coerce")
    median_runtime = runtime.median()
    runtime = runtime.fillna(median_runtime)
    result["runtime_minutes"] = runtime
    logger.info(
        "Runtime feature: median=%.1f, %d NaN filled.",
        median_runtime,
        runtime.isna().sum(),
    )
    return result


def extract_decade_feature(
    df: pd.DataFrame,
    year_column: str = "year",
) -> pd.DataFrame:
    """Extract decade dummies from release year.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    year_column : str, default="year"
        Column with release year.

    Returns
    -------
    pd.DataFrame
        Decade dummies (e.g., ``decade_2000``, ``decade_2010``).
    """
    result = pd.DataFrame(index=df.index)
    years = pd.to_numeric(df[year_column], errors="coerce")
    decades = (years // 10 * 10).astype("Int64")
    decade_dummies = pd.get_dummies(decades, prefix="decade", dummy_na=False)
    result = pd.concat([result, decade_dummies], axis=1)
    logger.info("Decade features extracted: %s columns", result.shape[1])
    return result


def build_feature_matrix(
    df: pd.DataFrame,
    include_genre: bool = True,
    include_cast: bool = True,
    include_runtime: bool = True,
    include_decade: bool = True,
    include_budget: bool = True,
    target_column_rating: Optional[str] = None,
    target_column_boxoffice: Optional[str] = None,
) -> tuple[pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Build a complete feature matrix (X) and optional target vectors.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned joined DataFrame.
    include_genre : bool, default=True
        Include one-hot genre features.
    include_cast : bool, default=True
        Include cast features.
    include_runtime : bool, default=True
        Include runtime feature.
    include_decade : bool, default=True
        Include decade dummies.
    include_budget : bool, default=True
        Include budget feature.
    target_column_rating : str, optional
        Column name for the rating target.
    target_column_boxoffice : str, optional
        Column name for the box-office target.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None, pd.Series | None]
        ``(X, y_rating, y_boxoffice)``.
    """
    parts: list[pd.DataFrame] = []

    if include_genre:
        genre_col = [c for c in df.columns if "genre" in c.lower()]
        if genre_col:
            parts.append(extract_genre_features(df, genre_column=genre_col[0]))
        else:
            logger.warning("No genre column found; skipping genre features.")

    if include_cast:
        cast_col = [c for c in df.columns if c.lower() == "cast"]
        director_col = [c for c in df.columns if c.lower() == "director"]
        if cast_col:
            parts.append(
                extract_cast_features(
                    df,
                    cast_column=cast_col[0],
                    director_column=director_col[0] if director_col else "director",
                )
            )
        else:
            logger.warning("No cast column found; skipping cast features.")

    if include_runtime:
        runtime_col = [c for c in df.columns if "runtime" in c.lower()]
        if runtime_col:
            parts.append(extract_runtime_feature(df, runtime_column=runtime_col[0]))

    if include_decade:
        year_col = [c for c in df.columns if "year" in c.lower()]
        if year_col:
            parts.append(extract_decade_feature(df, year_column=year_col[0]))

    if include_budget:
        budget_cols_inr = [c for c in df.columns if "budget_inr" in c.lower()]
        if budget_cols_inr:
            bcol = budget_cols_inr[0]
            budget = pd.to_numeric(df[bcol], errors="coerce").fillna(0)
            parts.append(
                pd.DataFrame({"budget_inr": budget}, index=df.index)
            )

    X = pd.concat(parts, axis=1)

    # Targets
    y_rating = None
    if target_column_rating:
        y_col = target_column_rating
        # Also try common variations
        if y_col not in df.columns:
            candidates = [c for c in df.columns if "rating" in c.lower()]
            if candidates:
                y_col = candidates[0]
        if y_col in df.columns:
            y_rating = pd.to_numeric(df[y_col], errors="coerce")

    y_boxoffice = None
    if target_column_boxoffice:
        y_col = target_column_boxoffice
        if y_col not in df.columns:
            candidates = [c for c in df.columns if "collection_inr" in c.lower()]
            if candidates:
                y_col = candidates[0]
        if y_col in df.columns:
            y_boxoffice = pd.to_numeric(df[y_col], errors="coerce")
        elif "box_office_inr" in df.columns:
            y_boxoffice = pd.to_numeric(df["box_office_inr"], errors="coerce")

    logger.info("Feature matrix built: %s rows x %s cols", X.shape[0], X.shape[1])
    if y_rating is not None:
        logger.info("Rating target: %d non-null", y_rating.notna().sum())
    if y_boxoffice is not None:
        logger.info("Box-office target: %d non-null", y_boxoffice.notna().sum())

    return X, y_rating, y_boxoffice
