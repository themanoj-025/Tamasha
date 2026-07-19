"""Functions for loading raw datasets from disk.

Expected raw file names are defined here so that callers don't
hardcode paths.  After the Kaggle downloads are complete, run
``load_imdb_india()`` and ``load_bollywood_boxoffice()`` to get
the unmodified DataFrames.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from tamasha.config import settings

logger = logging.getLogger(__name__)

# ── Column name mappings ──────────────────────────────────────────────
# Map actual dataset column names to canonical names used internally.
# This lets the rest of the codebase speak a consistent language.

IMDB_COLUMN_MAP: dict[str, str] = {
    "Name": "title",
    "Year": "year_raw",
    "Duration": "duration_raw",
    "Genre": "genre",
    "Rating": "rating",
    "Votes": "votes",
    "Director": "director",
    "Actor 1": "actor_1",
    "Actor 2": "actor_2",
    "Actor 3": "actor_3",
}

BOXOFFICE_COLUMN_MAP: dict[str, str] = {
    "Movie": "title",
    "Worldwide": "worldwide_collection_inr",
    "India Net": "india_net_collection_inr",
    "India Gross": "india_gross_collection_inr",
    "Overseas": "overseas_collection_inr",
    "Budget": "budget_inr",
    "Verdict": "verdict",
}


def _parse_imdb_year(year_raw: str) -> Optional[int]:
    """Parse IMDB year string like ``"(2019)"`` into integer 2019.

    Parameters
    ----------
    year_raw : str
        Raw year string from the IMDB dataset.

    Returns
    -------
    int or None
        Parsed year, or None if unparseable.
    """
    if pd.isna(year_raw):
        return None
    match = re.search(r"(\d{4})", str(year_raw))
    return int(match.group(1)) if match else None


def _parse_imdb_duration(duration_raw: str) -> Optional[int]:
    """Parse IMDB duration string like ``"109 min"`` into integer minutes.

    Parameters
    ----------
    duration_raw : str
        Raw duration string.

    Returns
    -------
    int or None
        Duration in minutes, or None if unparseable.
    """
    if pd.isna(duration_raw):
        return None
    match = re.search(r"(\d+)\s*min", str(duration_raw))
    return int(match.group(1)) if match else None


def load_imdb_india(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the IMDB India Movies dataset.

    The raw dataset uses ``latin1`` encoding and has columns
    ``Name``, ``Year``, ``Duration``, ``Genre``, ``Rating``,
    ``Votes``, ``Director``, ``Actor 1``, ``Actor 2``, ``Actor 3``.
    These are renamed to canonical names and the ``Year`` / ``Duration``
    columns are parsed into numeric form.

    Parameters
    ----------
    path : Path, optional
        Full path to the CSV.  Defaults to
        ``settings.DATA_RAW / "IMDb Movies India.csv"``.

    Returns
    -------
    pd.DataFrame
        IMDB India dataset with canonical column names.
        Contains a ``year`` (int) and ``duration_minutes`` (int) column.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    """
    path = path or settings.DATA_RAW / "IMDb Movies India.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"IMDB India Movies CSV not found at {path}. "
            f"Please download from Kaggle first."
        )
    df = pd.read_csv(path, encoding="latin1")
    logger.info("Loaded IMDB India: %s rows x %s cols", df.shape[0], df.shape[1])

    # Rename columns to canonical names
    df = df.rename(columns=IMDB_COLUMN_MAP)
    # Keep only mapped columns
    canonical_cols = list(IMDB_COLUMN_MAP.values())
    df = df[[c for c in canonical_cols if c in df.columns]]

    # Parse year
    df["year"] = df["year_raw"].apply(_parse_imdb_year)
    df = df.drop(columns=["year_raw"])

    # Parse duration
    df["duration_minutes"] = df["duration_raw"].apply(_parse_imdb_duration)
    df = df.drop(columns=["duration_raw"])

    # Build a unified cast column from Actor 1/2/3
    cast_cols = [c for c in ["actor_1", "actor_2", "actor_3"] if c in df.columns]
    df["cast"] = df[cast_cols].apply(
        lambda row: ", ".join(
            c for c in row if pd.notna(c) and str(c).strip().lower() != "nan"
        ),
        axis=1,
    )
    df = df.drop(columns=cast_cols)

    logger.info(
        "IMDB India post-process: %s rows x %s cols, %d with ratings",
        df.shape[0],
        df.shape[1],
        df["rating"].notna().sum(),
    )
    return df


def load_bollywood_boxoffice(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the Bollywood Box Office dataset.

    Columns: ``Movie``, ``Worldwide``, ``India Net``, ``India Gross``,
    ``Overseas``, ``Budget``, ``Verdict``.

    Parameters
    ----------
    path : Path, optional
        Full path to the CSV.  Defaults to
        ``settings.DATA_RAW / "Top 1000 Bollywood Movies and their boxoffice.csv"``.

    Returns
    -------
    pd.DataFrame
        Box Office dataset with canonical column names.
        All monetary values remain in raw integer form (₹).

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist at the given path.
    """
    path = path or settings.DATA_RAW / "Top 1000 Bollywood Movies and their boxoffice.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Bollywood Box Office CSV not found at {path}. "
            f"Please download from Kaggle first."
        )
    df = pd.read_csv(path)
    logger.info("Loaded Bollywood Box Office: %s rows x %s cols", df.shape[0], df.shape[1])

    # Rename columns to canonical names
    df = df.rename(columns=BOXOFFICE_COLUMN_MAP)
    # Keep only mapped columns (skip Unnamed: 0, SN)
    canonical_cols = list(BOXOFFICE_COLUMN_MAP.values())
    df = df[[c for c in canonical_cols if c in df.columns]]

    logger.info(
        "Box Office post-process: %s rows x %s cols",
        df.shape[0],
        df.shape[1],
    )
    return df


def list_raw_files() -> list[Path]:
    """List all CSV files in the raw data directory.

    Returns
    -------
    list[Path]
        Sorted list of CSV file paths.
    """
    return sorted(settings.DATA_RAW.glob("*.csv"))
