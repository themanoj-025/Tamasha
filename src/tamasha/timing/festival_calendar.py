"""Festival calendar and release-window reference for Indian cinema.

Major Indian release windows include:
- **Eid** (movable, ~April–May)
- **Diwali** (movable, ~October–November)
- **Christmas** (December 25)
- **Independence Day** (August 15)
- **Republic Day** (January 26)
- **Holî** (movable, ~March)
- **New Year** (January 1)
- **Gandhi Jayanti** (October 2)
- **Dussehra** (movable, ~September–October)

The calendar uses approximate dates for movable festivals.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from tamasha.config import settings

logger = logging.getLogger(__name__)

# ── Festival date approximations ──────────────────────────────────────
# Movable festivals are approximated to a typical date range.
# For precise dates, a library like ``hijri-converter`` or an API
# would be needed.

_FESTIVALS: dict[str, tuple[int, int]] = {
    "New Year": (1, 1),
    "Republic Day": (1, 26),
    "Holî": (3, 14),  # Approximate
    "Eid": (4, 22),  # Approximate — varies by lunar calendar
    "Independence Day": (8, 15),
    "Dussehra": (10, 5),  # Approximate
    "Gandhi Jayanti": (10, 2),
    "Diwali": (10, 31),  # Approximate
    "Christmas": (12, 25),
}

_CURRENT_YEAR_FESTIVALS: dict[str, date] = {}


def _get_festival_date(name: str, year: int) -> Optional[date]:
    """Get an approximate date for a festival in a given year.

    Parameters
    ----------
    name : str
        Festival name.
    year : int
        Year.

    Returns
    -------
    date or None
        Approximate date, or None if unknown.
    """
    if name not in _FESTIVALS:
        return None
    month, day = _FESTIVALS[name]
    try:
        return date(year, month, day)
    except (ValueError, OverflowError):
        return None


def get_major_release_windows(year: int) -> dict[str, date]:
    """Return a dict of major release windows for a given year.

    Parameters
    ----------
    year : int
        Year of interest.

    Returns
    -------
    dict[str, date]
        ``{festival_name: approximate_date}``.
    """
    return {
        name: dt
        for name in _FESTIVALS
        if (dt := _get_festival_date(name, year)) is not None
    }


def is_festival_release(
    release_date: date,
    festival_windows: dict[str, date],
    window_days: int = 7,
) -> tuple[bool, Optional[str], int]:
    """Check if a release date falls near a major festival.

    Parameters
    ----------
    release_date : date
        Movie release date.
    festival_windows : dict[str, date]
        Map of festival name → approximate date.
    window_days : int, default=7
        Days before/after to consider as "festival release".

    Returns
    -------
    tuple[bool, str | None, int]
        ``(is_festival, festival_name, days_difference)``.
    """
    for name, fdate in festival_windows.items():
        diff = abs((release_date - fdate).days)
        if diff <= window_days:
            return True, name, diff
    return False, None, 0


def compute_festival_features(
    df: pd.DataFrame,
    date_column: str = "release_date",
    year_column: str = "year",
    window_days: int = 7,
) -> pd.DataFrame:
    """Compute festival-related features for a movie DataFrame.

    Adds columns:
    - ``is_festival_release``: bool
    - ``festival_name``: str (if applicable)
    - ``days_to_festival``: int

    Parameters
    ----------
    df : pd.DataFrame
        Movie DataFrame.
    date_column : str, default="release_date"
        Column with release dates (parsed by pandas).
    year_column : str, default="year"
        Column with release year (fallback if date parsing fails).
    window_days : int, default=7
        Days before/after a festival to flag.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added festival columns.
    """
    result = df.copy()
    result["is_festival_release"] = False
    result["festival_name"] = None
    result["days_to_festival"] = 0

    # Determine year for each row
    if date_column in result.columns:
        dates = pd.to_datetime(result[date_column], errors="coerce")
    else:
        dates = pd.NaT

    years_raw = result.get(year_column)
    if years_raw is None or years_raw.empty:
        # Try to find any year-like column
        year_candidates = [c for c in result.columns if "year" in c.lower()]
        if year_candidates:
            years_raw = result[year_candidates[0]]
            logger.info("  Using year column '%s' instead of '%s'", year_candidates[0], year_column)
        else:
            logger.warning("  No year column found. Skipping festival features.")
            return result
    years = pd.to_numeric(years_raw, errors="coerce")
    if not isinstance(years, pd.Series):
        logger.warning("  Year column is not a Series. Skipping festival features.")
        return result

    for idx in result.index:
        year_val = years.loc[idx]
        if pd.isna(year_val):
            continue
        year_int = int(year_val)

        # Get festival windows for that year
        windows = get_major_release_windows(year_int)

        # Determine release date
        if pd.notna(dates.loc[idx]):
            rel_date = dates.loc[idx].date()
        else:
            # Approximate to July 1 of that year if no exact date
            rel_date = date(year_int, 7, 1)

        is_fest, fest_name, days_diff = is_festival_release(rel_date, windows, window_days)
        result.at[idx, "is_festival_release"] = is_fest
        result.at[idx, "festival_name"] = fest_name
        result.at[idx, "days_to_festival"] = days_diff

    n_fest = result["is_festival_release"].sum()
    logger.info(
        "Festival features: %d / %d movies flagged as festival releases.",
        n_fest,
        len(result),
    )
    return result


def compute_clash_feature(
    df: pd.DataFrame,
    date_column: str = "release_date",
    clash_window_days: int = 7,
) -> pd.DataFrame:
    """Flag whether a movie clashes with another major release.

    A "clash" occurs when another movie in the dataset releases within
    ``clash_window_days`` days.

    Parameters
    ----------
    df : pd.DataFrame
        Movie DataFrame sorted by release date.
    date_column : str, default="release_date"
        Column with release dates.
    clash_window_days : int, default=7
        Window in days for considering a clash.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with ``has_clash`` column added.
    """
    result = df.copy()
    dates = pd.to_datetime(result.get(date_column), errors="coerce")
    result["has_clash"] = False

    # O(n log n) sort-by-date sweep instead of O(n²) nested loops
    # Sort by date, then for each movie only check nearby neighbors
    valid_mask = pd.notna(dates)
    valid_dates = dates[valid_mask].sort_values()
    valid_indices = valid_dates.index.tolist()

    for i, idx in enumerate(valid_indices):
        d1 = valid_dates.loc[idx]
        # Check neighbors forward (dates are sorted, so once diff > window, we break)
        for j in range(i + 1, len(valid_indices)):
            d2 = valid_dates.loc[valid_indices[j]]
            diff = (d2 - d1).days
            if diff > clash_window_days:
                break  # All remaining dates are further away
            if 0 < diff <= clash_window_days:
                result.at[idx, "has_clash"] = True
                break

    n_clash = result["has_clash"].sum()
    logger.info("Clash feature: %d / %d movies have a clash.", n_clash, len(result))
    return result
