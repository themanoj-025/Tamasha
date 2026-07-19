"""Data-cleaning functions for the joined Tamasha dataset.

Handles:
- Parsing budget/collection currency strings into numeric ``₹`` values.
- Optionally inflation-adjusting across release years.
- Dropping duplicates, handling missing values, standardising column names.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Currency-parsing helpers ───────────────────────────────────────────

_INR_MULTIPLIERS: dict[str, float] = {
    "crore": 1e7,
    "cr": 1e7,
    "lakh": 1e5,
    "lac": 1e5,
    "k": 1e3,
    "million": 1e6,
    "billion": 1e9,
}

_CURRENCY_REGEX = re.compile(
    r"(?:₹|Rs\.?|INR)?\s*([\d,]+(?:\.\d+)?)\s*(crore|cr|lakh|lac|k|million|billion)?",
    re.IGNORECASE,
)


def parse_inr_value(text: str) -> Optional[float]:
    """Parse an INR currency string to a float value in ₹.

    Handles formats like ``₹100 Cr``, ``Rs. 50 lakh``, ``1.5 billion``.

    Parameters
    ----------
    text : str
        Raw currency string.

    Returns
    -------
    float or None
        Value in rupees, or None if parsing fails.
    """
    if pd.isna(text) or not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None

    match = _CURRENCY_REGEX.search(text)
    if not match:
        logger.debug("Could not parse currency: %s", text)
        return None

    number_str = match.group(1).replace(",", "")
    try:
        number = float(number_str)
    except ValueError:
        return None

    unit = (match.group(2) or "").lower()
    multiplier = _INR_MULTIPLIERS.get(unit, 1.0)
    return number * multiplier


def parse_currency_column(
    df: pd.DataFrame,
    column: str,
    new_name: Optional[str] = None,
) -> pd.Series:
    """Parse a column of INR currency strings into numeric rupees.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to parse.
    new_name : str, optional
        Name for the new numeric column.  If None, ``f"{column}_inr"``.

    Returns
    -------
    pd.Series
        Numeric series in rupees.
    """
    out = df[column].apply(parse_inr_value)
    out.name = new_name or f"{column}_inr"
    logger.info(
        "Parsed column '%s': %d / %d rows successfully parsed.",
        column,
        out.notna().sum(),
        len(out),
    )
    return out


# ── Inflation adjustment ──────────────────────────────────────────────

def _india_cpi_deflator(year: int) -> float:
    """Return a rough CPI-based deflator factor relative to 2024.

    Uses a simple linear approximation:
    ``deflator = 1.0 + 0.06 * (2024 - year)``

    This is a documented simplification — real CPI data would be better
    but is outside scope.  The method is easily replaceable.

    Parameters
    ----------
    year : int
        Release year.

    Returns
    -------
    float
        Multiplier to convert nominal ₹ to approximate 2024 ₹.
    """
    return 1.0 + 0.06 * (2024 - year)


def inflation_adjust(
    df: pd.DataFrame,
    value_column: str,
    year_column: str,
    base_year: int = 2024,
) -> pd.Series:
    """Adjust a currency column to ``base_year`` rupees.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    value_column : str
        Column with numeric rupee values.
    year_column : str
        Column with release years.
    base_year : int, default=2024
        Target year for adjustment.

    Returns
    -------
    pd.Series
        Inflation-adjusted values.
    """
    years = pd.to_numeric(df[year_column], errors="coerce")
    values = pd.to_numeric(df[value_column], errors="coerce")

    deflator = years.apply(
        lambda y: 1.0 + 0.06 * (base_year - y) if pd.notna(y) else np.nan
    )
    adjusted = values / deflator
    adjusted.name = f"{value_column}_adj{base_year}"
    logger.info(
        "Inflation-adjusted column '%s' to %d base year.",
        value_column,
        base_year,
    )
    return adjusted


# ── General cleaning ──────────────────────────────────────────────────

def clean_joined_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standard cleaning to the joined dataset.

    Steps:
    1. Drop duplicate rows (based on all columns).
    2. Drop columns that are entirely NaN.
    3. Standardise string columns (strip whitespace).
    4. Parse common currency columns if they exist.
    5. Sort by release year.

    Parameters
    ----------
    df : pd.DataFrame
        Raw joined DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    result = df.copy()

    # Drop full-NaN columns
    before = result.shape[1]
    result = result.dropna(axis=1, how="all")
    logger.info("Dropped %d all-NaN columns.", before - result.shape[1])

    # Drop duplicate rows
    before = result.shape[0]
    result = result.drop_duplicates()
    logger.info("Dropped %d duplicate rows.", before - result.shape[0])

    # Strip string columns
    for col in result.select_dtypes(include="object").columns:
        result[col] = result[col].astype(str).str.strip()

    # Parse currency columns if they exist
    for col in ["budget", "collection", "box_office", "worldwide_collection"]:
        if col in result.columns:
            new_col = f"{col}_inr"
            if new_col not in result.columns:
                result[new_col] = parse_currency_column(result, col)

    # Sort by year if available
    year_cols = [c for c in result.columns if "year" in c.lower()]
    if year_cols:
        year_col = year_cols[0]
        result = result.sort_values(year_col).reset_index(drop=True)

    logger.info("Cleaning complete: %s rows x %s cols.", result.shape[0], result.shape[1])
    return result
