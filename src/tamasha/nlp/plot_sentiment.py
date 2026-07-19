"""Sentiment and tone analysis on movie plot summaries.

Supports VADER (lightweight, offline) and a configurable Hugging Face
model.  The main entry point is ``score_plot_sentiment()`` which
returns per-movie sentiment scores.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import nltk
import numpy as np
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from tamasha.config import settings

logger = logging.getLogger(__name__)

# ── Lazy initialisation ───────────────────────────────────────────────

_SENTIMENT_ANALYZER: Any = None


def _get_analyzer() -> Any:
    """Get or initialise the sentiment analyzer.

    Returns
    -------
    SentimentIntensityAnalyzer
        VADER sentiment analyzer.
    """
    global _SENTIMENT_ANALYZER
    if _SENTIMENT_ANALYZER is None:
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        _SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()
        logger.info("VADER sentiment analyzer initialised.")
    return _SENTIMENT_ANALYZER


def score_plot_sentiment(
    df: pd.DataFrame,
    plot_column: str = "plot",
) -> pd.DataFrame:
    """Score the sentiment of each plot summary.

    Returns a DataFrame with columns:
    - ``compound``: overall sentiment (−1 to +1)
    - ``pos``, ``neu``, ``neg``: proportional scores (0 to 1)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a plot/text column.
    plot_column : str, default="plot"
        Column containing plot summaries.

    Returns
    -------
    pd.DataFrame
        Sentiment scores indexed the same as ``df``.
    """
    analyzer = _get_analyzer()
    plots = df[plot_column].fillna("").astype(str)

    scores = plots.apply(lambda text: analyzer.polarity_scores(text))
    sentiment_df = pd.DataFrame(scores.tolist(), index=df.index)

    logger.info("Plot sentiment scored for %d movies.", len(sentiment_df))
    return sentiment_df


def genre_conditional_correlation(
    df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    target_column: str,
    genre_column: str = "genre",
    min_samples: int = 10,
) -> pd.DataFrame:
    """Compute sentiment-target correlation within each genre.

    This is the key analysis: genre-conditional correlation reveals
    whether a "dark" tone helps or hurts *within* a specific genre.

    Parameters
    ----------
    df : pd.DataFrame
        Movie DataFrame.
    sentiment_df : pd.DataFrame
        Sentiment scores from :func:`score_plot_sentiment`.
    target_column : str
        Numeric column to correlate against (rating or box office).
    genre_column : str, default="genre"
        Column with comma-separated genres.
    min_samples : int, default=10
        Minimum movies per genre to report.

    Returns
    -------
    pd.DataFrame
        Genre-level correlation table.
    """
    genres = df[genre_column].fillna("").astype(str).str.split(r"\s*,\s*")
    target = pd.to_numeric(df[target_column], errors="coerce")
    compound = sentiment_df["compound"]

    rows: list[dict[str, Any]] = []
    unique_genres = set()
    for g_list in genres:
        unique_genres.update(g.strip() for g in g_list if g.strip())

    for genre in sorted(unique_genres):
        mask = genres.apply(lambda gl: genre in gl)
        n = mask.sum()
        if n < min_samples:
            continue

        t = target[mask]
        c = compound[mask]
        valid = t.notna() & c.notna()
        if valid.sum() < min_samples:
            continue

        corr = t[valid].corr(c[valid])
        rows.append(
            {
                "genre": genre,
                "n_movies": int(valid.sum()),
                "correlation": round(corr, 4) if pd.notna(corr) else None,
                "mean_compound": round(c[valid].mean(), 4),
            }
        )

    result = pd.DataFrame(rows).sort_values("correlation", ascending=False, na_position="last")
    logger.info("Genre-conditional correlation computed for %d genres.", len(result))
    return result
