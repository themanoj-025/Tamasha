"""TMDb API enrichment for plot summaries and release dates.

Used by the training pipeline to fill in missing plot text (Stage 5) and
release-date columns (Stage 7) that the original datasets lack.

All responses are cached locally to avoid repeated API calls and respect
TMDb's fair-use policy.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import requests
from dotenv import load_dotenv

from tamasha.config import settings

logger = logging.getLogger(__name__)

# ── Auth ──────────────────────────────────────────────────────────────

load_dotenv(settings.PROJECT_ROOT / ".env")

_TMDB_API_KEY: str | None = os.getenv("TMDB_API_KEY")
_TMDB_ACCESS_TOKEN: str | None = os.getenv("TMDB_ACCESS_TOKEN")

_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
_HEADERS: dict[str, str] = {
    "accept": "application/json",
}
if _TMDB_ACCESS_TOKEN:
    _HEADERS["Authorization"] = f"Bearer {_TMDB_ACCESS_TOKEN}"

# ── Rate limiting ────────────────────────────────────────────────────

_LAST_REQUEST_TIME: float = 0.0
_MIN_INTERVAL_S = 0.25  # 4 requests per second — well within TMDb limits

_CACHE_PATH: Path = settings.DATA_PROCESSED / "tmdb_cache.json"
_CACHE: dict[str, dict[str, Any]] = {}


def _load_cache() -> None:
    """Load the local TMDb cache from disk."""
    global _CACHE
    if _CACHE_PATH.exists():
        try:
            _CACHE = json.loads(_CACHE_PATH.read_text())
            logger.info("Loaded TMDb cache: %d entries", len(_CACHE))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Cache load failed (%s); starting fresh.", exc)
            _CACHE = {}
    else:
        _CACHE = {}


def _save_cache() -> None:
    """Persist the cache to disk."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_CACHE, indent=2))
    logger.debug("Cache saved (%d entries).", len(_CACHE))


def _rate_limit() -> None:
    """Ensure minimum interval between API requests."""
    global _LAST_REQUEST_TIME
    elapsed = time.perf_counter() - _LAST_REQUEST_TIME
    if elapsed < _MIN_INTERVAL_S:
        time.sleep(_MIN_INTERVAL_S - elapsed)
    _LAST_REQUEST_TIME = time.perf_counter()


# ── API call ──────────────────────────────────────────────────────────


def _search_tmdb(title: str, year: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Search TMDb for a movie by title and year.

    Parameters
    ----------
    title : str
        Movie title.
    year : int, optional
        Release year for filtering.

    Returns
    -------
    dict or None
        The first result's movie object, or None if no match.
    """
    if not _TMDB_API_KEY and not _TMDB_ACCESS_TOKEN:
        logger.warning("No TMDb credentials found. Set TMDB_API_KEY or TMDB_ACCESS_TOKEN in .env")
        return None

    params: dict[str, Any] = {"query": title}
    if year is not None:
        params["primary_release_year"] = year
        params["year"] = year

    # Fallback to API key in query params if bearer token isn't available
    if not _TMDB_ACCESS_TOKEN and _TMDB_API_KEY:
        params["api_key"] = _TMDB_API_KEY

    _rate_limit()
    try:
        resp = requests.get(
            _SEARCH_URL,
            headers=_HEADERS,
            params=params,
            timeout=10,
        )
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 2))
            logger.warning("Rate limited. Waiting %ds...", retry_after)
            time.sleep(retry_after)
            resp = requests.get(
                _SEARCH_URL, headers=_HEADERS, params=params, timeout=10
            )

        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None

        # If we queried with a year, prefer result that matches the year
        if year is not None:
            for r in results:
                rd = r.get("release_date", "")
                if rd and rd.startswith(str(year)):
                    return r

        return results[0]

    except requests.RequestException as exc:
        logger.debug("TMDb search failed for '%s' (%s): %s", title, year, exc)
        return None


# ── Public API ────────────────────────────────────────────────────────


def get_movie_data(title: str, year: Optional[int] = None, force: bool = False) -> Optional[dict[str, Any]]:
    """Get plot summary, release date, and poster path for a movie.

    Results are cached locally.  Subsequent calls for the same ``(title, year)``
    return the cached result without hitting the API.

    Parameters
    ----------
    title : str
        Movie title.
    year : int, optional
        Release year for disambiguation.
    force : bool, default=False
        If True, bypass cache and re-fetch from API.

    Returns
    -------
    dict or None
        ``{"title": str, "overview": str, "release_date": str, "poster_path": str | None}``
        or ``None`` if not found.
    """
    cache_key = f"{title.strip().lower()}|{year}" if year else title.strip().lower()
    if not force and cache_key in _CACHE:
        return _CACHE[cache_key]

    result = _search_tmdb(title.strip(), year)
    if result is None:
        _CACHE[cache_key] = None
        return None

    data: dict[str, Any] = {
        "title": result.get("title", title),
        "overview": result.get("overview", ""),
        "release_date": result.get("release_date", ""),
        "poster_path": result.get("poster_path"),
        "tmdb_id": result.get("id"),
    }

    _CACHE[cache_key] = data
    return data


def enrich_dataset(
    df: Any,
    title_column: str = "title",
    year_column: Optional[str] = None,
    max_movies: Optional[int] = None,
) -> tuple[dict[str, list[str]], "pd.DataFrame"]:
    """Enrich a movie DataFrame with TMDb data.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to enrich.
    title_column : str, default="title"
        Column with movie titles.
    year_column : str, optional
        Column with release years.
    max_movies : int, optional
        Limit for testing (e.g. first 200 rows).

    Returns
    -------
    tuple[dict, pd.DataFrame]
        ``({"plots": [...], "dates": [...]}, enriched_df)``
        where the dict has matched columns and ``enriched_df`` has
        new ``plot_summary`` and ``release_date`` columns.
    """
    # Load cache once at the start
    _load_cache()

    result_df = df.copy()
    plots: list[str] = []
    dates: list[str] = []
    match_count = 0
    total = min(len(result_df), max_movies) if max_movies else len(result_df)

    logger.info("Enriching %d movies from TMDb...", total)

    for idx in range(total):
        row = result_df.iloc[idx]
        title = str(row[title_column])
        year: Optional[int] = None
        if year_column and year_column in result_df.columns:
            try:
                year = int(float(row[year_column]))
            except (ValueError, TypeError):
                year = None

        data = get_movie_data(title, year)
        has_plot = bool(data and data.get("overview", "").strip())
        has_date = bool(data and data.get("release_date", "").strip())

        plots.append(data["overview"] if data and has_plot else "")
        dates.append(data["release_date"] if data and has_date else "")

        if has_plot or has_date:
            match_count += 1

        if (idx + 1) % 50 == 0:
            logger.info("  Progress: %d/%d (%d matched)", idx + 1, total, match_count)
            _save_cache()  # Periodic save

    result_df["plot_summary"] = plots
    result_df["release_date"] = dates

    _save_cache()

    plot_coverage = sum(1 for p in plots if p.strip()) / total * 100 if total else 0
    date_coverage = sum(1 for d in dates if d.strip()) / total * 100 if total else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("TMDb Enrichment Complete")
    logger.info("  Movies attempted:   %d", total)
    logger.info("  Plot coverage:      %.1f%% (%d with plot)", plot_coverage, sum(1 for p in plots if p.strip()))
    logger.info("  Date coverage:      %.1f%% (%d with date)", date_coverage, sum(1 for d in dates if d.strip()))
    logger.info("  Overall matches:    %.1f%% (%d/%d)", match_count / total * 100, match_count, total)
    logger.info("=" * 60)

    coverage = {
        "plots": [p for p in plots if p.strip()],
        "dates": [d for d in dates if d.strip()],
    }

    return coverage, result_df
