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

import asyncio
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

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


class TMDbServerError(Exception):
    """Raised when TMDb returns a 5xx server error (retryable)."""


def _build_params(title: str, year: Optional[int] = None) -> dict[str, Any]:
    """Build TMDb search parameters."""
    params: dict[str, Any] = {"query": title}
    if year is not None:
        params["primary_release_year"] = year
        params["year"] = year
    if not _TMDB_ACCESS_TOKEN and _TMDB_API_KEY:
        params["api_key"] = _TMDB_API_KEY
    return params


@retry(
    retry=retry_if_exception_type((
        requests.Timeout,
        requests.ConnectionError,
        TMDbServerError,
    )),
    wait=wait_exponential_jitter(initial=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _fetch_tmdb(title: str, year: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Fetch TMDb search results with retry/backoff via tenacity.

    Retryable: timeouts, connection errors, 5xx server errors.
    Non-retryable (fail fast): 4xx (except 429), malformed JSON.

    The 429 case is handled separately — we respect Retry-After but
    don't use tenacity for it, since 429 has its own semantics.

    Parameters
    ----------
    title : str
        Movie title.
    year : int, optional
        Release year for filtering.

    Returns
    -------
    dict or None
        The first TMDb result, or None if no match.
    """
    if not _TMDB_API_KEY and not _TMDB_ACCESS_TOKEN:
        return None

    params = _build_params(title, year)
    _rate_limit()

    resp = requests.get(_SEARCH_URL, headers=_HEADERS, params=params, timeout=10)

    # Handle 429 Rate Limit separately — respect Retry-After header
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 2))
        logger.warning("Rate limited by TMDb. Waiting %ds (Retry-After)...", retry_after)
        time.sleep(retry_after)
        _rate_limit()
        resp = requests.get(_SEARCH_URL, headers=_HEADERS, params=params, timeout=10)

    # 5xx → raise for tenacity to retry
    if resp.status_code >= 500:
        raise TMDbServerError(f"TMDb returned {resp.status_code}: {resp.text[:200]}")

    resp.raise_for_status()  # 4xx (except 429) → raises immediately, no retry

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None

    if year is not None:
        for r in results:
            rd = r.get("release_date", "")
            if rd and rd.startswith(str(year)):
                return r

    return results[0]


def _search_tmdb(title: str, year: Optional[int] = None) -> Optional[dict[str, Any]]:
    """Search TMDb for a movie by title and year.

    Wraps :func:`_fetch_tmdb` with cache handling. The tenacity
    retry/backoff logic lives in ``_fetch_tmdb``.

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
    try:
        return _fetch_tmdb(title, year)
    except Exception as exc:
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


# ── Async enrichment ──────────────────────────────────────────────────


def _fetch_one_async(
    idx: int,
    title: str,
    year: Optional[int],
    cache: dict[str, Any],
    sem: asyncio.Semaphore,
    client: Any,
) -> tuple[int, str, str]:
    """Fetch a single movie's TMDb data asynchronously.

    Returns ``(idx, plot, date)``.
    """
    cache_key = f"{title.strip().lower()}|{year}" if year else title.strip().lower()
    if cache_key in cache:
        data = cache[cache_key]
        has_plot = bool(data and data.get("overview", "").strip())
        has_date = bool(data and data.get("release_date", "").strip())
        return (
            idx,
            data["overview"] if data and has_plot else "",
            data["release_date"] if data and has_date else "",
        )

    async with sem:
        result = _search_tmdb(title, year)
        if result is None:
            cache[cache_key] = None
            return (idx, "", "")

        data: dict[str, Any] = {
            "title": result.get("title", title),
            "overview": result.get("overview", ""),
            "release_date": result.get("release_date", ""),
            "poster_path": result.get("poster_path"),
            "tmdb_id": result.get("id"),
        }
        cache[cache_key] = data

        has_plot = bool(data.get("overview", "").strip())
        has_date = bool(data.get("release_date", "").strip())
        return (
            idx,
            data["overview"] if has_plot else "",
            data["release_date"] if has_date else "",
        )


async def _enrich_async(
    titles: list[str],
    years: list[Optional[int]],
    cache: dict[str, Any],
    concurrency: int = 8,
) -> list[tuple[int, str, str]]:
    """Run async enrichment with bounded concurrency.

    Parameters
    ----------
    titles : list[str]
        Movie titles.
    years : list[Optional[int]]
        Corresponding years.
    cache : dict
        Shared cache dict.
    concurrency : int, default=8
        Max concurrent TMDb API calls. 8 is chosen to stay within
        TMDb's rate limits while maximizing throughput.

    Returns
    -------
    list[tuple[int, str, str]]
        Results as ``(idx, plot, date)`` tuples.
    """
    sem = asyncio.Semaphore(concurrency)
    # _search_tmdb uses requests (sync), so we run it in a thread pool
    # to avoid blocking the event loop. httpx.AsyncClient would be
    # faster but requires refactoring the entire _search_tmdb function.
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        tasks = [
            loop.run_in_executor(
                pool,
                _search_tmdb,
                titles[i],
                years[i],
            )
            for i in range(len(titles))
        ]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    output: list[tuple[int, str, str]] = []
    for i, result in enumerate(results_raw):
        if isinstance(result, Exception):
            logger.debug("Async enrichment failed for %s: %s", titles[i], result)
            output.append((i, "", ""))
            continue
        if result is None:
            cache_key = f"{titles[i].strip().lower()}|{years[i]}" if years[i] else titles[i].strip().lower()
            cache[cache_key] = None
            output.append((i, "", ""))
            continue
        data = result
        has_plot = bool(data.get("overview", "").strip())
        has_date = bool(data.get("release_date", "").strip())
        output.append((
            i,
            data.get("overview", "") if has_plot else "",
            data.get("release_date", "") if has_date else "",
        ))
    return output


def enrich_dataset_async(
    df: Any,
    title_column: str = "title",
    year_column: Optional[str] = None,
    max_movies: Optional[int] = None,
    concurrency: int = 8,
) -> tuple[dict[str, list[str]], "pd.DataFrame"]:
    """Enrich a movie DataFrame with TMDb data using async I/O.

    Uses :func:`enrich_dataset` internally — the async variant runs
    ``_search_tmdb`` calls in a thread pool with bounded concurrency
    instead of sequential blocking.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to enrich.
    title_column : str, default="title"
        Column with movie titles.
    year_column : str, optional
        Column with release years.
    max_movies : int, optional
        Limit for testing.
    concurrency : int, default=8
        Max concurrent API calls.

    Returns
    -------
    tuple[dict, pd.DataFrame]
        Same signature as :func:`enrich_dataset`.
    """
    _load_cache()

    result_df = df.copy()
    total = min(len(result_df), max_movies) if max_movies else len(result_df)

    titles = [str(result_df.iloc[i][title_column]) for i in range(total)]
    years_list: list[Optional[int]] = []
    for i in range(total):
        if year_column and year_column in result_df.columns:
            try:
                years_list.append(int(float(result_df.iloc[i][year_column])))
            except (ValueError, TypeError):
                years_list.append(None)
        else:
            years_list.append(None)

    logger.info("Enriching %d movies from TMDb (async, concurrency=%d)...", total, concurrency)

    # Run the async enrichment
    results = asyncio.run(_enrich_async(titles, years_list, _CACHE, concurrency))

    plots = [""] * total
    dates = [""] * total
    match_count = 0
    for idx, plot, date in results:
        plots[idx] = plot
        dates[idx] = date
        if plot.strip() or date.strip():
            match_count += 1
        if (idx + 1) % 50 == 0:
            logger.info("  Progress: %d/%d (%d matched)", idx + 1, total, match_count)
            _save_cache()

    result_df["plot_summary"] = plots
    result_df["release_date"] = dates

    _save_cache()

    plot_coverage = sum(1 for p in plots if p.strip()) / total * 100 if total else 0
    date_coverage = sum(1 for d in dates if d.strip()) / total * 100 if total else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("TMDb Enrichment Complete (async)")
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
