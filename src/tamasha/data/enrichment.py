"""TMDb dataset enrichment - poster/photo URLs and async batch enrichment.

Uses the TMDb client from :mod:"tmdb_client" for API access.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, cast

import httpx
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from tamasha.data.tmdb_client import (
    _CACHE,
    _HEADERS,
    _IMAGE_BASE,
    _PERSON_SEARCH_URL,
    _SEARCH_URL,
    _TMDB_ACCESS_TOKEN,
    _TMDB_API_KEY,
    TMDbCircuitBreaker,
    TMDbServerError,
    _build_params,
    _load_cache,
    _rate_limit,
    _save_cache,
    get_movie_data,
)

logger = logging.getLogger(__name__)

_tmdb_breaker = TMDbCircuitBreaker()

def get_poster_url(title: str, year: int | None = None, size: str = "w500") -> str | None:
    """Get a movie poster URL from TMDb.

    Uses the existing cached TMDb data if available, otherwise hits the API.
    Returns ``None`` if no poster found or API unavailable.

    Parameters
    ----------
    title : str
        Movie title.
    year : int, optional
        Release year for disambiguation.
    size : str, default="w500"
        TMDb image size ("w92", "w154", "w185", "w342", "w500", "w780", "original").

    Returns
    -------
    str or None
        Full poster URL (e.g. ``https://image.tmdb.org/t/p/w500/abc.jpg``)
        or ``None`` if not found.
    """
    data = get_movie_data(title, year)
    if data is None or not data.get("poster_path"):
        return None
    return f"{_IMAGE_BASE}/{size}{data['poster_path']}"


def get_actor_photo_url(name: str, size: str = "w185") -> str | None:
    """Search TMDb for an actor/director and return their profile photo URL.

    Parameters
    ----------
    name : str
        Actor or director name.
    size : str, default="w185"
        TMDb image size ("w45", "w185", "h632", "original").

    Returns
    -------
    str or None
        Full photo URL or ``None`` if not found.
    """
    if not _TMDB_API_KEY and not _TMDB_ACCESS_TOKEN:
        return None

    # Check circuit breaker
    if _tmdb_breaker.is_open():
        logger.warning("TMDb circuit breaker open — skipping actor photo search for '%s'", name)
        return None

    params: dict[str, Any] = {"query": name}
    if not _TMDB_ACCESS_TOKEN and _TMDB_API_KEY:
        params["api_key"] = _TMDB_API_KEY

    try:
        _rate_limit()
        resp = requests.get(_PERSON_SEARCH_URL, headers=_HEADERS, params=params, timeout=10)
        # Handle 429 Rate Limit separately
        if resp.status_code == 429:
            retry_after = int(resp.headers.get("Retry-After", 2))
            logger.warning("Rate limited by TMDb (actor search). Waiting %ds...", retry_after)
            time.sleep(retry_after)
            _rate_limit()
            resp = requests.get(_PERSON_SEARCH_URL, headers=_HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        profile_path = results[0].get("profile_path")
        if not profile_path:
            return None
        _tmdb_breaker.record_success()
        return f"{_IMAGE_BASE}/{size}{profile_path}"
    except (requests.Timeout, requests.ConnectionError, TMDbServerError):
        _tmdb_breaker.record_failure()
        return None
    except (ValueError, KeyError, TypeError, IndexError, requests.HTTPError):
        return None


# Async enrichment


# Async fetch (httpx.AsyncClient + tenacity async)


@retry(
    retry=retry_if_exception_type(
        (
            httpx.TimeoutException,
            httpx.ConnectError,
            TMDbServerError,
        )
    ),
    wait=wait_exponential_jitter(initial=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
async def _fetch_tmdb_async(
    client: httpx.AsyncClient,
    title: str,
    year: int | None = None,
) -> dict[str, Any | None]:
    """Async TMDb search with tenacity retry/backoff.

    Parameters
    ----------
    client : httpx.AsyncClient
        Shared async HTTP client.
    title : str
        Movie title to search for.
    year : int, optional
        Release year for disambiguation.

    Returns
    -------
    dict or None
        The first TMDb result, or None if no match.
    """
    if not _TMDB_API_KEY and not _TMDB_ACCESS_TOKEN:
        return None

    # Check circuit breaker before making API call
    if _tmdb_breaker.is_open():
        logger.warning("TMDb circuit breaker open — skipping async search for '%s'", title)
        return None

    params = _build_params(title, year)

    resp = await client.get(_SEARCH_URL, headers=_HEADERS, params=params, timeout=10)

    # Handle 429 Rate Limit — respect Retry-After
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", 2))
        logger.warning("Rate limited (async). Waiting %ds (Retry-After)...", retry_after)
        await asyncio.sleep(retry_after)
        resp = await client.get(_SEARCH_URL, headers=_HEADERS, params=params, timeout=10)

    # 5xx → raise for tenacity to retry
    if resp.status_code >= 500:
        raise TMDbServerError(f"TMDb returned {resp.status_code}: {resp.text[:200]}")

    resp.raise_for_status()  # 4xx (except 429) → fail fast, no retry

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


async def _enrich_async(
    titles: list[str],
    years: list[int | None],
    cache: dict[str, Any],
    concurrency: int = 8,
) -> list[tuple[int, str, str]]:
    """Run async enrichment with httpx.AsyncClient + bounded concurrency.

    Uses true async I/O via ``httpx.AsyncClient``, NOT a thread pool.
    Concurrency is bounded by an ``asyncio.Semaphore(concurrency)``.

    Parameters
    ----------
    titles : list[str]
        Movie titles.
    years : list[int | None]
        Corresponding years.
    cache : dict
        Shared cache dict.
    concurrency : int, default=8
        Max concurrent TMDb API calls. 8 keeps us safely within TMDb's
        rate limits (40 req/10s) while maximizing throughput.

    Returns
    -------
    list[tuple[int, str, str]]
        Results as ``(idx, plot, date)`` tuples.
    """
    sem = asyncio.Semaphore(concurrency)

    async def _fetch_one(i: int) -> tuple[int, str, str]:
        cache_key = (
            f"{titles[i].strip().lower()}|{years[i]}" if years[i] else titles[i].strip().lower()
        )
        if cache_key in cache:
            data = cache[cache_key]
            has_plot = bool(data and data.get("overview", "").strip())
            has_date = bool(data and data.get("release_date", "").strip())
            return (
                i,
                data["overview"] if data and has_plot else "",
                data["release_date"] if data and has_date else "",
            )

        async with sem:
            try:
                result = await _fetch_tmdb_async(client, titles[i], years[i])
                _tmdb_breaker.record_success()
            except (OSError, ValueError) as exc:
                _tmdb_breaker.record_failure()
                logger.debug("Async TMDb fetch failed for %s: %s", titles[i], exc)
                cache[cache_key] = None
                return (i, "", "")

        if result is None:
            cache[cache_key] = None
            return (i, "", "")

        data: dict[str, Any] = {
            "title": result.get("title", titles[i]),
            "overview": result.get("overview", ""),
            "release_date": result.get("release_date", ""),
            "poster_path": result.get("poster_path"),
            "tmdb_id": result.get("id"),
        }
        cache[cache_key] = data

        has_plot = bool(data.get("overview", "").strip())
        has_date = bool(data.get("release_date", "").strip())
        return (
            i,
            data["overview"] if has_plot else "",
            data["release_date"] if has_date else "",
        )

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        tasks = [_fetch_one(i) for i in range(len(titles))]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    output: list[tuple[int, str, str]] = []
    for i, result in enumerate(results):
        if isinstance(result, tuple):
            output.append(cast(tuple[int, str, str], result))
        else:
            logger.debug("Unexpected error in async enrichment for %s: %s", titles[i], result)
            output.append((i, "", ""))
    return output


def enrich_dataset_async(
    df: Any,
    title_column: str = "title",
    year_column: str | None = None,
    max_movies: int | None = None,
    concurrency: int = 8,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Enrich a movie DataFrame with TMDb data using true async I/O.

    Uses ``httpx.AsyncClient`` (not ThreadPoolExecutor) for genuine
    async HTTP calls. Bounded concurrency via ``asyncio.Semaphore``.

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
    years_list: list[int | None] = []
    for i in range(total):
        if year_column and year_column in result_df.columns:
            try:
                years_list.append(int(float(result_df.iloc[i][year_column])))
            except (ValueError, TypeError):
                years_list.append(None)
        else:
            years_list.append(None)

    logger.info(
        "Enriching %d movies from TMDb (async httpx, concurrency=%d)...", total, concurrency
    )

    # Run the async enrichment — asyncio.run() bridges sync→async
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
    logger.info("TMDb Enrichment Complete (async httpx)")
    logger.info("  Movies attempted:   %d", total)
    logger.info(
        "  Plot coverage:      %.1f%% (%d with plot)",
        plot_coverage,
        sum(1 for p in plots if p.strip()),
    )
    logger.info(
        "  Date coverage:      %.1f%% (%d with date)",
        date_coverage,
        sum(1 for d in dates if d.strip()),
    )
    logger.info(
        "  Overall matches:    %.1f%% (%d/%d)", match_count / total * 100, match_count, total
    )
    logger.info("=" * 60)

    coverage = {
        "plots": [p for p in plots if p.strip()],
        "dates": [d for d in dates if d.strip()],
    }

    return coverage, result_df
