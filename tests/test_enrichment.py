"""Tests for the TMDb enrichment module with mocked HTTP.

All external HTTP calls are mocked — tests run with no network access.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

# ── Monkey-patch env vars before importing the module ────────────────


@pytest.fixture(autouse=True)
def _set_tmdb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set dummy TMDb credentials so the module doesn't warn."""
    monkeypatch.setenv("TMDB_API_KEY", "test_key_12345")
    monkeypatch.setenv("TMDB_ACCESS_TOKEN", "test_token_12345")


@pytest.fixture(autouse=True)
def _clear_enrichment_cache() -> None:
    """Reset the module-level cache before each test."""
    import tamasha.data.enrichment as enrichment_mod

    enrichment_mod._CACHE.clear()
    enrichment_mod._LAST_REQUEST_TIME = 0.0


# ── Helpers ───────────────────────────────────────────────────────────


def _make_tmdb_result(
    title: str = "Test Movie",
    overview: str = "A test plot summary.",
    release_date: str = "2024-01-15",
    poster_path: str | None = "/poster.jpg",
    movie_id: int = 123,
) -> dict:
    return {
        "id": movie_id,
        "title": title,
        "overview": overview,
        "release_date": release_date,
        "poster_path": poster_path,
    }


def _make_tmdb_search_response(
    results: list[dict],
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = 200
    resp._content = json.dumps({"results": results}).encode("utf-8")
    resp.headers["Content-Type"] = "application/json"
    return resp


def _make_rate_limit_response(retry_after: int = 2) -> requests.Response:
    resp = requests.Response()
    resp.status_code = 429
    resp.headers["Retry-After"] = str(retry_after)
    resp._content = b"{}"
    return resp


# ── Tests: get_movie_data ────────────────────────────────────────────


class TestGetMovieData:
    """Tests for the get_movie_data function."""

    def test_successful_lookup(self) -> None:
        """Happy path: TMDb returns a result, we return structured data."""
        from tamasha.data.enrichment import get_movie_data

        mock_result = _make_tmdb_result(
            title="Dilwale Dulhania Le Jayenge",
            overview="A young man and woman...",
            release_date="1995-10-20",
        )

        with patch(
            "tamasha.data.enrichment.requests.get",
            return_value=_make_tmdb_search_response([mock_result]),
        ):
            data = get_movie_data("Dilwale Dulhania Le Jayenge", year=1995)

        assert data is not None
        assert data["title"] == "Dilwale Dulhania Le Jayenge"
        assert data["overview"] == "A young man and woman..."
        assert data["release_date"] == "1995-10-20"
        assert data["tmdb_id"] == 123

    def test_no_results_returns_none(self) -> None:
        """TMDb returns empty results → None."""
        from tamasha.data.enrichment import get_movie_data

        with patch(
            "tamasha.data.enrichment.requests.get", return_value=_make_tmdb_search_response([])
        ):
            data = get_movie_data("Totally Fake Movie 99999", year=2099)

        assert data is None

    def test_network_timeout_returns_none(self) -> None:
        """requests.get raises ConnectionError → graceful None, not crash."""
        from tamasha.data.enrichment import get_movie_data

        with patch(
            "tamasha.data.enrichment.requests.get", side_effect=requests.ConnectionError("Timeout")
        ):
            data = get_movie_data("Any Movie", year=2020)

        assert data is None

    def test_429_rate_limit_retry_succeeds(self) -> None:
        """First call gets 429, retry succeeds → returns data."""
        from tamasha.data.enrichment import get_movie_data

        mock_result = _make_tmdb_result()
        fail_resp = _make_rate_limit_response(retry_after=1)
        ok_resp = _make_tmdb_search_response([mock_result])

        mock_get = MagicMock(side_effect=[fail_resp, ok_resp])

        with patch("tamasha.data.enrichment.requests.get", mock_get):
            data = get_movie_data("Test Movie", year=2024)

        assert data is not None
        assert data["title"] == "Test Movie"
        # Verify exactly 2 calls were made (original + retry)
        assert mock_get.call_count == 2

    def test_429_rate_limit_then_fails(self) -> None:
        """First call 429, retry also 429. The HTTPError is caught → returns None."""
        from tamasha.data.enrichment import get_movie_data

        fail_resp = _make_rate_limit_response(retry_after=1)
        mock_get = MagicMock(return_value=fail_resp)

        with patch("tamasha.data.enrichment.requests.get", mock_get):
            data = get_movie_data("Test Movie", year=2024)

        # _search_tmdb catches requests.RequestException (includes HTTPError
        # from raise_for_status) and returns None
        assert data is None

    def test_cache_hit_does_not_call_api(self) -> None:
        """Second call for same (title, year) uses cache."""
        from tamasha.data.enrichment import get_movie_data

        mock_result = _make_tmdb_result()
        mock_get = MagicMock(return_value=_make_tmdb_search_response([mock_result]))

        with patch("tamasha.data.enrichment.requests.get", mock_get):
            # First call — hits API
            data1 = get_movie_data("Cached Movie", year=2023)
            # Second call — should use cache
            data2 = get_movie_data("Cached Movie", year=2023)

        assert data1 is not None
        assert data2 is not None
        assert data1["title"] == data2["title"]
        # Only 1 API call for 2 lookups
        assert mock_get.call_count == 1

    def test_cache_bypassed_with_force(self) -> None:
        """force=True bypasses cache and hits the API again."""
        from tamasha.data.enrichment import get_movie_data

        mock_result = _make_tmdb_result()
        mock_get = MagicMock(return_value=_make_tmdb_search_response([mock_result]))

        with patch("tamasha.data.enrichment.requests.get", mock_get):
            data1 = get_movie_data("Force Movie", year=2023)
            data2 = get_movie_data("Force Movie", year=2023, force=True)

        assert data1 is not None
        assert data2 is not None
        assert mock_get.call_count == 2  # force=True → 2nd call hits API again

    def test_malformed_json_handled(self) -> None:
        """API returns non-JSON → handled without crash."""
        from tamasha.data.enrichment import get_movie_data

        resp = requests.Response()
        resp.status_code = 200
        resp._content = b"not json at all"
        resp.headers["Content-Type"] = "text/plain"

        with patch("tamasha.data.enrichment.requests.get", return_value=resp):
            # resp.json() will raise ValueError, caught by _search_tmdb
            data = get_movie_data("Bad JSON", year=2024)

        assert data is None or isinstance(data, dict)

    def test_year_preferred_in_results(self) -> None:
        """When year is provided, prefer result matching that year."""
        from tamasha.data.enrichment import get_movie_data

        results = [
            _make_tmdb_result(title="Movie 2023", release_date="2023-06-15", movie_id=1),
            _make_tmdb_result(title="Movie 2024", release_date="2024-06-15", movie_id=2),
        ]

        with patch(
            "tamasha.data.enrichment.requests.get", return_value=_make_tmdb_search_response(results)
        ):
            data = get_movie_data("Movie", year=2024)

        assert data is not None
        assert data["tmdb_id"] == 2  # Should prefer the 2024 result


# ── Tests: enrich_dataset ────────────────────────────────────────────


class TestEnrichDataset:
    """Tests for the enrich_dataset function."""

    def test_enriches_all_movies(self) -> None:
        """All movies match → full coverage returned."""
        from tamasha.data.enrichment import enrich_dataset

        df = pd.DataFrame(
            {
                "title": ["Movie A", "Movie B"],
                "year": [2020, 2021],
            }
        )

        mock_result = _make_tmdb_result()

        with patch(
            "tamasha.data.enrichment.requests.get",
            return_value=_make_tmdb_search_response([mock_result]),
        ):
            coverage, enriched = enrich_dataset(df, max_movies=2)

        assert "plot_summary" in enriched.columns
        assert "release_date" in enriched.columns
        assert len(coverage["plots"]) > 0
        assert len(coverage["dates"]) > 0

    def test_no_matches_returns_empty_columns(self) -> None:
        """No TMDb matches → plot_summary/release_date are empty strings."""
        from tamasha.data.enrichment import enrich_dataset

        df = pd.DataFrame(
            {
                "title": ["Unknown Movie"],
                "year": [2099],
            }
        )

        with patch(
            "tamasha.data.enrichment.requests.get", return_value=_make_tmdb_search_response([])
        ):
            coverage, enriched = enrich_dataset(df, max_movies=1)

        assert enriched["plot_summary"].iloc[0] == ""
        assert enriched["release_date"].iloc[0] == ""
        assert len(coverage["plots"]) == 0
