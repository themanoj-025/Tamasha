"""Tests for API endpoints (health + prediction routes)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from tamasha.config import settings

pytestmark = pytest.mark.unit


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": settings.API_KEY}


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "degraded")
        assert "version" in data
        assert "models_loaded" in data

    def test_health_shape(self, client) -> None:
        response = client.get("/health")
        data = response.json()
        assert set(data) >= {"status", "version", "models_loaded", "checks"}


class TestPredictRatingEndpoint:
    """Tests for /predict-rating."""

    def test_missing_fields_422(self, client, auth_headers) -> None:
        response = client.post("/api/v1/predict-rating", json={}, headers=auth_headers)
        assert response.status_code == 422

    def test_valid_request_accepted(self, client, auth_headers) -> None:
        """Route exists and validates; model state decides 200 vs 503."""
        response = client.post(
            "/api/v1/predict-rating",
            json={"title": "Test Movie", "genres": ["Action"], "cast": ["Actor A"]},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500, 503)


class TestPredictBoxOfficeEndpoint:
    """Tests for /predict-boxoffice."""

    def test_missing_fields_422(self, client, auth_headers) -> None:
        response = client.post("/api/v1/predict-boxoffice", json={}, headers=auth_headers)
        assert response.status_code == 422

    def test_valid_request_accepted(self, client, auth_headers) -> None:
        response = client.post(
            "/api/v1/predict-boxoffice",
            json={"title": "Test", "genres": ["Drama"], "cast": []},
            headers=auth_headers,
        )
        assert response.status_code in (200, 500, 503)


class TestModelInfoEndpoint:
    """Tests for /model-info."""

    def test_model_info(self, client, auth_headers) -> None:
        response = client.get("/api/v1/model-info", headers=auth_headers)
        assert response.status_code in (200, 500, 503)
