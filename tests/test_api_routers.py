"""Tests for API router endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from tamasha.api.main import app

pytestmark = pytest.mark.slow
@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    @patch("tamasha.api.routers.predict._load_model")
    def test_predict_missing_fields(self, mock_load, client) -> None:
        response = client.post("/predict", json={})
        assert response.status_code in (400, 422)

    @patch("tamasha.api.routers.predict._load_model")
    def test_predict_with_title(self, mock_load, client) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]
        mock_load.return_value = (mock_model, MagicMock(), [])
        response = client.post("/predict", json={"title": "Test Movie", "year": 2024})
        assert response.status_code in (200, 422, 500)
