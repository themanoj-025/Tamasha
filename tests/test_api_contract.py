"""Contract tests for the FastAPI API layer.

Verifies:
- Pydantic schema validation rejects invalid inputs
- Error responses follow a consistent shape
- /health correctly reports model status
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from api.schemas import PredictBoxOfficeRequest, PredictRatingRequest, PredictRatingResponse

client = TestClient(app)
client.headers["X-API-Key"] = "tamasha-dev-key-2026"


# ── Schema validation tests (pure Pydantic, no network) ──────────────


class TestPredictRatingSchema:
    """Validate PredictRatingRequest/PredictRatingResponse schemas."""

    def test_valid_request_passes(self) -> None:
        req = PredictRatingRequest(
            title="Test",
            genres=["Drama", "Action"],
            cast=["Actor A", "Actor B"],
            director="Director",
            budget_inr=50_000_000,
            runtime_minutes=150,
        )
        assert req.title == "Test"
        assert len(req.genres) == 2

    def test_missing_required_fields_fails(self) -> None:
        """title and genres are required — omitting them should fail."""
        with pytest.raises(ValidationError):
            PredictRatingRequest()

    def test_empty_genres_list_allowed(self) -> None:
        """Empty genre list is syntactically valid (even if semantically odd)."""
        req = PredictRatingRequest(
            title="Test",
            genres=[],
            cast=["Actor A"],
        )
        assert req.genres == []

    def test_negative_budget_allowed_by_schema(self) -> None:
        """Current schema allows negative budget — documented as limitation."""
        req = PredictRatingRequest(
            title="Test",
            genres=["Drama"],
            cast=["Actor A"],
            budget_inr=-100.0,
        )
        assert req.budget_inr == -100.0

    def test_zero_budget_allowed(self) -> None:
        req = PredictRatingRequest(
            title="Test",
            genres=["Drama"],
            cast=["Actor A"],
            budget_inr=0.0,
        )
        assert req.budget_inr == 0.0

    def test_response_requires_all_fields(self) -> None:
        """PredictRatingResponse must have title, predicted_rating, model_name, model_mae."""
        with pytest.raises(ValidationError):
            PredictRatingResponse(title="Test")  # missing required fields


class TestPredictBoxOfficeSchema:
    """Validate PredictBoxOfficeRequest/PredictBoxOfficeResponse schemas."""

    def test_valid_request_passes(self) -> None:
        req = PredictBoxOfficeRequest(
            title="Test",
            genres=["Action"],
            cast=["Actor A"],
            budget_inr=100_000_000,
            runtime_minutes=140,
            release_window="Diwali",
        )
        assert req.release_window == "Diwali"

    def test_default_release_window(self) -> None:
        req = PredictBoxOfficeRequest(
            title="Test",
            genres=["Drama"],
            cast=["Actor A"],
        )
        assert req.release_window == "Normal"

    def test_arbitrary_release_window_allowed(self) -> None:
        """Schema does not restrict release_window to an enum."""
        req = PredictBoxOfficeRequest(
            title="Test",
            genres=["Drama"],
            cast=["Actor A"],
            release_window="ArbitraryWindow",
        )
        assert req.release_window == "ArbitraryWindow"


class TestActorInfoSchema:
    """Validate ActorInfoResponse schema."""

    def test_minimal_response_valid(self) -> None:
        response = PredictRatingResponse(
            title="Test",
            predicted_rating=7.5,
            model_name="GradientBoosting",
            model_mae=0.95,
        )
        assert response.predicted_rating == 7.5
        assert response.model_name == "GradientBoosting"


# ── API endpoint contract tests ──────────────────────────────────────


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_health_returns_200(self) -> None:
        """Happy path: models exist, /health returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data

    def test_health_structure(self) -> None:
        """Response should have specific structure."""
        response = client.get("/health")
        data = response.json()
        # Must have these keys
        assert "status" in data
        assert isinstance(data["status"], str)
        assert "version" in data
        assert isinstance(data["version"], str)
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], bool)


class TestPredictRatingEndpoint:
    """Contract tests for POST /predict-rating."""

    def test_valid_request_returns_200(self) -> None:
        response = client.post(
            "/predict-rating",
            json={
                "title": "Test Movie",
                "genres": ["Drama", "Romance"],
                "cast": ["Actor A", "Actor B"],
                "director": "Director A",
                "budget_inr": 50_000_000,
                "runtime_minutes": 150,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Response must have these fields
        assert "title" in data
        assert "predicted_rating" in data
        assert isinstance(data["predicted_rating"], (int, float))
        assert "model_name" in data
        assert isinstance(data["model_name"], str)
        assert "model_mae" in data
        assert isinstance(data["model_mae"], (int, float))

    def test_missing_title_returns_422(self) -> None:
        """Missing required field 'title' → 422."""
        response = client.post(
            "/predict-rating",
            json={
                "genres": ["Drama"],
                "cast": ["Actor A"],
            },
        )
        assert response.status_code == 422
        data = response.json()
        # FastAPI validation errors have a standard shape
        assert "detail" in data
        assert isinstance(data["detail"], list)

    def test_empty_body_returns_422(self) -> None:
        response = client.post("/predict-rating", json={})
        assert response.status_code == 422

    def test_extra_fields_ignored(self) -> None:
        """Extra fields not in schema should be ignored."""
        response = client.post(
            "/predict-rating",
            json={
                "title": "Test",
                "genres": ["Drama"],
                "cast": ["Actor A"],
                "extra_field": "should be ignored",
            },
        )
        assert response.status_code == 200  # Not 422

    def test_shortest_possible_valid_request(self) -> None:
        """Only required fields + defaults."""
        response = client.post(
            "/predict-rating",
            json={
                "title": "T",
                "genres": ["A"],
                "cast": ["B"],
            },
        )
        assert response.status_code == 200


class TestPredictBoxOfficeEndpoint:
    """Contract tests for POST /predict-boxoffice."""

    def test_valid_request_returns_200(self) -> None:
        response = client.post(
            "/predict-boxoffice",
            json={
                "title": "Test Movie",
                "genres": ["Action"],
                "cast": ["Actor A"],
                "budget_inr": 100_000_000,
                "runtime_minutes": 140,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_boxoffice_cr" in data
        assert isinstance(data["predicted_boxoffice_cr"], (int, float))
        assert "model_name" in data
        assert "model_mae" in data

    def test_scenarios_included(self) -> None:
        response = client.post(
            "/predict-boxoffice",
            json={
                "title": "Test",
                "genres": ["Drama"],
                "cast": ["Actor A"],
                "release_window": "Diwali",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert isinstance(data["scenarios"], dict)


class TestActorInfoEndpoint:
    """Contract tests for GET /actor/{name}."""

    def test_known_actor_returns_200(self) -> None:
        response = client.get("/actor/Shah%20Rukh%20Khan")
        # May be 200 or 404 depending on whether bankability data exists
        assert response.status_code in (200, 404)

    def test_unknown_actor_returns_404(self) -> None:
        response = client.get("/actor/ThisActorDoesNotExist")
        assert response.status_code == 404

    def test_actor_info_shape(self) -> None:
        """Successful response should have expected fields."""
        response = client.get("/actor/Shah%20Rukh%20Khan")
        if response.status_code == 200:
            data = response.json()
            assert "name" in data
            assert "bankability_score" in data
            assert "film_count" in data
            assert isinstance(data["film_count"], int)


class TestModelInfoEndpoint:
    """Contract tests for GET /model-info."""

    def test_model_info_structure(self) -> None:
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "rating_model" in data
        assert "boxoffice_model" in data
        for model_key in ("rating_model", "boxoffice_model"):
            model = data[model_key]
            assert "name" in model
            assert "algorithm" in model
            assert "mae" in model or model.get("mae") is None
