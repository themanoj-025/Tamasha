"""Smoke tests for the FastAPI API layer."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealth:
    """Tests for the health endpoint."""

    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestPredictRating:
    """Tests for the predict-rating endpoint."""

    def test_predict_rating_valid_input(self):
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
        assert "predicted_rating" in data
        assert "model_name" in data

    def test_predict_rating_missing_fields(self):
        response = client.post("/predict-rating", json={})
        assert response.status_code == 422


class TestPredictBoxOffice:
    """Tests for the predict-boxoffice endpoint."""

    def test_predict_boxoffice_valid_input(self):
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

    def test_predict_boxoffice_with_scenarios(self):
        response = client.post(
            "/predict-boxoffice",
            json={
                "title": "Test Movie",
                "genres": ["Drama"],
                "cast": ["Actor A", "Actor B"],
                "release_window": "Diwali",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data


class TestActorInfo:
    """Tests for the actor info endpoint."""

    def test_get_actor_info(self):
        response = client.get("/actor/Shah Rukh Khan")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "bankability_score" in data

    def test_get_unknown_actor(self):
        # Should return 404 if actor not in Bankability dataset
        response = client.get("/actor/Unknown%20Actor")
        assert response.status_code == 404


class TestModelInfo:
    """Tests for the model-info endpoint."""

    def test_get_model_info(self):
        response = client.get("/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "rating_model" in data
        assert "boxoffice_model" in data
