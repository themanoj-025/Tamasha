"""Tests for API request/response schemas."""

import pytest

from tamasha.api.schemas import PredictionRequest, PredictionResponse


class TestPredictionRequest:
    """Tests for PredictionRequest schema."""

    def test_valid_request(self) -> None:
        req = PredictionRequest(title="Test Movie", year=2024)
        assert req.title == "Test Movie"
        assert req.year == 2024

    def test_optional_fields(self) -> None:
        req = PredictionRequest(title="Test")
        assert req.year is None

    def test_with_cast(self) -> None:
        req = PredictionRequest(title="Test", cast=["Actor A", "Actor B"])
        assert len(req.cast) == 2


class TestPredictionResponse:
    """Tests for PredictionResponse schema."""

    def test_valid_response(self) -> None:
        resp = PredictionResponse(
            title="Test Movie",
            bankability_score=7.5,
            confidence=0.85,
            tier="A",
        )
        assert resp.bankability_score == 7.5
        assert resp.tier == "A"

    def test_response_with_details(self) -> None:
        resp = PredictionResponse(
            title="Test",
            bankability_score=6.0,
            confidence=0.7,
            tier="B",
            similar_films=["Film 1", "Film 2"],
        )
        assert len(resp.similar_films) == 2
