"""Tests for API request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.schemas import (
    PredictBoxOfficeRequest,
    PredictBoxOfficeResponse,
    PredictRatingRequest,
    PredictRatingResponse,
)
from tamasha.config import settings  # noqa: F401  (ensures src layout is importable)

pytestmark = pytest.mark.unit


class TestPredictRatingRequest:
    """Tests for the rating prediction request schema."""

    def test_valid_request(self) -> None:
        req = PredictRatingRequest(
            title="Test Movie",
            genres=["Action", "Drama"],
            cast=["Actor A", "Actor B"],
        )
        assert req.title == "Test Movie"
        assert req.genres == ["Action", "Drama"]
        assert req.budget_inr == 0.0  # default
        assert req.runtime_minutes == 150  # default

    def test_custom_fields(self) -> None:
        req = PredictRatingRequest(
            title="Test",
            genres=["Comedy"],
            cast=["Actor A"],
            director="Dir B",
            budget_inr=500.0,
            runtime_minutes=120,
        )
        assert req.director == "Dir B"
        assert req.budget_inr == 500.0

    def test_title_is_required(self) -> None:
        with pytest.raises(ValidationError):
            PredictRatingRequest(genres=[], cast=[])


class TestPredictRatingResponse:
    """Tests for the rating prediction response schema."""

    def test_valid_response(self) -> None:
        resp = PredictRatingResponse(
            title="Test Movie",
            predicted_rating=7.5,
            model_name="random_forest",
            model_mae=0.42,
        )
        assert resp.predicted_rating == 7.5
        assert resp.model_name == "random_forest"


class TestPredictBoxOfficeRequest:
    """Tests for the box-office request schema."""

    def test_valid_request(self) -> None:
        req = PredictBoxOfficeRequest(
            title="Test",
            genres=["Action"],
            cast=["Actor A"],
            release_window="Holiday",
        )
        assert req.release_window == "Holiday"

    def test_default_release_window(self) -> None:
        req = PredictBoxOfficeRequest(title="T", genres=[], cast=[])
        assert req.release_window == "Normal"


class TestPredictBoxOfficeResponse:
    """Tests for the box-office response schema."""

    def test_valid_response(self) -> None:
        resp = PredictBoxOfficeResponse(
            title="Test Movie",
            predicted_boxoffice_cr=120.5,
            model_name="gradient_boosting",
            model_mae=18.3,
        )
        assert resp.predicted_boxoffice_cr == 120.5
        assert resp.scenarios is None

    def test_response_with_scenarios(self) -> None:
        resp = PredictBoxOfficeResponse(
            title="Test",
            predicted_boxoffice_cr=80.0,
            model_name="m",
            model_mae=10.0,
            scenarios={"Holiday": 95.0, "Normal": 80.0},
        )
        assert resp.scenarios is not None
        assert len(resp.scenarios) == 2
