"""Pydantic schemas for Tamasha API request/response models."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Predict Rating ────────────────────────────────────────────────────

class PredictRatingRequest(BaseModel):
    """Request body for rating prediction."""

    title: str = Field(..., description="Movie title")
    genres: list[str] = Field(..., description="List of genres")
    cast: list[str] = Field(..., description="List of cast members")
    director: str = Field("Unknown", description="Director name")
    budget_inr: float = Field(0.0, description="Budget in rupees")
    runtime_minutes: int = Field(150, description="Runtime in minutes")


class PredictRatingResponse(BaseModel):
    """Response body for rating prediction."""

    title: str
    predicted_rating: float = Field(..., description="Predicted IMDB rating (0-10)")
    model_name: str = Field(..., description="Winning model name")
    model_mae: float = Field(..., description="Model's MAE on validation")


# ── Predict Box Office ────────────────────────────────────────────────

class PredictBoxOfficeRequest(BaseModel):
    """Request body for box-office prediction."""

    title: str = Field(..., description="Movie title")
    genres: list[str] = Field(..., description="List of genres")
    cast: list[str] = Field(..., description="List of cast members")
    director: str = Field("Unknown", description="Director name")
    budget_inr: float = Field(0.0, description="Budget in rupees")
    runtime_minutes: int = Field(150, description="Runtime in minutes")
    release_window: str = Field("Normal", description="Release window scenario")


class PredictBoxOfficeResponse(BaseModel):
    """Response body for box-office prediction."""

    title: str
    predicted_boxoffice_cr: float = Field(..., description="Predicted box office in ₹ Crore")
    model_name: str = Field(..., description="Winning model name")
    model_mae: float = Field(..., description="Model's MAE on validation")
    scenarios: Optional[dict[str, Any]] = Field(None, description="Scenario comparison results")


# ── Actor Info ────────────────────────────────────────────────────────

class ActorInfoResponse(BaseModel):
    """Response body for actor information."""

    name: str = Field(..., description="Actor name")
    bankability_score: float = Field(..., description="Bankability Score (0-1)")
    film_count: int = Field(..., description="Number of films in dataset")
    top_chemistry_pairs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Top chemistry pairings",
    )


# ── Model Info ────────────────────────────────────────────────────────

class ModelInfoResponse(BaseModel):
    """Response body for model information."""

    rating_model: dict[str, Any] = Field(..., description="Rating model details")
    boxoffice_model: dict[str, Any] = Field(..., description="Box office model details")
