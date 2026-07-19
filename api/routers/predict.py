"""Prediction routes for rating and box office."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import (
    PredictBoxOfficeRequest,
    PredictBoxOfficeResponse,
    PredictRatingRequest,
    PredictRatingResponse,
)
from tamasha.predict import predict_rating, predict_boxoffice

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["predict"])


@router.post("/predict-rating", response_model=PredictRatingResponse)
async def predict_rating_endpoint(request: PredictRatingRequest) -> PredictRatingResponse:
    """Predict movie rating from cast, genre, and budget features."""
    try:
        result = predict_rating(
            genres=request.genres,
            cast=request.cast,
            director=request.director,
            budget_inr=request.budget_inr,
            runtime_minutes=request.runtime_minutes,
        )
        if result["predicted_rating"] is None:
            raise HTTPException(status_code=503, detail="Rating model not available. Run: make train")
        return PredictRatingResponse(
            title=request.title,
            predicted_rating=result["predicted_rating"],
            model_name=result["model_name"],
            model_mae=result["model_mae"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Rating prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict-boxoffice", response_model=PredictBoxOfficeResponse)
async def predict_boxoffice_endpoint(
    request: PredictBoxOfficeRequest,
) -> PredictBoxOfficeResponse:
    """Predict movie box office using the Bankability-enhanced model."""
    try:
        result = predict_boxoffice(
            genres=request.genres,
            cast=request.cast,
            director=request.director,
            budget_inr=request.budget_inr,
            runtime_minutes=request.runtime_minutes,
            release_window=request.release_window,
        )
        if result["predicted_boxoffice_cr"] is None:
            raise HTTPException(status_code=503, detail="Box office model not available. Run: make train")
        return PredictBoxOfficeResponse(
            title=request.title,
            predicted_boxoffice_cr=result["predicted_boxoffice_cr"],
            model_name=result["model_name"],
            model_mae=result["model_mae"],
            scenarios=result.get("scenarios"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Box office prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
