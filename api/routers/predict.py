"""Prediction routes for rating and box office.

Uses FastAPI ``Depends()`` to receive a ``PredictionService`` instance
built during application startup.  Responses are cached via diskcache
with a model-version-aware key so redeploying a model auto-invalidates
stale entries.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.main import get_prediction_service
from api.schemas import (
    PredictBoxOfficeRequest,
    PredictBoxOfficeResponse,
    PredictRatingRequest,
    PredictRatingResponse,
)
from tamasha.cache import get_cached_prediction, set_cached_prediction

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["predict"])


def _model_version_key(svc) -> str:
    """Extract a model-version string for cache invalidation."""
    info = svc.get_model_info()
    return f"{info['rating_model'].get('name', '')}:{info['boxoffice_model'].get('name', '')}"


@router.post("/predict-rating", response_model=PredictRatingResponse)
async def predict_rating_endpoint(
    request: PredictRatingRequest,
    svc=Depends(get_prediction_service),
) -> PredictRatingResponse:
    """Predict movie rating from cast, genre, and budget features."""
    try:
        # Check cache
        cache_payload = request.model_dump()
        mv = _model_version_key(svc)
        cached = get_cached_prediction(cache_payload, mv)
        if cached is not None:
            return PredictRatingResponse(**cached)

        result = svc.predict_rating(
            genres=request.genres,
            cast=request.cast,
            director=request.director,
            budget_inr=request.budget_inr,
            runtime_minutes=request.runtime_minutes,
        )
        if result["predicted_rating"] is None:
            raise HTTPException(
                status_code=503, detail="Rating model not available. Run: make train"
            )

        response = PredictRatingResponse(
            title=request.title,
            predicted_rating=result["predicted_rating"],
            model_name=result["model_name"],
            model_mae=result["model_mae"],
        )
        # Cache the response dict
        set_cached_prediction(cache_payload, response.model_dump(), mv)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Rating prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict-boxoffice", response_model=PredictBoxOfficeResponse)
async def predict_boxoffice_endpoint(
    request: PredictBoxOfficeRequest,
    svc=Depends(get_prediction_service),
) -> PredictBoxOfficeResponse:
    """Predict movie box office using the Bankability-enhanced model."""
    try:
        # Check cache
        cache_payload = request.model_dump()
        mv = _model_version_key(svc)
        cached = get_cached_prediction(cache_payload, mv)
        if cached is not None:
            return PredictBoxOfficeResponse(**cached)

        result = svc.predict_boxoffice(
            genres=request.genres,
            cast=request.cast,
            director=request.director,
            budget_inr=request.budget_inr,
            runtime_minutes=request.runtime_minutes,
            release_window=request.release_window,
        )
        if result["predicted_boxoffice_cr"] is None:
            raise HTTPException(
                status_code=503, detail="Box office model not available. Run: make train"
            )

        response = PredictBoxOfficeResponse(
            title=request.title,
            predicted_boxoffice_cr=result["predicted_boxoffice_cr"],
            model_name=result["model_name"],
            model_mae=result["model_mae"],
            scenarios=result.get("scenarios"),
        )
        # Cache the response dict
        set_cached_prediction(cache_payload, response.model_dump(), mv)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Box office prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
