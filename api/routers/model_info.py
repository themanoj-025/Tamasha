"""Model info route for deployed model metadata.

Uses FastAPI ``Depends()`` to receive a ``PredictionService`` instance.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api.schemas import ModelInfoResponse
from tamasha.predict import PredictionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["model-info"])


def _get_svc() -> PredictionService:
    from api.main import get_prediction_service
    return get_prediction_service()


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info_endpoint(
    svc: PredictionService = Depends(_get_svc),
) -> ModelInfoResponse:
    """Get metadata about currently deployed models."""
    info = svc.get_model_info()
    return ModelInfoResponse(
        rating_model=info["rating_model"],
        boxoffice_model=info["boxoffice_model"],
    )
