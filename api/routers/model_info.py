"""Model info route for deployed model metadata.

Uses FastAPI ``Depends()`` to receive a ``PredictionService`` instance.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from api.main import get_prediction_service
from api.schemas import ModelInfoResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["model-info"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info_endpoint(
    svc=Depends(get_prediction_service),
) -> ModelInfoResponse:
    """Get metadata about currently deployed models."""
    info = svc.get_model_info()
    return ModelInfoResponse(
        rating_model=info["rating_model"],
        boxoffice_model=info["boxoffice_model"],
    )
