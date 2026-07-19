"""Model info route for deployed model metadata."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from api.schemas import ModelInfoResponse
from tamasha.predict import get_model_info

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["model-info"])


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info_endpoint() -> ModelInfoResponse:
    """Get metadata about currently deployed models."""
    info = get_model_info()
    return ModelInfoResponse(
        rating_model=info["rating_model"],
        boxoffice_model=info["boxoffice_model"],
    )
