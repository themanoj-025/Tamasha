"""Network routes for actor information.

Uses FastAPI ``Depends()`` to receive a ``PredictionService`` instance.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from api.schemas import ActorInfoResponse
from tamasha.predict import PredictionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["network"])


def _get_svc() -> PredictionService:
    from api.main import get_prediction_service
    return get_prediction_service()


@router.get("/actor/{name}", response_model=ActorInfoResponse)
async def get_actor_info_endpoint(
    name: str,
    svc: PredictionService = Depends(_get_svc),
) -> ActorInfoResponse:
    """Get Bankability Score and top chemistry pairs for an actor."""
    try:
        info = svc.get_actor_info(name)
        if not info["found"]:
            raise HTTPException(
                status_code=404,
                detail=f"'{name}' not found in Bankability dataset (1,010 individuals scored).",
            )
        return ActorInfoResponse(
            name=info["name"],
            bankability_score=info["bankability_score"],
            film_count=info["film_count"],
            top_chemistry_pairs=info["top_chemistry_pairs"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to get actor info for %s", name)
        raise HTTPException(status_code=500, detail=str(exc))
