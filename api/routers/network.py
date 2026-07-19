"""Network routes for actor information."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.schemas import ActorInfoResponse
from tamasha.predict import get_actor_info

logger = logging.getLogger(__name__)
router = APIRouter(prefix="", tags=["network"])


@router.get("/actor/{name}", response_model=ActorInfoResponse)
async def get_actor_info_endpoint(name: str) -> ActorInfoResponse:
    """Get Bankability Score and top chemistry pairs for an actor."""
    try:
        info = get_actor_info(name)
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
