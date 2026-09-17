"""Shared FastAPI dependencies.

Lives outside ``api.main`` so router modules can import it without a
circular import (routers used to import the getter from ``api.main``,
which itself imports the routers).
"""

from __future__ import annotations

from fastapi import Request

from tamasha.predict import PredictionService


def get_prediction_service(request: Request) -> PredictionService:
    """FastAPI dependency — yields the singleton from ``app.state``.

    Falls back to creating one on the fly (for tests / scripts that
    create ``TestClient`` without triggering the lifespan).
    """
    svc: PredictionService | None = getattr(request.app.state, "prediction_service", None)
    if svc is None:
        svc = PredictionService()
        svc.load()
        request.app.state.prediction_service = svc
    return svc
