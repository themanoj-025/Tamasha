"""FastAPI application entry point for Tamasha API.

Uses a ``lifespan`` context manager to build the ``PredictionService``
once at startup and inject it into route handlers via ``Depends()``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from tamasha.predict import PredictionService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build PredictionService on startup, clean up on shutdown."""
    logger.info("Tamasha API starting up...")
    svc = PredictionService()
    svc.load()
    app.state.prediction_service = svc
    logger.info("PredictionService loaded. Healthy: %s", svc.healthy)
    yield
    logger.info("Tamasha API shutting down...")
    # Nothing explicit to clean up — model files remain on disk


app = FastAPI(
    title="Tamasha API",
    description="Bollywood Movie Intelligence Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

# ── CORS — restricted in production ───────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependency injection helper ──────────────────────────────────────

def get_prediction_service() -> PredictionService:
    """FastAPI dependency — yields the singleton from ``app.state``."""
    svc: PredictionService = app.state.prediction_service
    return svc


# ── Health ───────────────────────────────────────────────────────────

@app.get("/health")
async def health(
    svc: PredictionService = Depends(get_prediction_service),
) -> dict:
    """Health check — reflects model-availability status."""
    healthy = svc.healthy
    return {
        "status": "ok" if healthy else "degraded",
        "version": "0.1.0",
        "models_loaded": healthy,
    }


# ── Routers ──────────────────────────────────────────────────────────

from api.routers import predict, network, model_info  # noqa: E402

app.include_router(predict.router)
app.include_router(network.router)
app.include_router(model_info.router)
