"""FastAPI application entry point for Tamasha API.

Uses a ``lifespan`` context manager to build the ``PredictionService``
once at startup and inject it into route handlers via ``Depends()``.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from tamasha.predict import PredictionService

# ── Structured logging via structlog ──────────────────────────────────

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if __debug__
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger()  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build PredictionService on startup, clean up on shutdown."""
    logger.info("Tamaha API starting up...")
    svc = PredictionService()
    svc.load()
    app.state.prediction_service = svc
    logger.info("prediction_service_loaded", healthy=svc.healthy)
    yield
    logger.info("Tamasha API shutting down...")


app = FastAPI(
    title="Tamasha API",
    description="Bollywood Movie Intelligence Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

# ── Request‑ID middleware ─────────────────────────────────────────────

@app.middleware("http")
async def add_request_id(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


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
    """FastAPI dependency — yields the singleton from ``app.state``.

    Falls back to creating one on the fly (for tests / scripts that
    create ``TestClient`` without triggering the lifespan).
    """
    svc: PredictionService | None = getattr(app.state, "prediction_service", None)
    if svc is None:
        svc = PredictionService()
        svc.load()
        app.state.prediction_service = svc
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
