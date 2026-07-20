"""FastAPI application entry point for Tamasha API.

Uses a ``lifespan`` context manager to build the ``PredictionService``
once at startup and inject it into route handlers via ``Depends()``.
Exposes rate-limited, auth-guarded endpoints with structured logging.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from tamasha.config import settings
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


# ── App factory ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build PredictionService on startup, clean up on shutdown."""
    logger.info("Tamasha API starting up...")
    svc = PredictionService()
    svc.load()
    app.state.prediction_service = svc
    logger.info("prediction_service_loaded", healthy=svc.healthy)
    yield
    logger.info("Tamasha API shutting down...")


# ── Rate limiter ─────────────────────────────────────────────────────

def _rate_limit_key(request) -> str:
    """Use API key as the rate-limit identifier if present; fall back to IP."""
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        return api_key
    return get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key, default_limits=[settings.RATE_LIMIT])

app = FastAPI(
    title="Tamasha API",
    description="Bollywood Movie Intelligence Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── API key authentication middleware ─────────────────────────────────
# Exempts health-check and OpenAPI doc endpoints.

_AUTH_EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


@app.middleware("http")
async def verify_api_key_middleware(request, call_next):
    if request.url.path in _AUTH_EXEMPT_PATHS:
        return await call_next(request)

    api_key = request.headers.get("X-API-Key", "")
    if api_key != settings.API_KEY:
        request_id = str(uuid.uuid4())[:8]
        logger.warning("auth_rejected", path=request.url.path, request_id=request_id)
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid or missing API key. Provide via X-API-Key header."},
            headers={"X-Request-ID": request_id},
        )
    return await call_next(request)


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

_allowed_origins = [
    o.strip() for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("cors_configured", allowed_origins=_allowed_origins)


# ── Apply rate‑limiting middleware (after CORS so headers aren't stripped) ─

app.add_middleware(SlowAPIMiddleware)


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
