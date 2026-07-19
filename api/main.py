"""FastAPI application entry point for Tamasha API.

Endpoints:
- ``GET /health`` — Health check
- ``POST /predict-rating`` — Predict movie rating
- ``POST /predict-boxoffice`` — Predict box office
- ``GET /actor/{name}`` — Bankability Score + chemistry pairs
- ``GET /model-info`` — Currently deployed models and metrics
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tamasha.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models on startup, clean up on shutdown."""
    logger.info("Tamasha API starting up...")
    # Models will be loaded here after training
    yield
    logger.info("Tamasha API shutting down...")


app = FastAPI(
    title="Tamasha API",
    description="Bollywood Movie Intelligence Platform API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint.

    Returns
    -------
    dict
        ``{"status": "ok", "version": "0.1.0"}``
    """
    return {"status": "ok", "version": "0.1.0"}


# Import routers
from api.routers import predict, network, model_info  # noqa: E402

app.include_router(predict.router)
app.include_router(network.router)
app.include_router(model_info.router)
