"""Prediction response cache using diskcache.

Caches prediction results keyed on a stable hash of the request payload
plus the currently-loaded model version, so cache invalidates automatically
on model redeploy. TTL of 1 hour by default.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

import diskcache

from tamasha.config import settings

logger = logging.getLogger(__name__)

# Cache directory under DATA_PROCESSED
_CACHE_DIR = settings.DATA_PROCESSED / "prediction_cache"
_CACHE: Optional[diskcache.Cache] = None

_DEFAULT_TTL = 3600  # 1 hour
_EXPLANATION_TTL = 86400  # 24 hours for LLM explanations


def _get_cache() -> diskcache.Cache:
    """Get-or-create the diskcache instance."""
    global _CACHE
    if _CACHE is None:
        _CACHE = diskcache.Cache(str(_CACHE_DIR))
        logger.info("Prediction cache initialized at %s", _CACHE_DIR)
    return _CACHE


def _make_key(payload: dict[str, Any], model_version: str = "") -> str:
    """Create a deterministic cache key from request payload + model version.

    Parameters
    ----------
    payload : dict
        The prediction request payload.
    model_version : str
        Current model version string for cache invalidation on redeploy.

    Returns
    -------
    str
        SHA-256 hex digest suitable as a cache key.
    """
    # Sort keys for deterministic serialization
    canonical = json.dumps(payload, sort_keys=True, default=str)
    key_input = f"{model_version}:{canonical}"
    return hashlib.sha256(key_input.encode()).hexdigest()


def get_cached_prediction(payload: dict[str, Any], model_version: str = "") -> Optional[dict[str, Any]]:
    """Look up a cached prediction result.

    Parameters
    ----------
    payload : dict
        The prediction request payload.
    model_version : str
        Current model version for invalidation.

    Returns
    -------
    dict or None
        Cached result, or None on miss.
    """
    cache = _get_cache()
    key = _make_key(payload, model_version)
    result = cache.get(key, default=None, expire_time=True)
    if result is not None:
        # diskcache returns (value, expire_time) tuple with expire_time=True
        # Actually, with expire_time=True it returns (value, expire_time) or (default, None)
        if isinstance(result, tuple):
            value, expire_time = result
            if value is not None:
                logger.debug("Cache HIT for key=%s", key[:12])
                return value
        else:
            # Shouldn't happen with expire_time=True, but handle gracefully
            logger.debug("Cache HIT for key=%s", key[:12])
            return result
    logger.debug("Cache MISS for key=%s", key[:12])
    return None


def set_cached_prediction(
    payload: dict[str, Any],
    result: dict[str, Any],
    model_version: str = "",
    ttl: int = _DEFAULT_TTL,
) -> None:
    """Store a prediction result in cache.

    Parameters
    ----------
    payload : dict
        The request payload (used to generate key).
    result : dict
        The prediction result to cache.
    model_version : str
        Current model version.
    ttl : int
        Time-to-live in seconds. Default 3600 (1 hour).
    """
    cache = _get_cache()
    key = _make_key(payload, model_version)
    cache.set(key, result, expire=ttl)
    logger.debug("Cache SET for key=%s (ttl=%ds)", key[:12], ttl)


def get_cached_explanation(payload: dict[str, Any], model_version: str = "") -> Optional[dict[str, Any]]:
    """Look up a cached LLM explanation (longer TTL)."""
    cache = _get_cache()
    key = f"explain:{_make_key(payload, model_version)}"
    result = cache.get(key, default=None, expire_time=True)
    if isinstance(result, tuple):
        value, _ = result
        if value is not None:
            return value
        return None
    return result if result is not None else None


def set_cached_explanation(
    payload: dict[str, Any],
    result: dict[str, Any],
    model_version: str = "",
    ttl: int = _EXPLANATION_TTL,
) -> None:
    """Store an LLM explanation in cache (24h TTL)."""
    cache = _get_cache()
    key = f"explain:{_make_key(payload, model_version)}"
    cache.set(key, result, expire=ttl)


def clear_cache() -> int:
    """Clear all cached predictions. Returns number of entries removed."""
    cache = _get_cache()
    count = len(cache)
    cache.clear()
    logger.info("Prediction cache cleared (%d entries removed)", count)
    return count
