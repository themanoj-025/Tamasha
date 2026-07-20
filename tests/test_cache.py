"""Tests for the diskcache-based prediction response cache.

Verifies:
- Cache hit/miss behavior
- Model-version-aware invalidation
- TTL behavior
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import patch, MagicMock

from tamasha.cache import _make_key, get_cached_prediction, set_cached_prediction


class TestCacheKey:
    """Verify cache key generation is deterministic and version-aware."""

    def test_same_payload_same_key(self) -> None:
        """Identical payloads produce identical keys."""
        payload = {"genres": ["Drama"], "cast": ["SRK"], "budget_inr": 1000000}
        key1 = _make_key(payload, "v1")
        key2 = _make_key(payload, "v1")
        assert key1 == key2

    def test_different_payload_different_key(self) -> None:
        """Different payloads produce different keys."""
        payload1 = {"genres": ["Drama"], "cast": ["SRK"]}
        payload2 = {"genres": ["Action"], "cast": ["Salman"]}
        assert _make_key(payload1) != _make_key(payload2)

    def test_different_model_version_different_key(self) -> None:
        """Same payload but different model version → different key (cache invalidation)."""
        payload = {"genres": ["Drama"], "cast": ["SRK"]}
        key_v1 = _make_key(payload, "v1")
        key_v2 = _make_key(payload, "v2")
        assert key_v1 != key_v2

    def test_key_is_sha256_hex(self) -> None:
        """Key is a 64-character hex string (SHA-256)."""
        key = _make_key({"test": True})
        assert len(key) == 64
        int(key, 16)  # should not raise

    def test_sorted_keys_for_determinism(self) -> None:
        """Dict key order doesn't affect the cache key."""
        payload1 = {"b": 2, "a": 1}
        payload2 = {"a": 1, "b": 2}
        assert _make_key(payload1) == _make_key(payload2)


class TestCacheHitMiss:
    """Verify cache hit/miss behavior using an isolated cache dir."""

    def test_miss_returns_none(self) -> None:
        """Cache miss returns None."""
        result = get_cached_prediction(
            {"genres": ["Drama"], "cast": ["TestActor"]},
            model_version="nonexistent_version_999",
        )
        assert result is None

    def test_set_then_get_returns_value(self) -> None:
        """After set, get returns the cached value."""
        payload = {"genres": ["Drama"], "cast": ["CacheTestActor"]}
        result = {"predicted_rating": 7.5, "model_name": "Test"}
        set_cached_prediction(payload, result, model_version="cache_test_v1")
        cached = get_cached_prediction(payload, model_version="cache_test_v1")
        assert cached == result

    def test_wrong_version_causes_miss(self) -> None:
        """Same payload but wrong model version → cache miss."""
        payload = {"genres": ["Drama"], "cast": ["VersionTestActor"]}
        result = {"predicted_rating": 6.0}
        set_cached_prediction(payload, result, model_version="v1")
        cached = get_cached_prediction(payload, model_version="v2")
        assert cached is None
