import pytest

pytestmark = pytest.mark.unit

"""Tests for prediction response cache."""

import tempfile
from unittest.mock import patch

from tamasha.cache import _make_key, get_cached_prediction, set_cached_prediction


class TestMakeKey:
    """Tests for _make_key."""

    def test_deterministic(self) -> None:
        payload = {"title": "Test Movie", "year": 2024}
        key1 = _make_key(payload, "v1")
        key2 = _make_key(payload, "v1")
        assert key1 == key2

    def test_different_payloads(self) -> None:
        key1 = _make_key({"a": 1}, "v1")
        key2 = _make_key({"a": 2}, "v1")
        assert key1 != key2

    def test_different_versions(self) -> None:
        payload = {"a": 1}
        key1 = _make_key(payload, "v1")
        key2 = _make_key(payload, "v2")
        assert key1 != key2

    def test_order_independent(self) -> None:
        key1 = _make_key({"a": 1, "b": 2}, "v1")
        key2 = _make_key({"b": 2, "a": 1}, "v1")
        assert key1 == key2


class TestCachedPrediction:
    """Tests for get/set cached prediction."""

    def test_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("tamasha.cache._CACHE_DIR", tmpdir):
                import tamasha.cache as mod
                mod._CACHE = None
                result = get_cached_prediction({"test": True}, "v1")
                assert result is None
                mod._CACHE = None

    def test_cache_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("tamasha.cache._CACHE_DIR", tmpdir):
                import tamasha.cache as mod
                mod._CACHE = None
                payload = {"title": "Test"}
                set_cached_prediction(payload, {"result": "hit"}, "v1")
                result = get_cached_prediction(payload, "v1")
                assert result is not None
                assert result["result"] == "hit"
                mod._CACHE = None
