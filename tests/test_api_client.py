"""Smoke tests for the dashboard API client helper.

Verifies that api_client.py correctly builds auth headers and handles
missing keys gracefully.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.api_client import api_headers, get_api_key


class TestApiKeyRetrieval:
    """Test get_api_key() priority and error handling."""

    def test_reads_from_env_var(self) -> None:
        """Falls back to TAMASHA_API_KEY env var when st.secrets is unavailable."""
        with patch.dict(os.environ, {"TAMASHA_API_KEY": "test-key-123"}, clear=False):
            key = get_api_key()
            assert key == "test-key-123"

    def test_raises_when_no_key_configured(self) -> None:
        """Raises RuntimeError when neither secrets nor env var has the key."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove TAMASHA_API_KEY if it exists
            os.environ.pop("TAMASHA_API_KEY", None)
            with pytest.raises(RuntimeError, match="No API key configured"):
                get_api_key()


class TestApiHeaders:
    """Test api_headers() builds the correct header dict."""

    def test_returns_x_api_key_header(self) -> None:
        """api_headers() includes the X-API-Key header."""
        with patch.dict(os.environ, {"TAMASHA_API_KEY": "my-secret-key"}, clear=False):
            headers = api_headers()
            assert headers == {"X-API-Key": "my-secret-key"}

    def test_header_value_matches_env(self) -> None:
        """The header value exactly matches what get_api_key() returns."""
        with patch.dict(os.environ, {"TAMASHA_API_KEY": "another-key"}, clear=False):
            headers = api_headers()
            assert headers["X-API-Key"] == get_api_key()
