"""Shared API client helper for the Streamlit dashboard.

Provides a consistent way to retrieve the API key from Streamlit secrets
or environment variables, and to build auth headers for any HTTP call
the dashboard might make to the FastAPI backend.

.. note::

   The current dashboard calls prediction functions directly via Python
   imports (not HTTP), so these helpers are not yet wired into any page.
   They are provided as the standard pattern for any future HTTP-based
   dashboard calls, and for deployment scenarios where the dashboard
   runs separately from the API.
"""

from __future__ import annotations

import os

import streamlit as st


def get_api_key() -> str:
    """Retrieve the API key from Streamlit secrets or environment variables.

    Priority:
        1. ``st.secrets["API_KEY"]`` (works on Streamlit Cloud)
        2. ``TAMASHA_API_KEY`` env var (for local dev)

    Raises
    ------
    RuntimeError
        If neither source has a configured API key.
    """
    key: str | None = None

    # 1. Streamlit secrets (preferred for Streamlit Cloud deployments)
    try:
        if hasattr(st, "secrets") and "API_KEY" in st.secrets:
            key = st.secrets["API_KEY"]
    except Exception:
        pass

    # 2. Environment variable fallback (local development)
    if not key:
        key = os.getenv("TAMASHA_API_KEY")

    if not key:
        raise RuntimeError(
            "No API key configured. Set st.secrets['API_KEY'] or the "
            "TAMASHA_API_KEY environment variable. See README for details."
        )

    return key


def api_headers() -> dict[str, str]:
    """Return HTTP headers with the X-API-Key authentication header.

    Returns
    -------
    dict[str, str]
        Headers suitable for ``requests.get(..., headers=...)`` or
        ``httpx.Client(...).get(..., headers=...)``.
    """
    return {"X-API-Key": get_api_key()}


def api_key_error() -> None:
    """Display a clean Streamlit error when the API key is not configured.

    Call this at the top of any page that needs the API key, wrapped
    in a try/except around :func:`get_api_key`.
    """
    st.error(
        "🔑 **API key not configured**\n\n"
        "Set `API_KEY` in `.streamlit/secrets.toml` (for Streamlit Cloud) "
        "or the `TAMASHA_API_KEY` environment variable (for local dev).\n\n"
        "See the [README](../README.md#authentication) for setup instructions."
    )
