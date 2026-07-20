"""Main entry point for the Streamlit multi-page dashboard.

Uses ``st.cache_resource`` to create the ``PredictionService`` singleton
once, avoiding model reloading on every page rerun.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from tamasha.predict import PredictionService


@st.cache_resource(ttl=3600)
def _load_prediction_service() -> PredictionService:
    """Load the PredictionService singleton (cached by Streamlit)."""
    svc = PredictionService()
    svc.load()
    return svc


# ── Ensure the service is loaded once at startup ──────────────────────
_svc = _load_prediction_service()

st.set_page_config(
    page_title="Tamasha — Bollywood Movie Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject global CSS + animations ────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "theme.css"
if css_path.exists():
    with css_path.open() as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Ensure NLTK VADER data is available ──────────────────────────────
try:
    import nltk

    nltk.download("vader_lexicon", quiet=True)
except Exception:
    pass

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div class="sidebar-header">
        <h1>🎬 Tamasha</h1>
        <p class="subtitle">Bollywood Intelligence Platform</p>
        <div style="margin-top:12px;font-size:0.7rem;color:#5a6380;">
            GradientBoosting · XGBoost · 1,010 scores
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Animated status indicator
healthy = _svc.healthy
dot_color = "#4ade80" if healthy else "#f87171"
status_text = (
    "Models loaded &mdash; ready" if healthy else "Models not trained &mdash; run: make train"
)
st.sidebar.markdown(
    f"""
    <div class="sidebar-status">
        <span class="dot" style="background:{dot_color};"></span>
        <span>{status_text}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigate",
    [
        "🔮 Predict a Release",
        "⭐ Star Network Explorer",
        "📊 Industry Trends",
        "📈 Model Performance",
    ],
    label_visibility="collapsed",
)

# Decorative sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <p style="margin:0 0 4px;font-weight:600;color:#6b7280;">
            🎯 Key Insight
        </p>
        <p style="margin:0;font-size:0.75rem;line-height:1.4;">
            Bankability Score improved<br>
            box-office MAE by <strong style="color:#4ade80;">8.7%</strong>
        </p>
        <p style="margin-top:12px;font-size:0.65rem;opacity:0.4;">
            ML Portfolio Project
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Page routing ──────────────────────────────────────────────────────
if page == "🔮 Predict a Release":
    from app.pages._1_Predict_a_Release import show as show_page1

    show_page1()
elif page == "⭐ Star Network Explorer":
    from app.pages._2_Star_Network_Explorer import show as show_page2

    show_page2()
elif page == "📊 Industry Trends":
    from app.pages._3_Industry_Trends import show as show_page3

    show_page3()
elif page == "📈 Model Performance":
    from app.pages._4_Model_Performance import show as show_page4

    show_page4()
