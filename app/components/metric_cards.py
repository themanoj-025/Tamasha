"""Custom metric-card components for the Tamasha dashboard."""

from __future__ import annotations

from typing import Optional

import streamlit as st


def glass_card(
    content: str,
    height: Optional[int] = None,
) -> None:
    """Render a glassmorphism container card."""
    style = f"height:{height}px;" if height else ""
    st.markdown(
        f'<div class="glass-card" style="{style}">{content}</div>',
        unsafe_allow_html=True,
    )


def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
    featured: bool = False,
    gradient_value: bool = False,
) -> None:
    """Render a styled metric card.

    Parameters
    ----------
    label : str
        Metric label.
    value : str
        Metric value (formatted).
    delta : str, optional
        Change indicator (e.g. "+5%").
    help_text : str, optional
        Tooltip text.
    featured : bool, default=False
        Whether to apply featured (highlighted) styling.
    gradient_value : bool, default=False
        Whether to apply gradient text to the value.
    """
    featured_cls = " featured" if featured else ""
    gradient_cls = " gradient-text" if gradient_value else ""
    delta_html = (
        f'<span class="metric-delta">{delta}</span>'
        if delta else ""
    )
    help_icon = (
        f'<span class="metric-help" title="{help_text}">ⓘ</span>'
        if help_text else ""
    )
    st.markdown(
        f"""
        <div class="metric-card{featured_cls}">
            <div class="metric-label">{label}{help_icon}</div>
            <div class="metric-value{gradient_cls}">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge(text: str, color: str = "blue") -> None:
    """Render a small inline badge."""
    colors = {
        "blue": "#3b82f6",
        "green": "#4ade80",
        "red": "#f87171",
        "yellow": "#facc15",
        "purple": "#a78bfa",
        "pink": "#f472b6",
    }
    c = colors.get(color, "#3b82f6")
    st.markdown(
        f'<span style="display:inline-block;padding:2px 10px;'
        f'border-radius:20px;font-size:0.7rem;font-weight:600;'
        f'background:{c}20;color:{c};border:1px solid {c}30;">'
        f"{text}</span>",
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: Optional[str] = None) -> None:
    """Render a styled section header."""
    st.markdown(
        f"""
        <div class="section-header">
            <h2>{title}</h2>
            {f'<p>{subtitle}</p>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_box(text: str, box_type: str = "info") -> None:
    """Render a styled info/warning/error/success box."""
    st.markdown(
        f'<div class="info-box {box_type}">{text}</div>',
        unsafe_allow_html=True,
    )
