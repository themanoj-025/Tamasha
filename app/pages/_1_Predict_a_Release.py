"""Page 1: Predict a Release — genre/cast/budget inputs -> predictions."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from tamasha.predict import predict_boxoffice, predict_rating


def _star_rating_html(rating: float, max_stars: float = 10.0) -> str:
    """Generate HTML for a visual star rating."""
    normalized = rating / max_stars * 5
    full = int(normalized)
    half = 1 if normalized - full >= 0.3 else 0
    empty = 5 - full - half
    stars = "⭐" * full + "✨" * half + "☆" * empty
    return f'<span style="font-size:1.4rem;letter-spacing:2px;">{stars}</span>'


def _gauge_chart(value: float, max_val: float, title: str, color: str = "#3b82f6") -> go.Figure:
    """Create a speedometer-style gauge chart."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "", "font": {"color": "#ffffff", "size": 28}},
            title={"text": title, "font": {"color": "#8a94b0", "size": 14}},
            gauge={
                "axis": {
                    "range": [0, max_val],
                    "tickcolor": "#5a6380",
                    "tickfont": {"color": "#8a94b0"},
                },
                "bar": {"color": color},
                "bgcolor": "#1a1d24",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, max_val * 0.33], "color": "rgba(239,68,68,0.15)"},
                    {"range": [max_val * 0.33, max_val * 0.66], "color": "rgba(234,179,8,0.15)"},
                    {"range": [max_val * 0.66, max_val], "color": "rgba(74,222,128,0.15)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e0e0e0"},
    )
    return fig


def show() -> None:
    """Render the enhanced Predict a Release page."""
    st.markdown(
        "<div class='section-header'><h2>🔮 Predict a Release</h2>"
        "<p>Enter a hypothetical film profile and get real-time rating + box office predictions</p></div>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        st.markdown(
            "<div class='glass-card'><h3 style='margin:0 0 1rem;color:#fff;'>🎬 Movie Profile</h3>",
            unsafe_allow_html=True,
        )

        st.text_input("Movie Title", value="My Bollywood Film")

        genre_options = [
            "Action",
            "Comedy",
            "Drama",
            "Romance",
            "Thriller",
            "Horror",
            "Sci-Fi",
            "Musical",
            "Biography",
            "Crime",
            "Mystery",
            "Fantasy",
            "Animation",
            "Family",
        ]
        genres = st.multiselect("Genres", genre_options, default=["Drama", "Romance"])

        director = st.text_input("Director", value="Sanjay Leela Bhansali")

        cast_str = st.text_input(
            "Cast (comma-separated)",
            value="Shah Rukh Khan, Deepika Padukone",
            help="Enter actor names separated by commas",
        )
        cast_list = [c.strip() for c in cast_str.split(",") if c.strip()]

        budget_cr = st.number_input(
            "Budget (₹ Crore)", min_value=1.0, max_value=500.0, value=80.0, step=5.0
        )
        budget_inr = budget_cr * 1e7

        runtime = st.slider(
            "Runtime (minutes)", 90, 240, 150, help="Typical Bollywood films run 120-180 min"
        )

        col_y, col_w = st.columns(2)
        with col_y:
            release_year = st.number_input(
                "Release Year", min_value=2020, max_value=2030, value=2024
            )
        with col_w:
            release_windows = [
                "Normal",
                "Diwali",
                "Eid",
                "Christmas",
                "Independence Day",
                "Republic Day",
                "New Year",
            ]
            release_window = st.selectbox("Release Window", release_windows)

        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("🎯 Generate Predictions", type="primary", use_container_width=True)

    with col2:
        st.markdown(
            "<h3 style='color:#fff;margin:0 0 1rem;'>📊 Results</h3>", unsafe_allow_html=True
        )

        if predict_btn:
            if not genres:
                st.error("Please select at least one genre.")
                return
            if not cast_list:
                st.error("Please enter at least one cast member.")
                return

            with st.spinner("Running predictions..."):
                rating_result = predict_rating(
                    genres=genres,
                    cast=cast_list,
                    director=director,
                    budget_inr=budget_inr,
                    runtime_minutes=runtime,
                    year=release_year,
                )
                box_result = predict_boxoffice(
                    genres=genres,
                    cast=cast_list,
                    director=director,
                    budget_inr=budget_inr,
                    runtime_minutes=runtime,
                    year=release_year,
                    release_window=release_window,
                )

            # ── Rating Card ────────────────────────────────────────
            if rating_result["predicted_rating"] is not None:
                pred_r = rating_result["predicted_rating"]
                stars_html = _star_rating_html(pred_r)
                mae_str = f"{rating_result['model_mae']:.2f}"
                st.markdown(
                    f"""
                    <div class="metric-card featured">
                        <div class="metric-label">⭐ Predicted IMDB Rating</div>
                        <div class="metric-value gradient-text">{pred_r} / 10</div>
                        <div style="margin:4px 0;">{stars_html}</div>
                        <div class="metric-delta">Model: {rating_result['model_name']} · MAE: ±{mae_str}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Mini gauge
                fig_r = _gauge_chart(pred_r, 10.0, "Rating Score", "#a78bfa")
                st.plotly_chart(fig_r, use_container_width=True)
            else:
                st.markdown(
                    "<div class='metric-card'>"
                    "<div class='metric-label'>⭐ Predicted IMDB Rating</div>"
                    "<div class='metric-value' style='color:#f87171;'>⚠️ Unavailable</div>"
                    "<div class='metric-delta'>Run: make train</div></div>",
                    unsafe_allow_html=True,
                )

            # ── Box Office Card ────────────────────────────────────
            if box_result["predicted_boxoffice_cr"] is not None:
                pred_b = box_result["predicted_boxoffice_cr"]
                mae_b = box_result["model_mae"]

                bank_note = ""
                if box_result.get("fallback_actors"):
                    bank_note = "<div class='info-box warning'>Some actors lack historical data — used genre-average Bankability fallback.</div>"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">💰 Predicted Box Office</div>
                        <div class="metric-value gradient-text">₹ {pred_b:,.1f} Cr</div>
                        <div class="metric-delta">Model: {box_result['model_name']} · MAE: ±₹{mae_b:.1f} Cr</div>
                    </div>
                    {bank_note}
                    """,
                    unsafe_allow_html=True,
                )

                # Bankability info
                bank_info = box_result.get("bankability_info", {})
                if bank_info:
                    fallback_str = (
                        f" ({bank_info['fallback_count']}/{bank_info['total_count']} fallback)"
                        if bank_info["fallback_count"] > 0
                        else ""
                    )
                    st.markdown(
                        f"<div class='info-box info'>"
                        f"<b>🎭 Cast Bankability:</b> Avg Score = {bank_info['avg_score']:.3f}{fallback_str}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Gauge: Budget vs Box Office
                if budget_cr > 0:
                    roi = pred_b / budget_cr
                    fig_b = _gauge_chart(roi, 5.0, "Budget Multiple (ROI)", "#4ade80")
                    st.plotly_chart(fig_b, use_container_width=True)

                # ── Scenario Comparison ─────────────────────────────
                st.markdown(
                    "<h4 style='color:#fff;margin:1rem 0 0.5rem;'>📅 Release Scenario Comparison</h4>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    "<div class='info-box info'>Scenario simulation — shows relative box office under different release windows. Not a guaranteed forecast.</div>",
                    unsafe_allow_html=True,
                )

                scenarios = box_result.get("scenarios", {})
                if scenarios:
                    window_list = list(scenarios.keys())
                    values = [scenarios[w] for w in window_list]
                    colors = ["#6bcbff"] * len(window_list)
                    if release_window in window_list:
                        idx = window_list.index(release_window)
                        colors[idx] = "#ffd93d"

                    fig_s = go.Figure()
                    fig_s.add_trace(
                        go.Bar(
                            x=window_list,
                            y=values,
                            marker_color=colors,
                            text=[f"₹{v:.0f} Cr" for v in values],
                            textposition="outside",
                            textfont=dict(color="#e0e0e0", size=11),
                        )
                    )
                    fig_s.update_layout(
                        title=dict(
                            text="Predicted Box Office by Window (₹ Cr)",
                            font=dict(color="#ffffff", size=14),
                        ),
                        template="plotly_dark",
                        height=320,
                        margin=dict(l=20, r=20, t=40, b=60),
                        xaxis=dict(tickfont=dict(color="#8a94b0", size=10)),
                        yaxis=dict(visible=False),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig_s, use_container_width=True)
            else:
                st.markdown(
                    "<div class='metric-card'><div class='metric-label'>💰 Predicted Box Office</div>"
                    "<div class='metric-value' style='color:#f87171;'>⚠️ Unavailable</div>"
                    "<div class='metric-delta'>Run: make train</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='glass-card' style='text-align:center;padding:3rem 1rem;'>"
                "<div style='font-size:3rem;margin-bottom:1rem;'>🎯</div>"
                "<h3 style='color:#fff;margin:0 0 0.5rem;'>Ready to Predict</h3>"
                "<p style='color:#8a94b0;margin:0;'>Adjust the movie profile on the left and click <strong>Generate Predictions</strong>.</p>"
                "</div>",
                unsafe_allow_html=True,
            )
