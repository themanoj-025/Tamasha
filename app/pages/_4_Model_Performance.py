"""Page 4: Model Performance — comparison charts, SHAP plots, model info."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from tamasha.config import settings
from tamasha.predict import get_model_info, get_comparison_csv


def _load_csv(csv_name: str) -> pd.DataFrame | None:
    path = settings.REPORTS_DIR / csv_name
    return pd.read_csv(path) if path.exists() else None


def _plot_comparison_bar(df: pd.DataFrame, title: str) -> go.Figure:
    """Render a grouped bar chart with styled bars."""
    fig = go.Figure()
    colors_mae = ["#3b82f6"] * len(df)
    colors_rmse = ["#6366f1"] * len(df)
    if len(df) > 0:
        colors_mae[0] = "#4ade80"
        colors_rmse[0] = "#4ade80"

    fig.add_trace(go.Bar(name="MAE", x=df["model"], y=df["MAE"], marker_color=colors_mae))
    fig.add_trace(go.Bar(name="RMSE", x=df["model"], y=df["RMSE"], marker_color=colors_rmse))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff", size=14)),
        xaxis_title="Model", yaxis_title="Error",
        barmode="group", template="plotly_dark", height=400,
        xaxis=dict(tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(color="#8a94b0")),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e0e0e0")),
    )
    return fig


def _radar_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Create a radar chart comparing top models across metrics."""
    if df is None or df.empty:
        return None
    top = df.head(5).copy()
    # Normalize metrics to 0-1 for radar
    for col in ["MAE", "RMSE"]:
        max_v = top[col].max()
        min_v = top[col].min()
        top[f"{col}_norm"] = 1 - (top[col] - min_v) / (max_v - min_v + 1e-8)

    fig = go.Figure()
    for _, row in top.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row["MAE_norm"], row["RMSE_norm"], row.get("R2", 0)],
            theta=["MAE (lower→better)", "RMSE (lower→better)", "R² (higher→better)"],
            fill="toself", name=row["model"],
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#fff")),
        template="plotly_dark", height=400,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color="#5a6380")),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="#e0e0e0")),
    )
    return fig


def show() -> None:
    """Render the enhanced Model Performance page."""
    st.markdown(
        "<div class='section-header'><h2>📊 Model Performance</h2>"
        "<p>Full model comparison across 9 candidates, best-model auto-selection, and SHAP explainability</p></div>",
        unsafe_allow_html=True,
    )

    info = get_model_info()

    # ── Top-level model info cards ──────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        rating_info = info["rating_model"]
        mae = rating_info["mae"]
        rmse = rating_info["rmse"]
        r2 = rating_info["r2"]
        metrics = (
            f"MAE: {mae:.4f} · RMSE: {rmse:.4f} · R²: {r2:.4f}"
            if mae is not None else "Metrics unavailable"
        )
        st.markdown(
            f"""
            <div class="metric-card featured">
                <div class="metric-label">🏆 Best Rating Model</div>
                <div class="metric-value gradient-text">{rating_info['algorithm']}</div>
                <div style="color:#8a94b0;font-size:0.85rem;margin-top:4px;">{metrics}</div>
            </div>
            """, unsafe_allow_html=True,
        )
    with col2:
        box_info = info["boxoffice_model"]
        mae_b = box_info["mae"]
        rmse_b = box_info["rmse"]
        r2_b = box_info["r2"]
        metrics_b = (
            f"MAE: ₹{mae_b/1e7:.1f}Cr · RMSE: ₹{rmse_b/1e7:.1f}Cr · R²: {r2_b:.4f}"
            if mae_b is not None else "Metrics unavailable"
        )
        st.markdown(
            f"""
            <div class="metric-card featured">
                <div class="metric-label">🏆 Best Box Office Model</div>
                <div class="metric-value gradient-text">{box_info['algorithm']}</div>
                <div style="color:#8a94b0;font-size:0.85rem;margin-top:4px;">{metrics_b}</div>
                <div class="metric-delta" style="margin-top:4px;">✨ + Bankability Score feature</div>
            </div>
            """, unsafe_allow_html=True,
        )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Rating Comparison", "💰 Box Office Comparison", "🔬 SHAP Explainability"]
    )

    # ── TAB 1: Rating Model Comparison ──────────────────────────────
    with tab1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        rating_csv = _load_csv("model_comparison_rating.csv")
        if rating_csv is not None:
            st.markdown("<h3 style='color:#fff;'>9 Models · 5-Fold CV · Auto-Selected by MAE</h3>", unsafe_allow_html=True)
            fig = _plot_comparison_bar(rating_csv, "Rating Model: MAE & RMSE")
            st.plotly_chart(fig, use_container_width=True)

            # Radar chart
            radar = _radar_chart(rating_csv, "Top 5 Models — Multi-Metric Radar")
            if radar:
                st.plotly_chart(radar, use_container_width=True)

            st.markdown("### Full Comparison Table")
            stylized = rating_csv.copy()
            stylized["MAE"] = stylized["MAE"].round(4)
            stylized["RMSE"] = stylized["RMSE"].round(4)
            stylized["R2"] = stylized["R2"].round(4)
            stylized["training_time_s"] = stylized["training_time_s"].round(2)
            st.dataframe(stylized, use_container_width=True, hide_index=True)

            # Scatter plots
            st.markdown("### Predicted vs. Actual (Top 3 Models)")
            cols = st.columns(3)
            for i, model_name in enumerate(rating_csv.head(3)["model"].tolist()):
                with cols[i]:
                    img_path = settings.FIGURES_DIR / f"rating_pred_vs_actual_{model_name.lower()}.png"
                    if img_path.exists():
                        st.image(str(img_path), caption=model_name, use_container_width=True)
                    else:
                        st.info(f"No plot for {model_name}")
        else:
            st.info("Rating comparison CSV not found. Run: make train")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 2: Box Office Comparison ────────────────────────────────
    with tab2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color:#fff;'>Baseline vs. With-Bankability-Score</h3>"
            "<p style='color:#8a94b0;'>This comparison proves our custom Bankability feature actually helped — "
            "not just that a model was trained.</p>",
            unsafe_allow_html=True,
        )

        baseline_csv = _load_csv("model_comparison_boxoffice_baseline.csv")
        bank_csv = _load_csv("model_comparison_boxoffice_with_bankability.csv")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<h4 style='color:#8a94b0;'>Baseline (No Bankability)</h4>", unsafe_allow_html=True)
            if baseline_csv is not None:
                fig = _plot_comparison_bar(baseline_csv, "")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(baseline_csv, use_container_width=True, hide_index=True)
            else:
                st.info("Not found")
        with col_b:
            st.markdown("<h4 style='color:#8a94b0;'>✨ With Bankability Score</h4>", unsafe_allow_html=True)
            if bank_csv is not None:
                fig = _plot_comparison_bar(bank_csv, "")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(bank_csv, use_container_width=True, hide_index=True)
            else:
                st.info("Not found")

        # Headline comparison
        if baseline_csv is not None and bank_csv is not None:
            base_mae = baseline_csv.iloc[0]["MAE"]
            bank_mae = bank_csv.iloc[0]["MAE"]
            improvement = (base_mae - bank_mae) / base_mae * 100
            st.markdown(
                f"""
                <div class="metric-card featured">
                    <div class="metric-label">🔥 Headline Result</div>
                    <div class="metric-value gradient-text">{improvement:.1f}% MAE Reduction</div>
                    <div class="metric-delta">
                        Baseline: ₹{base_mae/1e7:.1f} Cr → With Bankability: ₹{bank_mae/1e7:.1f} Cr
                    </div>
                    <div style="margin-top:8px;color:#8a94b0;font-size:0.85rem;">
                        Adding the Bankability Score feature improved box-office 
                        prediction error by <strong>{improvement:.1f}%</strong> — 
                        the strongest evidence that our custom network analysis feature 
                        engineering added genuine predictive signal.
                    </div>
                </div>
                """, unsafe_allow_html=True,
            )

        # Scatter plots
        st.markdown("### Predicted vs. Actual (Top 3 Bankability Models)")
        cols_s = st.columns(3)
        if bank_csv is not None:
            for i, model_name in enumerate(bank_csv.head(3)["model"].tolist()):
                with cols_s[i]:
                    img_path = settings.FIGURES_DIR / f"boxoffice_with_bank_pred_vs_actual_{model_name.lower()}.png"
                    if img_path.exists():
                        st.image(str(img_path), caption=model_name, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TAB 3: SHAP Explainability ──────────────────────────────────
    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("<h4 style='color:#fff;'>⭐ Rating Model (GradientBoosting)</h4>", unsafe_allow_html=True)
            shap_rating = settings.FIGURES_DIR / "shap_rating.png"
            if shap_rating.exists():
                st.image(str(shap_rating), use_container_width=True)
            else:
                st.info("SHAP plot not found")
        with col_s2:
            st.markdown("<h4 style='color:#fff;'>💰 Box Office Model (XGBoost + Bankability)</h4>", unsafe_allow_html=True)
            shap_box = settings.FIGURES_DIR / "shap_boxoffice.png"
            if shap_box.exists():
                st.image(str(shap_box), use_container_width=True)
            else:
                st.info("SHAP plot not found")
        st.markdown("</div>", unsafe_allow_html=True)

        # Key findings cards
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#fff;'>🔑 SHAP Key Findings</h3>", unsafe_allow_html=True)
        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            st.markdown(
                "<div class='metric-card'><div class='metric-label'>🎭 Genre Dominates Rating</div>"
                "<div class='metric-value' style='font-size:1.1rem;'>Drama, Action top the SHAP ranking</div>"
                "<div class='metric-delta'>Rating model</div></div>",
                unsafe_allow_html=True,
            )
        with col_k2:
            st.markdown(
                "<div class='metric-card featured'><div class='metric-label'>⭐ Bankability = #2 Feature</div>"
                "<div class='metric-value' style='font-size:1.1rem;'>Right after Budget</div>"
                "<div class='metric-delta'>Box office model · Validates custom feature</div></div>",
                unsafe_allow_html=True,
            )
        with col_k3:
            st.markdown(
                "<div class='metric-card'><div class='metric-label'>💰 Budget Matters Most</div>"
                "<div class='metric-value' style='font-size:1.1rem;'>#1 for Box Office</div>"
                "<div class='metric-delta'>Higher budget = higher predicted collection</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
