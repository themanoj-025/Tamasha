"""Page 2: Star Network Explorer — interactive force-directed graph."""

from __future__ import annotations

import streamlit as st

from tamasha.data.enrichment import get_actor_photo_url
from tamasha.predict import get_actor_info, get_bankability_scores, get_chemistry_pairs


def _actor_card_html(info: dict) -> str:
    """Generate rich actor info card HTML."""
    score = info["bankability_score"]
    score_color = (
        "#4ade80" if score and score >= 1.0 else "#facc15" if score and score >= 0.5 else "#f87171"
    )
    score_str = f"{score:.4f}" if score else "N/A"
    pairs = info["top_chemistry_pairs"]
    pairs_html = ""
    if pairs:
        items = "".join(
            f'<span style="display:inline-block;background:rgba(99,102,241,0.1);'
            f"border-radius:12px;padding:2px 10px;margin:2px;font-size:0.75rem;"
            f'color:#a78bfa;">{p["actor"]} +{p["chemistry_score"]:.3f}</span>'
            for p in pairs[:5]
        )
        pairs_html = f'<div style="margin-top:8px;"><strong style="color:#8a94b0;font-size:0.75rem;">CHEMISTRY PARTNERS</strong><br>{items}</div>'

    return f"""
    <div class="glass-card" style="text-align:center;">
        <div style="font-size:2.5rem;margin-bottom:4px;">{'🎭' if info['type']=='actor' else '🎬'}</div>
        <h3 style="color:#fff;margin:0;font-size:1.3rem;">{info['name']}</h3>
        <div style="margin:4px 0;">
            <span style="display:inline-block;padding:2px 12px;border-radius:20px;
            font-size:0.7rem;font-weight:600;text-transform:uppercase;
            background:{score_color}20;color:{score_color};
            border:1px solid {score_color}30;">{info['type']}</span>
        </div>
        <div class="metric-value gradient-text" style="font-size:2rem;">{score_str}</div>
        <div style="color:#8a94b0;font-size:0.8rem;margin-bottom:4px;">Bankability Score</div>
        <div style="color:#5a6380;font-size:0.75rem;">{info['film_count']} films in dataset</div>
        {pairs_html}
    </div>
    """


def show() -> None:
    """Render the enhanced Star Network Explorer page."""
    st.markdown(
        "<div class='section-header'><h2>⭐ Star Network Explorer</h2>"
        "<p>Bankability Scores, chemistry pairings, and the Bollywood collaboration network</p></div>",
        unsafe_allow_html=True,
    )

    scores_df = get_bankability_scores()
    chem_df = get_chemistry_pairs()

    col1, col2 = st.columns([1, 1.8], gap="large")

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color:#fff;margin:0 0 1rem;'>🔍 Search & Filters</h3>",
            unsafe_allow_html=True,
        )

        actor_search = st.text_input("Search Actor", placeholder="e.g., Shah Rukh Khan")

        if actor_search:
            info = get_actor_info(actor_search)
            if info["found"]:
                col_photo, col_card = st.columns([1, 2])
                with col_photo:
                    photo_url = get_actor_photo_url(actor_search)
                    if photo_url:
                        st.image(photo_url, width=120, caption=actor_search)
                    else:
                        st.markdown(
                            f"<div style='font-size:3rem;text-align:center;'>{'🎭' if info['type']=='actor' else '🎬'}</div>",
                            unsafe_allow_html=True,
                        )
                with col_card:
                    st.markdown(_actor_card_html(info), unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<div class='info-box warning'>"
                    f"<b>{actor_search}</b> not found in the Bankability dataset (1,010 individuals)."
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

        # Filters
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        if scores_df is not None and not scores_df.empty:
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                min_score = st.slider(
                    "Min Score", 0.0, 2.0, 0.3, 0.05, help="Minimum Bankability Score"
                )
            with col_s2:
                min_films = st.slider("Min Films", 1, 30, 2, help="Minimum number of films")

            top_n = st.number_input("Top N Actors", min_value=5, max_value=100, value=30)

            filtered = scores_df[
                (scores_df["bankability_score"] >= min_score)
                & (scores_df["film_count"] >= min_films)
            ].head(top_n)

            st.markdown(
                f"<p style='color:#8a94b0;font-size:0.8rem;'>{len(filtered)} actors match filters</p>",
                unsafe_allow_html=True,
            )

            st.markdown(f"### Top {len(filtered)} Bankability Scores")
            st.dataframe(
                filtered[["actor", "type", "bankability_score", "film_count"]],
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # Chemistry pairs panel
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color:#fff;margin:0 0 0.75rem;'>🤝 Top Chemistry Pairs</h3>",
            unsafe_allow_html=True,
        )
        if chem_df is not None and not chem_df.empty:
            for _, row in chem_df.head(8).iterrows():
                uplift_pct = row["uplift"] * 100
                st.markdown(
                    f'<div style="display:flex;align-items:center;justify-content:space-between;'
                    f'padding:6px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
                    f'<span style="color:#e0e0e0;">{row["actor_1"]} & {row["actor_2"]}</span>'
                    f'<span style="color:#4ade80;font-weight:600;">+{uplift_pct:.2f}%</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='info-box info'>No chemistry pairs found with ≥2 joint films.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color:#fff;margin:0 0 0.5rem;'>🌐 Collaboration Network</h3>",
            unsafe_allow_html=True,
        )

        if scores_df is not None and not scores_df.empty:
            import networkx as nx

            from app.components.network_graph import render_force_directed_graph

            G = nx.Graph()
            top_actors = scores_df[
                (scores_df["bankability_score"] >= min_score)
                & (scores_df["film_count"] >= min_films)
            ].head(50)

            score_map = {}
            for _, row in top_actors.iterrows():
                G.add_node(row["actor"], type=row["type"])
                score_map[row["actor"].lower()] = row["bankability_score"]

            if chem_df is not None and not chem_df.empty:
                for _, row in chem_df.iterrows():
                    a1, a2 = row["actor_1"], row["actor_2"]
                    if a1.lower() in score_map and a2.lower() in score_map:
                        G.add_edge(a1, a2, weight=row["uplift"])

            if G.number_of_nodes() > 0:
                node_scores = scores_df[
                    scores_df["actor"].str.lower().isin([n.lower() for n in G.nodes()])
                ]
                fig = render_force_directed_graph(
                    G,
                    node_scores=node_scores,
                    highlight_actor=actor_search if actor_search else None,
                    title=f"Star Network ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)",
                    height=600,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown(
                    "<div class='info-box info'>Not enough connected nodes. "
                    "Lower the minimum Bankability Score or Films threshold.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div class='info-box info'>Bankability scores not yet computed. Run: make train</div>",
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Filmography Timeline
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color:#fff;margin:0 0 0.5rem;'>📅 Filmography</h3>", unsafe_allow_html=True
        )
        if actor_search:
            info = get_actor_info(actor_search)
            if info["found"]:
                chem = info["top_chemistry_pairs"]
                chem_str = (
                    ", ".join(
                        f"<strong>{p['actor']}</strong> (×{p['joint_films']})" for p in chem[:5]
                    )
                    if chem
                    else "None found"
                )
                st.markdown(
                    f"<div class='info-box info'>"
                    f"<b>{info['name']}</b> — Bankability: {info['bankability_score']:.4f}, "
                    f"{info['film_count']} films, type: {info['type']}.<br>"
                    f"<b>Chemistry partners:</b> {chem_str}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='info-box warning'>Actor not found in dataset.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<div style='text-align:center;padding:1.5rem;color:#5a6380;'>"
                "Search for an actor above to see their filmography details.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
