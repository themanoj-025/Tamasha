"""Network-graph rendering components for the Tamasha dashboard.

Uses Plotly for interactive force-directed graph visualization of the
cast/crew collaboration network.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import pandas as pd
import plotly.graph_objects as go


def render_force_directed_graph(
    G: nx.Graph,
    node_scores: Optional[pd.DataFrame] = None,
    highlight_actor: Optional[str] = None,
    title: str = "Star Collaboration Network",
    height: int = 600,
) -> go.Figure:
    """Render an interactive force-directed graph with Plotly.

    Node size encodes Bankability Score (if provided).
    Edge thickness encodes collaboration count.

    Parameters
    ----------
    G : nx.Graph
        Collaboration graph.
    node_scores : pd.DataFrame, optional
        DataFrame with ``actor`` and ``bankability_score`` columns.
    highlight_actor : str, optional
        If set, highlight this node and its neighbors.
    title : str, default="Star Collaboration Network"
        Plot title.
    height : int, default=600
        Plot height in pixels.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    # Compute layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Build score lookup
    score_map: dict[str, float] = {}
    if node_scores is not None:
        for _, row in node_scores.iterrows():
            score_map[row["actor"].lower()] = row["bankability_score"]

    # Node traces
    node_x = []
    node_y = []
    node_size = []
    node_color = []
    node_text = []
    node_hover = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Size
        score = score_map.get(node.lower(), 0.5)
        node_size.append(10 + score * 40)

        # Color
        is_highlighted = highlight_actor is not None and node.lower() == highlight_actor.lower()
        is_neighbor = highlight_actor is not None and G.has_edge(node, highlight_actor)
        if is_highlighted:
            node_color.append("#ff6b6b")
        elif is_neighbor:
            node_color.append("#ffd93d")
        else:
            node_color.append("#6bcbff")

        # Hover
        node_text.append(node)
        node_hover.append(
            f"<b>{node}</b><br>" f"Bankability: {score:.3f}<br>" f"Connections: {G.degree(node)}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=1, color="#ffffff"),
        ),
        textposition="top center",
        textfont=dict(size=10, color="#e0e0e0"),
    )

    # Edge traces
    edge_x = []
    edge_y = []
    edge_width = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_width.append(data.get("weight", 1.0))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(
            width=1,
            color="rgba(150, 150, 200, 0.3)",
        ),
        hoverinfo="none",
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text=title, font=dict(color="#ffffff")),
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=20, r=20, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            height=height,
            font=dict(color="#e0e0e0"),
        ),
    )

    return fig
