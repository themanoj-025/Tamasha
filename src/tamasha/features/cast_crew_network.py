"""Build and analyse the cast/crew collaboration graph.

Nodes represent actors and directors. Weighted edges represent
co-appearances, weighted by the film's rating and box-office
performance.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def build_collaboration_graph(
    df: pd.DataFrame,
    cast_column: str = "cast",
    director_column: str = "director",
    rating_column: Optional[str] = None,
    boxoffice_column: Optional[str] = None,
) -> nx.Graph:
    """Build an undirected collaboration graph from movie data.

    Parameters
    ----------
    df : pd.DataFrame
        Movie dataset with cast and director info.
    cast_column : str, default="cast"
        Column with comma-separated cast names.
    director_column : str, default="director"
        Column with director name.
    rating_column : str, optional
        Column with numeric rating (used for edge weight).
    boxoffice_column : str, optional
        Column with numeric box office (used for edge weight).

    Returns
    -------
    nx.Graph
        Graph where node attributes include ``total_appearances``.
        Edge weight is derived from rating + box office.
    """
    G = nx.Graph()

    for _, row in df.iterrows():
        director = str(row.get(director_column, "")).strip()
        cast_raw = str(row.get(cast_column, "")).strip()

        # Parse director
        if director and director.lower() != "nan":
            director = director
            G.add_node(director, type="director", total_appearances=0)
        else:
            director = None

        # Parse cast
        cast_members = [c.strip() for c in cast_raw.split(",") if c.strip()]
        cast_members = [
            c for c in cast_members if c.lower() != "nan"
        ]

        # Edge weight based on performance
        weight = 1.0
        if rating_column and rating_column in row:
            r = pd.to_numeric(row[rating_column], errors="coerce")
            if pd.notna(r):
                weight *= r / 10.0  # Normalize rating 0-10 → 0-1
        if boxoffice_column and boxoffice_column in row:
            b = pd.to_numeric(row[boxoffice_column], errors="coerce")
            if pd.notna(b) and b > 0:
                weight *= 1 + (b / b)  # Relative contribution

        # Add cast nodes and edges between co-stars
        for actor in cast_members:
            G.add_node(actor, type="actor", total_appearances=0)
            G.nodes[actor]["total_appearances"] = (
                G.nodes[actor].get("total_appearances", 0) + 1
            )

        # Edges between co-stars
        for i, a1 in enumerate(cast_members):
            for a2 in cast_members[i + 1 :]:
                if G.has_edge(a1, a2):
                    G[a1][a2]["weight"] += weight
                    G[a2][a1]["weight"] += weight
                else:
                    G.add_edge(a1, a2, weight=weight)

        # Edge between director and each cast member
        if director:
            for actor in cast_members:
                G.nodes[director]["total_appearances"] = (
                    G.nodes[director].get("total_appearances", 0) + 1
                )
                if G.has_edge(director, actor):
                    G[director][actor]["weight"] += weight
                else:
                    G.add_edge(director, actor, weight=weight)

    logger.info(
        "Collaboration graph: %d nodes, %d edges",
        G.number_of_nodes(),
        G.number_of_edges(),
    )
    return G


def get_top_collaborators(
    G: nx.Graph,
    actor: str,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Get the top collaborators for a given actor/director.

    Parameters
    ----------
    G : nx.Graph
        Collaboration graph.
    actor : str
        Node name.
    top_n : int, default=10
        Number of top collaborators to return.

    Returns
    -------
    list[tuple[str, float]]
        List of ``(collaborator_name, edge_weight)`` sorted descending.
    """
    if actor not in G:
        return []
    neighbors = list(G.neighbors(actor))
    weighted = [
        (n, G[actor][n].get("weight", 1.0))
        for n in neighbors
    ]
    weighted.sort(key=lambda x: x[1], reverse=True)
    return weighted[:top_n]
