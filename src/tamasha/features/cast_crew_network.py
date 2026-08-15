"""Build and analyse the cast/crew collaboration graph.

Nodes represent actors and directors. Weighted edges represent
co-appearances, weighted by the film's rating and box-office
performance.

.. note::

   The graph-building functions ``build_collaboration_graph`` and
   ``get_top_collaborators`` were removed in a refactor as they were
   unused (no imports from any other module).  The ``networkx``
   dependency remains because the dashboard's network-graph component
   uses it directly to build small co-occurrence subgraphs from
   Bankability and chemistry data.
"""
