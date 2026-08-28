"""
pipeline_pkg — Backward-compatible re-exporter.

All pipeline stages live in focused sub-modules.
This file re-exports ``main`` so existing
``from tamasha.train_pipeline import main``
continues to work unchanged.
"""

from tamasha.pipeline_pkg.run import main

__all__ = ["main"]
