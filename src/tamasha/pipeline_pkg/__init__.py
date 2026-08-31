"""
pipeline_pkg — Backward-compatible re-exporter.

The implementation has been consolidated back into ``train_pipeline.py``
after a previous split attempt produced broken files.

This file re-exports ``main`` so existing
``from tamasha.train_pipeline import main`` imports continue
to work unchanged.
"""

from tamasha.train_pipeline import main

__all__ = ["main"]
