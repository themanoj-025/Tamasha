"""
train_pipeline.py — Tamasha Training Pipeline

.. note::

   The implementation has been refactored into ``pipeline_pkg/``
   for maintainability.  This module re-exports ``main`` so existing
   ``from tamasha.train_pipeline import main`` imports continue to
   work unchanged.
"""

from tamasha.pipeline_pkg import main

__all__ = ["main"]
