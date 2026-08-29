"""Convenience wrappers around PredictionService."""

from __future__ import annotations

from typing import Any

import pandas as pd

from tamasha.prediction_service import PredictionService

_service: PredictionService | None = None


def _get_service() -> PredictionService:
    global _service
    if _service is None:
        _service = PredictionService()
        _service.load()
    return _service


def predict_rating(title: str, year: int | None = None, **kw: Any) -> dict[str, Any]:
    return _get_service().predict_rating(title, year, **kw)


def predict_boxoffice(title: str, year: int | None = None, **kw: Any) -> dict[str, Any]:
    return _get_service().predict_boxoffice(title, year, **kw)


def get_actor_info(name: str) -> dict[str, Any]:
    return _get_service().get_actor_info(name)


def get_model_info() -> dict[str, Any]:
    return _get_service().get_model_info()


def get_bankability_scores() -> pd.DataFrame:
    return _get_service().get_bankability_scores()


def get_chemistry_pairs() -> pd.DataFrame:
    return _get_service().get_chemistry_pairs()


def get_comparison_csv(task: str) -> pd.DataFrame | None:
    return _get_service().get_comparison_csv(task)
