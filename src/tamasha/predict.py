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


def predict_rating(**kw: Any) -> dict[str, Any]:
    result: dict[str, Any] = _get_service().predict_rating(**kw)
    return result


def predict_boxoffice(**kw: Any) -> dict[str, Any]:
    result: dict[str, Any] = _get_service().predict_boxoffice(**kw)
    return result


def get_actor_info(name: str) -> dict[str, Any]:
    result: dict[str, Any] = _get_service().get_actor_info(name)
    return result


def get_model_info() -> dict[str, Any]:
    result: dict[str, Any] = _get_service().get_model_info()
    return result


def get_bankability_scores() -> pd.DataFrame:
    result: pd.DataFrame = _get_service().get_bankability_scores()
    return result


def get_chemistry_pairs() -> pd.DataFrame:
    result: pd.DataFrame = _get_service().get_chemistry_pairs()
    return result


def get_comparison_csv(task: str) -> pd.DataFrame | None:
    result: pd.DataFrame | None = _get_service().get_comparison_csv(task)
    return result
