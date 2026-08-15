"""Release-scenario simulator.

Given a hypothetical film's profile, predict box office under different
release-window scenarios using the winning box-office model with timing
features swapped.

This is explicitly labeled as a **scenario simulation**, not a
guaranteed forecast, in code docstrings and in the UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MovieProfile:
    """Hypothetical movie profile for scenario simulation.

    Parameters
    ----------
    title : str
        Movie name (for display).
    genres : list[str]
        List of genres.
    cast_size : int
        Number of cast members.
    budget_inr : float
        Budget in rupees.
    runtime_minutes : int
        Runtime in minutes.
    avg_bankability_score : float
        Average Bankability Score of the cast.
    director : str
        Director name.
    year : int
        Release year.
    """

    title: str = "Untitled"
    genres: list[str] = field(default_factory=list)
    cast_size: int = 5
    budget_inr: float = 50_000_000
    runtime_minutes: int = 150
    avg_bankability_score: float = 0.5
    director: str = "Unknown"
    year: int = 2024


@dataclass
class ScenarioResult:
    """Result of a single scenario simulation.

    Parameters
    ----------
    scenario_name : str
        Human-readable name (e.g., "Diwali Release").
    predicted_boxoffice : float
        Predicted box office in rupees.
    festival_flag : bool
        Whether this scenario is a festival release.
    has_clash : bool
        Whether this scenario has a major clash.
    """

    scenario_name: str
    predicted_boxoffice: float
    festival_flag: bool
    has_clash: bool


def _build_feature_vector_from_profile(
    profile: MovieProfile,
    feature_columns: list[str],
    bankability_map: dict[str, float],
    director_encoder: Any = None,
) -> np.ndarray:
    """Build a feature vector matching the training data columns.

    Uses the same encoding logic as ``PredictionService._build_prediction_vector``
    — single source of truth via ``pandas.Series`` index alignment.

    Parameters
    ----------
    profile : MovieProfile
        Movie profile.
    feature_columns : list[str]
        Expected feature column names (from ``*_features.json``).
    bankability_map : dict[str, float]
        Map of actor name → bankability score for fallback.
    director_encoder : LabelEncoder, optional
        Fitted director encoder from training.

    Returns
    -------
    np.ndarray
        Row-vector of shape ``(1, len(feature_columns))``.
    """
    vec = pd.Series(0.0, index=feature_columns)

    # Genre features: genre_{name}
    for g in profile.genres:
        col = f"genre_{g}"
        if col in vec.index:
            vec[col] = 1.0

    # Cast size
    if "cast_size" in vec.index:
        vec["cast_size"] = profile.cast_size

    # Director encoding
    if "director_encoded" in vec.index and director_encoder is not None:
        try:
            vec["director_encoded"] = int(director_encoder.transform([profile.director.strip()])[0])
        except (ValueError, AttributeError):
            vec["director_encoded"] = 0

    # Runtime
    if "runtime_minutes" in vec.index:
        vec["runtime_minutes"] = profile.runtime_minutes

    # Budget
    if "budget_inr" in vec.index:
        vec["budget_inr"] = profile.budget_inr

    # Decade
    decade = (profile.year // 10) * 10
    decade_col = f"decade_{decade}"
    if decade_col in vec.index:
        vec[decade_col] = 1.0

    # Bankability
    if "avg_bankability_score" in vec.index:
        vec["avg_bankability_score"] = profile.avg_bankability_score

    return vec.values.reshape(1, -1)


def simulate_scenarios(
    model: Any,
    profile: MovieProfile,
    feature_columns: list[str],
    bankability_map: Optional[dict[str, float]] = None,
    director_encoder: Any = None,
    festival_scenarios: Optional[list[str]] = None,
) -> list[ScenarioResult]:
    """Run a release-scenario simulation for a hypothetical movie.

    Parameters
    ----------
    model : Any
        Trained box-office model (must have a ``predict`` method).
    profile : MovieProfile
        Movie profile.
    feature_columns : list[str]
        Ordered list of feature names expected by the model.
    bankability_map : dict[str, float], optional
        Map of actor name → bankability score for fallback.
    director_encoder : LabelEncoder, optional
        Fitted director encoder.
    festival_scenarios : list[str], optional
        Festival windows to simulate.  Defaults to
        ``["Normal", "Diwali", "Eid", "Christmas", "Independence Day"]``.

    Returns
    -------
    list[ScenarioResult]
        Results for each scenario.

    Notes
    -----
    This is a **simulation**, not a forecast.  Results depend on
    the quality of the input model and the representativeness of the
    training data.  Always interpret with appropriate caveats.
    """
    if festival_scenarios is None:
        festival_scenarios = ["Normal", "Diwali", "Eid", "Christmas", "Independence Day"]

    bankability_map = bankability_map or {}

    results: list[ScenarioResult] = []

    for scenario in festival_scenarios:
        features = _build_feature_vector_from_profile(
            profile,
            feature_columns,
            bankability_map,
            director_encoder,
        )
        pred = float(model.predict(features)[0])

        results.append(
            ScenarioResult(
                scenario_name=scenario,
                predicted_boxoffice=pred,
                festival_flag=scenario != "Normal",
                has_clash=False,
            )
        )

    logger.info(
        "Scenario simulation for '%s': %d scenarios run.",
        profile.title,
        len(results),
    )
    return results
