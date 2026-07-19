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
    genre : list[str]
        List of genres.
    cast_size : int
        Number of cast members.
    budget_inr : float
        Budget in rupees.
    runtime_minutes : int
        Runtime in minutes.
    avg_bankability_score : float
        Average Bankability Score of the cast.
    """

    title: str = "Untitled"
    genre: list[str] = field(default_factory=list)
    cast_size: int = 5
    budget_inr: float = 50_000_000
    runtime_minutes: int = 150
    avg_bankability_score: float = 0.5


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


def simulate_scenarios(
    model: Any,
    profile: MovieProfile,
    feature_columns: list[str],
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
        festival_scenarios = [
            "Normal",
            "Diwali",
            "Eid",
            "Christmas",
            "Independence Day",
        ]

    results: list[ScenarioResult] = []

    for scenario in festival_scenarios:
        # Build a feature vector based on the scenario
        features = _build_scenario_features(profile, scenario, feature_columns)
        pred = model.predict(features.reshape(1, -1))[0]

        results.append(
            ScenarioResult(
                scenario_name=scenario,
                predicted_boxoffice=float(pred),
                festival_flag=scenario != "Normal",
                has_clash=False,  # No clash in simulation
            )
        )

    logger.info(
        "Scenario simulation for '%s': %d scenarios run.",
        profile.title,
        len(results),
    )
    return results


def _build_scenario_features(
    profile: MovieProfile,
    scenario: str,
    feature_columns: list[str],
) -> np.ndarray:
    """Build a feature vector for a given scenario.

    This is a simplified mock for the placeholder stage.  In
    production, it would one-hot encode genres, set festival flags,
    etc., to match the exact feature columns the model expects.

    Parameters
    ----------
    profile : MovieProfile
        Movie profile.
    scenario : str
        Scenario name.
    feature_columns : list[str]
        Expected feature column names.

    Returns
    -------
    np.ndarray
        Feature vector matching ``feature_columns``.
    """
    # Placeholder: create a zero vector with some known features
    n_features = len(feature_columns)
    vec = np.zeros(n_features, dtype=float)

    known_features = {
        "cast_size": profile.cast_size,
        "runtime_minutes": profile.runtime_minutes,
        "budget_inr": profile.budget_inr,
        "avg_bankability_score": profile.avg_bankability_score,
        "is_festival_release": 1.0 if scenario != "Normal" else 0.0,
        "has_clash": 0.0,
    }

    for col in feature_columns:
        if col in known_features:
            vec[feature_columns.index(col)] = known_features[col]

    return vec
