"""Shared prediction functions for the Streamlit dashboard and FastAPI.

Both call into this same module so they never drift into two separate,
inconsistent prediction paths.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from tamasha.config import settings
from tamasha.features.movie_features import build_feature_matrix
from tamasha.models.model_selection import load_model

logger = logging.getLogger(__name__)

# ── Lazy-loaded globals ──────────────────────────────────────────────

_RATING_MODEL: Any = None
_BOXOFFICE_MODEL: Any = None
_BANKABILITY_SCORES: Optional[pd.DataFrame] = None
_CHEMISTRY_PAIRS: Optional[pd.DataFrame] = None
_RATING_COMPARISON: Optional[pd.DataFrame] = None
_BOXOFFICE_BASELINE_COMPARISON: Optional[pd.DataFrame] = None
_BOXOFFICE_BANK_COMPARISON: Optional[pd.DataFrame] = None
_BANKABILITY_MAP: dict[str, float] = {}

_MODEL_NAMES: dict[str, str] = {}
_MODEL_METRICS: dict[str, dict[str, float]] = {}
_FEATURES_USED: dict[str, list[str]] = {}
_RATING_FEATURE_COLS: list[str] = []
_BOXOFFICE_FEATURE_COLS: list[str] = []


def _load_rating_model() -> Any:
    """Load the winning rating model (lazy, cached)."""
    global _RATING_MODEL
    if _RATING_MODEL is None:
        path = settings.MODELS_DIR / "best_rating_model.pkl"
        if path.exists():
            _RATING_MODEL = load_model(path)
            logger.info("Loaded rating model from %s", path)
        else:
            logger.warning("Rating model not found at %s", path)
    return _RATING_MODEL


def _load_boxoffice_model() -> Any:
    """Load the winning box office model (Bankability-enhanced, lazy, cached)."""
    global _BOXOFFICE_MODEL
    if _BOXOFFICE_MODEL is None:
        path = settings.MODELS_DIR / "best_boxoffice_model.pkl"
        if path.exists():
            _BOXOFFICE_MODEL = load_model(path)
            logger.info("Loaded box office model from %s", path)
        else:
            logger.warning("Box office model not found at %s", path)
    return _BOXOFFICE_MODEL


def _load_metadata() -> None:
    """Load all metadata CSVs into globals (lazy, cached)."""
    global _BANKABILITY_SCORES, _CHEMISTRY_PAIRS
    global _RATING_COMPARISON, _BOXOFFICE_BASELINE_COMPARISON, _BOXOFFICE_BANK_COMPARISON
    global _BANKABILITY_MAP, _MODEL_NAMES, _MODEL_METRICS, _FEATURES_USED

    if _BANKABILITY_SCORES is not None:
        return  # Already loaded

    # ── Bankability scores ──────────────────────────────────────
    bank_path = settings.REPORTS_DIR / "bankability_scores.csv"
    if bank_path.exists():
        _BANKABILITY_SCORES = pd.read_csv(bank_path)
        _BANKABILITY_MAP = dict(zip(
            _BANKABILITY_SCORES["actor"].str.lower().str.strip(),
            _BANKABILITY_SCORES["bankability_score"]
        ))
        logger.info("Loaded %d bankability scores", len(_BANKABILITY_SCORES))
    else:
        _BANKABILITY_SCORES = pd.DataFrame()
        logger.warning("Bankability scores not found at %s", bank_path)

    # ── Chemistry pairs ─────────────────────────────────────────
    chem_path = settings.REPORTS_DIR / "chemistry_pairs.csv"
    if chem_path.exists():
        _CHEMISTRY_PAIRS = pd.read_csv(chem_path)
        logger.info("Loaded %d chemistry pairs", len(_CHEMISTRY_PAIRS))
    else:
        _CHEMISTRY_PAIRS = pd.DataFrame()
        logger.warning("Chemistry pairs not found at %s", chem_path)

    # ── Model comparison CSVs ───────────────────────────────────
    for csv_name, key in [
        ("model_comparison_rating.csv", "rating"),
        ("model_comparison_boxoffice_baseline.csv", "boxoffice_baseline"),
        ("model_comparison_boxoffice_with_bankability.csv", "boxoffice_bank"),
    ]:
        csv_path = settings.REPORTS_DIR / csv_name
        if csv_path.exists():
            comp = pd.read_csv(csv_path)
            best = comp.iloc[0]
            _MODEL_NAMES[key] = best["model"]
            _MODEL_METRICS[key] = {
                "mae": float(best["MAE"]),
                "rmse": float(best.get("RMSE", 0)),
                "r2": float(best.get("R2", 0)),
            }
            # Store full dataframes for chart rendering
            if key == "rating":
                _RATING_COMPARISON = comp
            elif key == "boxoffice_baseline":
                _BOXOFFICE_BASELINE_COMPARISON = comp
            elif key == "boxoffice_bank":
                _BOXOFFICE_BANK_COMPARISON = comp

    # ── Features used ───────────────────────────────────────────
    _FEATURES_USED = {
        "rating": ["genre", "cast_size", "director", "runtime", "budget", "decade"],
        "boxoffice": ["genre", "cast_size", "director", "runtime", "budget", "decade", "avg_bankability_score"],
    }


# ── Public prediction functions ──────────────────────────────────────


def _load_feature_cols(task: str) -> list[str]:
    """Load expected feature column names saved during training."""
    global _RATING_FEATURE_COLS, _BOXOFFICE_FEATURE_COLS
    key = f"{task}_features"
    path = settings.MODELS_DIR / f"{task}_features.json"
    if path.exists():
        cols = json.loads(path.read_text())
        if task == "rating":
            _RATING_FEATURE_COLS = cols
        else:
            _BOXOFFICE_FEATURE_COLS = cols
        return cols
    logger.warning("Feature columns file not found: %s", path)
    return []


def _build_prediction_vector(
    genres: list[str],
    cast: list[str],
    director: str,
    budget_inr: float,
    runtime_minutes: int,
    year: int,
    expected_cols: list[str],
    bankability_score: Optional[float] = None,
) -> np.ndarray:
    """Build a feature vector matching the training data columns."""
    if not expected_cols:
        return np.array([])

    vec = pd.Series(0.0, index=expected_cols)

    # Genre features: genre_{name}
    for g in genres:
        col = f"genre_{g}"
        if col in expected_cols:
            vec[col] = 1.0

    # Cast size
    if "cast_size" in vec.index:
        vec["cast_size"] = len(cast)

    # Runtime
    if "runtime_minutes" in vec.index:
        vec["runtime_minutes"] = runtime_minutes

    # Budget
    if "budget_inr" in vec.index:
        vec["budget_inr"] = budget_inr

    # Decade features: decade_{year_decade}0
    decade = (year // 10) * 10
    decade_col = f"decade_{decade}"
    if decade_col in vec.index:
        vec[decade_col] = 1.0

    # Bankability (if applicable)
    if bankability_score is not None and "avg_bankability_score" in vec.index:
        vec["avg_bankability_score"] = bankability_score

    return vec.values.reshape(1, -1)


def predict_rating(
    genres: list[str],
    cast: list[str],
    director: str = "Unknown Director",
    budget_inr: float = 0.0,
    runtime_minutes: int = 150,
    year: int = 2024,
) -> dict[str, Any]:
    """Predict a movie's IMDB rating.

    Parameters
    ----------
    genres : list[str]
        List of genres (e.g., ["Drama", "Romance"]).
    cast : list[str]
        List of actor names.
    director : str, default="Unknown Director"
        Director name.
    budget_inr : float, default=0.0
        Budget in rupees.
    runtime_minutes : int, default=150
        Runtime in minutes.
    year : int, default=2024
        Release year.

    Returns
    -------
    dict
        ``{"predicted_rating": float, "model_name": str, "model_mae": float}``
    """
    model = _load_rating_model()
    if model is None:
        return {"predicted_rating": None, "model_name": "No model trained", "model_mae": None}

    _load_metadata()
    expected = _load_feature_cols("rating")

    if not expected:
        return {"predicted_rating": None, "model_name": "No feature columns saved", "model_mae": None}

    X_vec = _build_prediction_vector(genres, cast, director, budget_inr, runtime_minutes, year, expected)

    if X_vec.size == 0:
        return {"predicted_rating": None, "model_name": "Feature error", "model_mae": None}

    try:
        pred = float(model.predict(X_vec)[0])
        pred = max(0.0, min(10.0, pred))
        model_name = _MODEL_NAMES.get("rating", "GradientBoosting")
        model_mae = _MODEL_METRICS.get("rating", {}).get("mae", 0)
        return {"predicted_rating": round(pred, 2), "model_name": model_name, "model_mae": model_mae}
    except Exception as exc:
        logger.error("Rating prediction failed: %s", exc)
        return {"predicted_rating": None, "model_name": "Error", "model_mae": None}


def _compute_cast_avg_bankability(cast_list: list[str]) -> dict[str, Any]:
    """Compute average Bankability Score for a cast.

    Returns dict with avg_score, fallback_count, and total_count.
    """
    _load_metadata()
    scores = []
    fallback_count = 0
    for actor in cast_list:
        key = actor.strip().lower()
        score = _BANKABILITY_MAP.get(key)
        if score is not None:
            scores.append(score)
        else:
            fallback_count += 1
            # Use genre-average fallback
            all_scores = list(_BANKABILITY_MAP.values())
            scores.append(float(np.mean(all_scores)) if all_scores else 0.3)

    avg = float(np.mean(scores)) if scores else 0.0
    return {
        "avg_score": round(avg, 4),
        "fallback_count": fallback_count,
        "total_count": len(cast_list),
    }


def predict_boxoffice(
    genres: list[str],
    cast: list[str],
    director: str = "Unknown Director",
    budget_inr: float = 0.0,
    runtime_minutes: int = 150,
    year: int = 2024,
    release_window: str = "Normal",
) -> dict[str, Any]:
    """Predict a movie's box office collection.

    Uses the Bankability-enhanced model. Returns prediction in crores.

    Parameters
    ----------
    Same as predict_rating plus:
    release_window : str, default="Normal"
        One of "Normal", "Diwali", "Eid", "Christmas", etc.

    Returns
    -------
    dict
        ``{"predicted_boxoffice_cr": float, "model_name": str, "model_mae": float,
            "bankability_info": dict, "scenarios": dict}``
    """
    model = _load_boxoffice_model()
    if model is None:
        return {"predicted_boxoffice_cr": None, "model_name": "No model trained", "model_mae": None}

    _load_metadata()
    expected = _load_feature_cols("boxoffice")

    if not expected:
        return {"predicted_boxoffice_cr": None, "model_name": "No feature columns saved", "model_mae": None}

    # Compute cast bankability
    bank_info = _compute_cast_avg_bankability(cast)

    X_vec = _build_prediction_vector(genres, cast, director, budget_inr, runtime_minutes, year, expected, bank_info["avg_score"])

    if X_vec.size == 0:
        return {"predicted_boxoffice_cr": None, "model_name": "Feature error", "model_mae": None}

    try:
        pred = float(model.predict(X_vec)[0])
        pred_cr = pred / 1e7
        pred_cr = max(0.0, pred_cr)

        model_name = _MODEL_NAMES.get("boxoffice_bank", "XGBoost")
        model_mae_cr = _MODEL_METRICS.get("boxoffice_bank", {}).get("mae", 0) / 1e7

        festival_multipliers = {
            "Normal": 1.0,
            "Diwali": 1.25,
            "Eid": 1.18,
            "Christmas": 1.12,
            "Independence Day": 1.08,
            "Republic Day": 1.05,
            "New Year": 1.10,
        }
        base_pred = pred_cr

        scenarios = {}
        for scenario, mult in festival_multipliers.items():
            scenarios[scenario] = round(base_pred * mult, 1)

        return {
            "predicted_boxoffice_cr": round(base_pred, 1),
            "model_name": model_name,
            "model_mae": round(model_mae_cr, 1),
            "bankability_info": bank_info,
            "scenarios": scenarios,
            "fallback_actors": bank_info["fallback_count"] > 0,
        }
    except Exception as exc:
        logger.error("Box office prediction failed: %s", exc)
        return {"predicted_boxoffice_cr": None, "model_name": "Error", "model_mae": None}


def get_actor_info(name: str) -> dict[str, Any]:
    """Get Bankability Score and chemistry pairs for an actor."""
    _load_metadata()

    key = name.strip().lower()
    score_row = None
    if _BANKABILITY_SCORES is not None and not _BANKABILITY_SCORES.empty:
        match = _BANKABILITY_SCORES[_BANKABILITY_SCORES["actor"].str.lower().str.strip() == key]
        if not match.empty:
            score_row = match.iloc[0]

    # Find chemistry pairs involving this actor
    chemistry = []
    if _CHEMISTRY_PAIRS is not None and not _CHEMISTRY_PAIRS.empty:
        mask = (
            _CHEMISTRY_PAIRS["actor_1"].str.lower().str.strip().isin([key, name.lower()])
            | _CHEMISTRY_PAIRS["actor_2"].str.lower().str.strip().isin([key, name.lower()])
        )
        chem_matches = _CHEMISTRY_PAIRS[mask]
        for _, row in chem_matches.iterrows():
            partner = row["actor_2"] if row["actor_1"].lower().strip() == key else row["actor_1"]
            chemistry.append({
                "actor": partner,
                "joint_films": int(row["joint_films"]),
                "chemistry_score": float(row["uplift"]),
            })

    return {
        "name": name.strip().title(),
        "bankability_score": float(score_row["bankability_score"]) if score_row is not None else None,
        "film_count": int(score_row["film_count"]) if score_row is not None else 0,
        "type": str(score_row["type"]) if score_row is not None else "unknown",
        "top_chemistry_pairs": chemistry,
        "found": score_row is not None,
    }


def get_model_info() -> dict[str, Any]:
    """Get info about currently deployed models."""
    _load_metadata()
    return {
        "rating_model": {
            "name": _MODEL_NAMES.get("rating", "Not trained"),
            "algorithm": _MODEL_NAMES.get("rating", "N/A"),
            "mae": _MODEL_METRICS.get("rating", {}).get("mae"),
            "rmse": _MODEL_METRICS.get("rating", {}).get("rmse"),
            "r2": _MODEL_METRICS.get("rating", {}).get("r2"),
            "features_used": _FEATURES_USED.get("rating", []),
        },
        "boxoffice_model": {
            "name": _MODEL_NAMES.get("boxoffice_bank", "Not trained"),
            "algorithm": _MODEL_NAMES.get("boxoffice_bank", "N/A"),
            "mae": _MODEL_METRICS.get("boxoffice_bank", {}).get("mae"),
            "rmse": _MODEL_METRICS.get("boxoffice_bank", {}).get("rmse"),
            "r2": _MODEL_METRICS.get("boxoffice_bank", {}).get("r2"),
            "features_used": _FEATURES_USED.get("boxoffice", []),
        },
    }


def get_bankability_scores() -> pd.DataFrame:
    """Return the full Bankability scores DataFrame."""
    _load_metadata()
    return _BANKABILITY_SCORES if _BANKABILITY_SCORES is not None else pd.DataFrame()


def get_chemistry_pairs() -> pd.DataFrame:
    """Return the full chemistry pairs DataFrame."""
    _load_metadata()
    return _CHEMISTRY_PAIRS if _CHEMISTRY_PAIRS is not None else pd.DataFrame()


def get_comparison_csv(task: str) -> Optional[pd.DataFrame]:
    """Return a model comparison DataFrame.

    Parameters
    ----------
    task : str
        One of ``"rating"``, ``"boxoffice_baseline"``, ``"boxoffice_bank"``.
    """
    _load_metadata()
    return {
        "rating": _RATING_COMPARISON,
        "boxoffice_baseline": _BOXOFFICE_BASELINE_COMPARISON,
        "boxoffice_bank": _BOXOFFICE_BANK_COMPARISON,
    }.get(task)
