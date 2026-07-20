"""Shared prediction functions for the Streamlit dashboard and FastAPI.

Both call into this same module so they never drift into two separate,
inconsistent prediction paths.

Architecture
------------
- ``PredictionService`` class — thread-safe, loads all artifacts once,
  used by FastAPI via dependency injection.
- Module-level singleton + wrapper functions — used by the Streamlit
  dashboard (which manages its own lifecycle via ``st.cache_resource``).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from tamasha.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  PredictionService  (thread‑safe by construction — no mutable shared
#  state after load(); each call creates its own local temporaries)
# ═══════════════════════════════════════════════════════════════════════


class PredictionService:
    """Prediction service that loads all trained artifacts once.

    Parameters
    ----------
    models_dir : Path, optional
        Directory containing ``.pkl`` model files and ``*_features.json``.
        Defaults to ``settings.MODELS_DIR``.
    reports_dir : Path, optional
        Directory containing metadata CSVs (bankability, chemistry, etc.).
        Defaults to ``settings.REPORTS_DIR``.

    Notes
    -----
    This class is **not** safe to call before :meth:`load` has completed.
    After :meth:`load` the instance is immutable and thread-safe for
    concurrent ``predict_*`` / ``get_*`` calls.
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        reports_dir: Optional[Path] = None,
    ) -> None:
        self._models_dir: Path = models_dir or settings.MODELS_DIR
        self._reports_dir: Path = reports_dir or settings.REPORTS_DIR

        # All internal state starts as None / empty.
        self._rating_model: Any = None
        self._boxoffice_model: Any = None
        self._bankability_scores: pd.DataFrame = pd.DataFrame()
        self._chemistry_pairs: pd.DataFrame = pd.DataFrame()
        self._bankability_map: dict[str, float] = {}
        self._rating_comparison: Optional[pd.DataFrame] = None
        self._boxoffice_baseline_comparison: Optional[pd.DataFrame] = None
        self._boxoffice_bank_comparison: Optional[pd.DataFrame] = None
        self._model_names: dict[str, str] = {}
        self._model_metrics: dict[str, dict[str, float]] = {}
        self._rating_feature_cols: list[str] = []
        self._boxoffice_feature_cols: list[str] = []
        self._director_encoder: Any = None
        self._loaded: bool = False
        self._load_lock: threading.Lock = threading.Lock()
        self._integrity_failures: list[dict[str, str]] = []

    # ── public load ────────────────────────────────────────────────

    def load(self) -> None:
        """Load all artifacts from disk. Safe to call multiple times.

        Uses double-checked locking for thread safety — two threads
        calling ``load()`` simultaneously will not double-load.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:  # double-checked locking
                return
            self._load_rating_model()
            self._load_boxoffice_model()
            self._load_metadata()
            self._load_feature_cols()
            self._load_director_encoder()
            self._loaded = True
            total = sum(
                [
                    1 if self._rating_model else 0,
                    1 if self._boxoffice_model else 0,
                    1 if self._rating_feature_cols else 0,
                    1 if self._boxoffice_feature_cols else 0,
                ]
            )
            logger.info("PredictionService loaded (%d/4 core artifacts).", total)

    # ── property: healthy ──────────────────────────────────────────

    @property
    def healthy(self) -> bool:
        """``True`` when all required artifacts are present and pass integrity checks.

        Returns ``False`` when models are missing or integrity checks failed,
        so callers can degrade gracefully.
        """
        if not self._loaded:
            return False
        has_no_integrity_failures = len(self._integrity_failures) == 0
        return (
            self._rating_model is not None
            and self._boxoffice_model is not None
            and len(self._rating_feature_cols) > 0
            and len(self._boxoffice_feature_cols) > 0
            and has_no_integrity_failures
        )

    @property
    def integrity_failures(self) -> list[dict[str, str]]:
        """List of artifacts that failed integrity verification.

        Each entry has keys: ``artifact``, ``expected``, ``actual``.
        """
        return list(self._integrity_failures)

    # ── private load helpers ───────────────────────────────────────

    def _load_rating_model(self) -> None:
        path = self._models_dir / "best_rating_model.pkl"
        if path.exists():
            import joblib

            self._verify_model_integrity(path)
            self._rating_model = joblib.load(path)
            logger.info("Loaded rating model from %s", path)
        else:
            logger.warning("Rating model not found at %s (run `make train`)", path)

    def _load_boxoffice_model(self) -> None:
        path = self._models_dir / "best_boxoffice_model.pkl"
        if path.exists():
            import joblib

            self._verify_model_integrity(path)
            self._boxoffice_model = joblib.load(path)
            logger.info("Loaded box office model from %s", path)
        else:
            logger.warning("Box office model not found at %s (run `make train`)", path)

    def _verify_model_integrity(self, model_path: Path) -> None:
        """Verify SHA-256 hash of a model artifact against metadata.json.

        If metadata.json doesn't exist or has no sha256 field, the check
        is skipped (backward compatibility with pre-v2 model saves).
        """
        from tamasha.models.model_selection import sha256_of_file

        # Look for metadata.json in the parent directory
        metadata_path = model_path.parent / "metadata.json"
        if not metadata_path.exists():
            return  # No metadata to verify against

        try:
            import json

            metadata = json.loads(metadata_path.read_text())
            expected_hash = metadata.get("sha256")
            if not expected_hash:
                return  # No hash stored

            actual_hash = sha256_of_file(model_path)
            if actual_hash != expected_hash:
                logger.error(
                    "Model integrity check FAILED: %s\n  Expected: %s\n  Actual: %s",
                    model_path,
                    expected_hash,
                    actual_hash,
                )
                self._integrity_failures.append(
                    {"artifact": str(model_path), "expected": expected_hash, "actual": actual_hash}
                )
                return  # Don't raise — log and mark as degraded
            logger.debug("Integrity verified for %s", model_path)
        except Exception as exc:
            logger.warning("Integrity check skipped for %s: %s", model_path, exc)

    def _load_director_encoder(self) -> None:
        path = self._models_dir / "director_encoder.pkl"
        if path.exists():
            import joblib

            self._director_encoder = joblib.load(path)
            logger.info("Loaded director encoder from %s", path)
        else:
            logger.info("Director encoder not found at %s (director feature will be ignored)", path)

    def _load_metadata(self) -> None:
        # Bankability scores
        bank_path = self._reports_dir / "bankability_scores.csv"
        if bank_path.exists():
            self._bankability_scores = pd.read_csv(bank_path)
            self._bankability_map = dict(
                zip(
                    self._bankability_scores["actor"].str.lower().str.strip(),
                    self._bankability_scores["bankability_score"],
                )
            )
            logger.info("Loaded %d bankability scores", len(self._bankability_scores))
        else:
            self._bankability_scores = pd.DataFrame()
            logger.warning("Bankability scores not found at %s", bank_path)

        # Chemistry pairs
        chem_path = self._reports_dir / "chemistry_pairs.csv"
        if chem_path.exists():
            self._chemistry_pairs = pd.read_csv(chem_path)
            logger.info("Loaded %d chemistry pairs", len(self._chemistry_pairs))
        else:
            self._chemistry_pairs = pd.DataFrame()
            logger.warning("Chemistry pairs not found at %s", chem_path)

        # Model comparison CSVs
        for csv_name, key in [
            ("model_comparison_rating.csv", "rating"),
            ("model_comparison_boxoffice_baseline.csv", "boxoffice_baseline"),
            ("model_comparison_boxoffice_with_bankability.csv", "boxoffice_bank"),
        ]:
            csv_path = self._reports_dir / csv_name
            if csv_path.exists():
                comp = pd.read_csv(csv_path)
                best = comp.iloc[0]
                self._model_names[key] = str(best["model"])
                self._model_metrics[key] = {
                    "mae": float(best["MAE"]),
                    "rmse": float(best.get("RMSE", 0)),
                    "r2": float(best.get("R2", 0)),
                }
                if key == "rating":
                    self._rating_comparison = comp
                elif key == "boxoffice_baseline":
                    self._boxoffice_baseline_comparison = comp
                elif key == "boxoffice_bank":
                    self._boxoffice_bank_comparison = comp

    def _load_feature_cols(self) -> None:
        for task in ("rating", "boxoffice"):
            path = self._models_dir / f"{task}_features.json"
            if path.exists():
                cols = json.loads(path.read_text())
                if task == "rating":
                    self._rating_feature_cols = cols
                else:
                    self._boxoffice_feature_cols = cols
                logger.info("Loaded %d %s feature columns", len(cols), task)

    # ── shared feature-vector builder ──────────────────────────────

    def _build_prediction_vector(
        self,
        genres: list[str],
        cast: list[str],
        director: str,
        budget_inr: float,
        runtime_minutes: int,
        year: int,
        expected_cols: list[str],
        bankability_score: Optional[float] = None,
    ) -> np.ndarray:
        """Build a feature vector matching the training data columns.

        Parameters
        ----------
        genres : list[str]
            Genre names (e.g. ``["Drama", "Romance"]``).
        cast : list[str]
            Cast member names.
        director : str
            Director name.
        budget_inr : float
            Budget in rupees.
        runtime_minutes : int
            Runtime in minutes.
        year : int
            Release year.
        expected_cols : list[str]
            Expected feature column names (from ``*_features.json``).
        bankability_score : float, optional
            Average bankability score for the cast.

        Returns
        -------
        np.ndarray
            Row-vector of shape ``(1, len(expected_cols))``.
        """
        if not expected_cols:
            return np.array([])

        vec = pd.Series(0.0, index=expected_cols)

        # Genre features: genre_{name}
        for g in genres:
            col = f"genre_{g}"
            if col in vec.index:
                vec[col] = 1.0

        # Cast size
        if "cast_size" in vec.index:
            vec["cast_size"] = len(cast)

        # Director encoding
        if "director_encoded" in vec.index and self._director_encoder is not None:
            try:
                encoded = int(self._director_encoder.transform([director.strip()])[0])
                vec["director_encoded"] = encoded
            except (ValueError, AttributeError):
                vec["director_encoded"] = 0  # unknown director

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

    # ── cast bankability helper ────────────────────────────────────

    def _compute_cast_avg_bankability(self, cast_list: list[str]) -> dict[str, Any]:
        """Compute average Bankability Score for a cast.

        Returns dict with ``avg_score``, ``fallback_count``, and ``total_count``.
        """
        scores: list[float] = []
        fallback_count = 0
        for actor in cast_list:
            key = actor.strip().lower()
            score = self._bankability_map.get(key)
            if score is not None:
                scores.append(score)
            else:
                fallback_count += 1
                all_scores = list(self._bankability_map.values())
                scores.append(float(np.mean(all_scores)) if all_scores else 0.3)

        avg = float(np.mean(scores)) if scores else 0.0
        return {
            "avg_score": round(avg, 4),
            "fallback_count": fallback_count,
            "total_count": len(cast_list),
        }

    # ── predict_rating ─────────────────────────────────────────────

    def predict_rating(
        self,
        genres: list[str],
        cast: list[str],
        director: str = "Unknown Director",
        budget_inr: float = 0.0,
        runtime_minutes: int = 150,
        year: int = 2024,
    ) -> dict[str, Any]:
        """Predict a movie's IMDB rating.

        Returns
        -------
        dict
            ``{"predicted_rating": float | None, "model_name": str, "model_mae": float | None}``
        """
        if self._rating_model is None:
            return {"predicted_rating": None, "model_name": "No model trained", "model_mae": None}
        if not self._rating_feature_cols:
            return {
                "predicted_rating": None,
                "model_name": "No feature columns saved",
                "model_mae": None,
            }

        X_vec = self._build_prediction_vector(
            genres,
            cast,
            director,
            budget_inr,
            runtime_minutes,
            year,
            self._rating_feature_cols,
        )
        if X_vec.size == 0:
            return {"predicted_rating": None, "model_name": "Feature error", "model_mae": None}

        try:
            pred = float(self._rating_model.predict(X_vec)[0])
            pred = max(0.0, min(10.0, pred))
            model_name = self._model_names.get("rating", "GradientBoosting")
            model_mae = self._model_metrics.get("rating", {}).get("mae", 0)
            return {
                "predicted_rating": round(pred, 2),
                "model_name": model_name,
                "model_mae": model_mae,
            }
        except Exception as exc:
            logger.error("Rating prediction failed: %s", exc)
            return {"predicted_rating": None, "model_name": "Error", "model_mae": None}

    # ── predict_boxoffice ──────────────────────────────────────────

    def predict_boxoffice(
        self,
        genres: list[str],
        cast: list[str],
        director: str = "Unknown Director",
        budget_inr: float = 0.0,
        runtime_minutes: int = 150,
        year: int = 2024,
        release_window: str = "Normal",
    ) -> dict[str, Any]:
        """Predict a movie's box office collection (in ₹ Crore).

        Returns
        -------
        dict
            ``{"predicted_boxoffice_cr": float | None, "model_name": str,
                "model_mae": float | None, "bankability_info": dict,
                "scenarios": dict, "fallback_actors": bool}``
        """
        if self._boxoffice_model is None:
            return {
                "predicted_boxoffice_cr": None,
                "model_name": "No model trained",
                "model_mae": None,
            }
        if not self._boxoffice_feature_cols:
            return {
                "predicted_boxoffice_cr": None,
                "model_name": "No feature columns saved",
                "model_mae": None,
            }

        bank_info = self._compute_cast_avg_bankability(cast)

        X_vec = self._build_prediction_vector(
            genres,
            cast,
            director,
            budget_inr,
            runtime_minutes,
            year,
            self._boxoffice_feature_cols,
            bank_info["avg_score"],
        )
        if X_vec.size == 0:
            return {
                "predicted_boxoffice_cr": None,
                "model_name": "Feature error",
                "model_mae": None,
            }

        try:
            pred = float(self._boxoffice_model.predict(X_vec)[0])
            pred_cr = pred / 1e7
            pred_cr = max(0.0, pred_cr)

            model_name = self._model_names.get("boxoffice_bank", "XGBoost")
            model_mae_cr = self._model_metrics.get("boxoffice_bank", {}).get("mae", 0) / 1e7

            # Festival multipliers — documented as domain-expert priors
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

    # ── get_actor_info ─────────────────────────────────────────────

    def get_actor_info(self, name: str) -> dict[str, Any]:
        """Get Bankability Score and chemistry pairs for an actor."""
        key = name.strip().lower()

        score_row = None
        if not self._bankability_scores.empty:
            match = self._bankability_scores[
                self._bankability_scores["actor"].str.lower().str.strip() == key
            ]
            if not match.empty:
                score_row = match.iloc[0]

        chemistry: list[dict[str, Any]] = []
        if not self._chemistry_pairs.empty:
            mask = self._chemistry_pairs["actor_1"].str.lower().str.strip().isin(
                [key, name.lower()]
            ) | self._chemistry_pairs["actor_2"].str.lower().str.strip().isin([key, name.lower()])
            chem_matches = self._chemistry_pairs[mask]
            for _, row in chem_matches.iterrows():
                partner = (
                    row["actor_2"] if row["actor_1"].lower().strip() == key else row["actor_1"]
                )
                chemistry.append(
                    {
                        "actor": partner,
                        "joint_films": int(row["joint_films"]),
                        "chemistry_score": float(row["uplift"]),
                    }
                )

        return {
            "name": name.strip().title(),
            "bankability_score": float(score_row["bankability_score"])
            if score_row is not None
            else None,
            "film_count": int(score_row["film_count"]) if score_row is not None else 0,
            "type": str(score_row["type"]) if score_row is not None else "unknown",
            "top_chemistry_pairs": chemistry,
            "found": score_row is not None,
        }

    # ── get_model_info ─────────────────────────────────────────────

    def get_model_info(self) -> dict[str, Any]:
        """Get info about currently deployed models."""
        return {
            "rating_model": {
                "name": self._model_names.get("rating", "Not trained"),
                "algorithm": self._model_names.get("rating", "N/A"),
                "mae": self._model_metrics.get("rating", {}).get("mae"),
                "rmse": self._model_metrics.get("rating", {}).get("rmse"),
                "r2": self._model_metrics.get("rating", {}).get("r2"),
                "features_used": ["genre", "cast_size", "director", "runtime", "budget", "decade"],
            },
            "boxoffice_model": {
                "name": self._model_names.get("boxoffice_bank", "Not trained"),
                "algorithm": self._model_names.get("boxoffice_bank", "N/A"),
                "mae": self._model_metrics.get("boxoffice_bank", {}).get("mae"),
                "rmse": self._model_metrics.get("boxoffice_bank", {}).get("rmse"),
                "r2": self._model_metrics.get("boxoffice_bank", {}).get("r2"),
                "features_used": [
                    "genre",
                    "cast_size",
                    "director",
                    "runtime",
                    "budget",
                    "decade",
                    "avg_bankability_score",
                ],
            },
        }

    # ── data accessors (for dashboard tables / charts) ─────────────

    def get_bankability_scores(self) -> pd.DataFrame:
        """Return the full Bankability scores DataFrame."""
        return self._bankability_scores

    def get_chemistry_pairs(self) -> pd.DataFrame:
        """Return the full chemistry pairs DataFrame."""
        return self._chemistry_pairs

    def get_comparison_csv(self, task: str) -> Optional[pd.DataFrame]:
        """Return a model comparison DataFrame.

        Parameters
        ----------
        task : str
            One of ``"rating"``, ``"boxoffice_baseline"``, ``"boxoffice_bank"``.
        """
        return {
            "rating": self._rating_comparison,
            "boxoffice_baseline": self._boxoffice_baseline_comparison,
            "boxoffice_bank": self._boxoffice_bank_comparison,
        }.get(task)


# ═══════════════════════════════════════════════════════════════════════
#  Module‑level singleton + thin wrappers  (Streamlit dashboard
#  manages lifecycle via ``st.cache_resource``)
# ═══════════════════════════════════════════════════════════════════════

_service: Optional[PredictionService] = None


def _get_service() -> PredictionService:
    """Get-or-create the module-level PredictionService singleton."""
    global _service
    if _service is None:
        _service = PredictionService()
        _service.load()
    return _service


def predict_rating(
    genres: list[str],
    cast: list[str],
    director: str = "Unknown Director",
    budget_inr: float = 0.0,
    runtime_minutes: int = 150,
    year: int = 2024,
) -> dict[str, Any]:
    """Module-level wrapper — delegates to the global singleton."""
    return _get_service().predict_rating(genres, cast, director, budget_inr, runtime_minutes, year)


def predict_boxoffice(
    genres: list[str],
    cast: list[str],
    director: str = "Unknown Director",
    budget_inr: float = 0.0,
    runtime_minutes: int = 150,
    year: int = 2024,
    release_window: str = "Normal",
) -> dict[str, Any]:
    """Module-level wrapper — delegates to the global singleton."""
    return _get_service().predict_boxoffice(
        genres, cast, director, budget_inr, runtime_minutes, year, release_window
    )


def get_actor_info(name: str) -> dict[str, Any]:
    """Module-level wrapper — delegates to the global singleton."""
    return _get_service().get_actor_info(name)


def get_model_info() -> dict[str, Any]:
    """Module-level wrapper — delegates to the global singleton."""
    return _get_service().get_model_info()


def get_bankability_scores() -> pd.DataFrame:
    """Module-level wrapper — delegates to the global singleton."""
    return _get_service().get_bankability_scores()


def get_chemistry_pairs() -> pd.DataFrame:
    """Module-level wrapper — delegates to the global singleton."""
    return _get_service().get_chemistry_pairs()


def get_comparison_csv(task: str) -> Optional[pd.DataFrame]:
    """Module-level wrapper — delegates to the global singleton."""
    return _get_service().get_comparison_csv(task)
