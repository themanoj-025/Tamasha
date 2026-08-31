"""Prediction service — model loading and inference for ratings and box office."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tamasha.config import settings

logger = logging.getLogger(__name__)

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
        models_dir: Path | None = None,
        reports_dir: Path | None = None,
    ) -> None:
        self._models_dir: Path = models_dir or settings.MODELS_DIR
        self._reports_dir: Path = reports_dir or settings.REPORTS_DIR

        # All internal state starts as None / empty.
        self._rating_model: Any = None
        self._boxoffice_model: Any = None
        self._bankability_scores: pd.DataFrame = pd.DataFrame()
        self._chemistry_pairs: pd.DataFrame = pd.DataFrame()
        self._bankability_map: dict[str, float] = {}
        self._rating_comparison: pd.DataFrame | None = None
        self._boxoffice_baseline_comparison: pd.DataFrame | None = None
        self._boxoffice_bank_comparison: pd.DataFrame | None = None
        self._model_names: dict[str, str] = {}
        self._model_metrics: dict[str, dict[str, float]] = {}
        self._rating_feature_cols: list[str] = []
        self._boxoffice_feature_cols: list[str] = []
        self._director_encoder: Any = None
        self._loaded: bool = False
        self._load_lock: threading.Lock = threading.Lock()
        self._integrity_failures: list[dict[str, str]] = []

    # â”€â”€ public load â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def load(self) -> None:
        """Load all artifacts from disk. Safe to call multiple times.

        Uses double-checked locking for thread safety â€” two threads
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

    # â”€â”€ property: healthy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    # â”€â”€ private load helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                return  # Don't raise â€” log and mark as degraded
            logger.debug("Integrity verified for %s", model_path)
        except (ValueError, OSError) as exc:
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

    # â”€â”€ shared feature-vector builder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _build_prediction_vector(
        self,
        genres: list[str],
        cast: list[str],
        director: str,
        budget_inr: float,
        runtime_minutes: int,
        year: int,
        expected_cols: list[str],
        bankability_score: float | None = None,
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

    # â”€â”€ cast bankability helper â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

