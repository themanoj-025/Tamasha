"""Train multiple candidate models, compare performance, auto-select best.

Selection rule (configurable via ``settings.MODEL_SELECTION_METRIC``):
- ``MAE``: lowest MAE wins (default)
- ``RMSE``: lowest RMSE wins
- ``R2``: highest RÂ² wins

All models are evaluated on the same k-fold split and same feature set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

from tamasha.config import settings

logger = logging.getLogger(__name__)

# Model registry

_MODEL_REGISTRY: dict[str, tuple[Any, dict[str, Any]]] = {
    "LinearRegression": (LinearRegression, {}),
    "Ridge": (Ridge, {"alpha": 1.0}),
    "Lasso": (Lasso, {"alpha": 0.01}),
    "DecisionTree": (DecisionTreeRegressor, {"max_depth": 10, "random_state": 42}),
    "RandomForest": (
        RandomForestRegressor,
        {"n_estimators": 200, "max_depth": 15, "n_jobs": -1, "random_state": 42},
    ),
    "GradientBoosting": (
        GradientBoostingRegressor,
        {"n_estimators": 200, "max_depth": 5, "random_state": 42},
    ),
}

# Optional heavy models â€” only imported if available
_EXTRA_MODEL_REGISTRY: dict[str, tuple[str, dict[str, Any]]] = {
    "XGBoost": (
        "xgboost.XGBRegressor",
        {"n_estimators": 200, "max_depth": 6, "random_state": 42, "verbosity": 0},
    ),
    "LightGBM": (
        "lightgbm.LGBMRegressor",
        {"n_estimators": 200, "max_depth": 6, "random_state": 42, "verbose": -1},
    ),
    "CatBoost": (
        "catboost.CatBoostRegressor",
        {"iterations": 200, "depth": 6, "random_state": 42, "verbose": 0},
    ),
}

# Hyperparameter search spaces for RandomizedSearchCV

_TUNING_SPACES: dict[str, dict[str, Any]] = {
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "min_samples_split": [2, 5, 10],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],  # bounded depth to prevent overfitting on small datasets
        "learning_rate": [0.05, 0.1],  # min 0.05 ensures convergence with 200-300 trees
        "num_leaves": [15, 31, 63],
    },
}


def _import_extra_model(import_path: str) -> Any:
    """Dynamically import an optional model class.

    Parameters
    ----------
    import_path : str
        Dotted path, e.g. ``"xgboost.XGBRegressor"``.

    Returns
    -------
    type or None
        The model class, or None if not installed.
    """
    try:
        parts = import_path.split(".")
        module = __import__(".".join(parts[:-1]), fromlist=[parts[-1]])
        return getattr(module, parts[-1])
    except ImportError:
        logger.warning("Optional model %s not installed. Skipping.", import_path)
        return None


def get_all_models() -> dict[str, Any]:
    """Return all available model instances keyed by name.

    Returns
    -------
    dict[str, sklearn.base.RegressorMixin]
        Mapping of display name â†’ instantiated model.
    """
    models: dict[str, Any] = {}
    for name, (cls, kwargs) in _MODEL_REGISTRY.items():
        try:
            models[name] = cls(**kwargs)
        except (ValueError, OSError) as exc:
            logger.warning("Failed to instantiate %s: %s", name, exc)

    for name, (import_path, kwargs) in _EXTRA_MODEL_REGISTRY.items():
        cls = _import_extra_model(import_path)
        if cls is not None:
            try:
                models[name] = cls(**kwargs)
            except (ValueError, OSError) as exc:
                logger.warning("Failed to instantiate %s: %s", name, exc)

    logger.info("Available models: %s", list(models.keys()))
    return models


# Training and comparison


def _get_metric_value(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    """Compute a single scalar metric.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth.
    y_pred : np.ndarray
        Predictions.
    metric : str
        One of ``"MAE"``, ``"RMSE"``, ``"R2"``.

    Returns
    -------
    float
    """
    if metric == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif metric == "RMSE":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    elif metric == "R2":
        return r2_score(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def sha256_of_file(path: Path) -> str:
    """Compute SHA-256 hash of a file.

    Parameters
    ----------
    path : Path
        File path to hash.

    Returns
    -------
    str
        Hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model_with_version(
    model: Any,
    task_name: str,
    metadata: dict[str, Any] | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Save a model with versioned metadata.

    Creates ``models/v{N}/{task_name}_model.pkl`` and a
    ``models/v{N}/metadata.json`` with training info.

    Parameters
    ----------
    model : Any
        Trained estimator.
    task_name : str
        e.g. ``"rating"`` or ``"boxoffice"``.
    metadata : dict, optional
        Extra metadata to include (CV scores, data hash, etc.).
    models_dir : Path, optional
        Root models directory.  Defaults to ``settings.MODELS_DIR``.

    Returns
    -------
    dict
        ``{"version": int, "path": Path, "metadata_path": Path}``
    """
    models_dir = models_dir or settings.MODELS_DIR

    # Find next version number
    version_dirs = sorted(models_dir.glob("v*"))
    next_version = 1
    if version_dirs:
        existing = [int(d.name[1:]) for d in version_dirs if d.name[1:].isdigit()]
        if existing:
            next_version = max(existing) + 1

    version_dir = models_dir / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / f"{task_name}_model.pkl"
    joblib.dump(model, model_path)

    # Compute SHA-256 hash of the model artifact for integrity verification
    model_hash = sha256_of_file(model_path)

    meta = {
        "version": next_version,
        "task": task_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "sha256": model_hash,
    }
    if metadata:
        meta.update(metadata)

    meta_path = version_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(
        "Model v%d for '%s' saved to %s",
        next_version,
        task_name,
        model_path,
    )
    return {"version": next_version, "path": model_path, "metadata_path": meta_path}


def save_model(model: Any, path: str | Path) -> Path:
    """Save a trained model to disk via ``joblib``.

    Parameters
    ----------
    model : Any
        Trained estimator.
    path : str or Path
        Destination path.

    Returns
    -------
    Path
        Absolute path where the model was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved to %s", path)
    return path


def load_model(path: str | Path) -> Any:
    """Load a trained model from disk.

    Parameters
    ----------
    path : str or Path
        Path to the ``.pkl`` / ``.joblib`` file.

    Returns
    -------
    Any
        Loaded estimator.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model

# Re-exports for backward compatibility
from tamasha.models.model_training import (  # noqa: F401
    compare_models_significance,
    train_and_compare,
    tune_model,
)
