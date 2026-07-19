"""Train multiple candidate models, compare performance, auto-select best.

Selection rule (configurable via ``settings.MODEL_SELECTION_METRIC``):
- ``MAE``: lowest MAE wins (default)
- ``RMSE``: lowest RMSE wins
- ``R2``: highest R² wins

All models are evaluated on the same k-fold split and same feature set.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tamasha.config import settings

logger = logging.getLogger(__name__)

# ── Model registry ────────────────────────────────────────────────────

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

# Optional heavy models — only imported if available
_EXTRA_MODEL_REGISTRY: dict[str, tuple[str, dict[str, Any]]] = {
    "XGBoost": ("xgboost.XGBRegressor", {"n_estimators": 200, "max_depth": 6, "random_state": 42, "verbosity": 0}),
    "LightGBM": ("lightgbm.LGBMRegressor", {"n_estimators": 200, "max_depth": 6, "random_state": 42, "verbose": -1}),
    "CatBoost": ("catboost.CatBoostRegressor", {"iterations": 200, "depth": 6, "random_state": 42, "verbose": 0}),
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
        Mapping of display name → instantiated model.
    """
    models: dict[str, Any] = {}
    for name, (cls, kwargs) in _MODEL_REGISTRY.items():
        try:
            models[name] = cls(**kwargs)
        except Exception as exc:
            logger.warning("Failed to instantiate %s: %s", name, exc)

    for name, (import_path, kwargs) in _EXTRA_MODEL_REGISTRY.items():
        cls = _import_extra_model(import_path)
        if cls is not None:
            try:
                models[name] = cls(**kwargs)
            except Exception as exc:
                logger.warning("Failed to instantiate %s: %s", name, exc)

    logger.info("Available models: %s", list(models.keys()))
    return models


# ── Training and comparison ──────────────────────────────────────────

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


def train_and_compare(
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    task_name: str = "model",
    models: Optional[dict[str, Any]] = None,
    cv_folds: int = 5,
    metric: str = "MAE",
    save_csv: Optional[str] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, str, Any]:
    """Train all candidate models under k-fold CV and return comparison.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : pd.Series or np.ndarray
        Target vector.
    task_name : str, default="model"
        Name for logging / saved CSV.
    models : dict[str, Any], optional
        Models to compare.  Defaults to :func:`get_all_models`.
    cv_folds : int, default=5
        Number of cross-validation folds.
    metric : str, default="MAE"
        Selection metric (``MAE``, ``RMSE``, or ``R2``).
    save_csv : str, optional
        Path to save comparison CSV.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, str, Any]
        ``(comparison_df, best_model_name, best_estimator)``.

        The best estimator is **refit on the full dataset**.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y).ravel()

    if models is None:
        models = get_all_models()

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results: list[dict[str, Any]] = []

    for name, model in models.items():
        logger.info("Training %s for %s ...", name, task_name)
        fold_metrics: list[dict[str, float]] = []
        start_time = time.perf_counter()

        for train_idx, val_idx in kf.split(X_arr):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            model_clone = (
                model.__class__(**model.get_params()) if hasattr(model, "get_params")
                else model.__class__()
            )
            model_clone.fit(X_train, y_train)
            y_pred = model_clone.predict(X_val)

            fold_metrics.append(
                {
                    "MAE": mean_absolute_error(y_val, y_pred),
                    "RMSE": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                    "R2": r2_score(y_val, y_pred),
                }
            )

        elapsed = time.perf_counter() - start_time

        avg = {
            k: np.mean([f[k] for f in fold_metrics])
            for k in fold_metrics[0]
        }
        std = {
            k: np.std([f[k] for f in fold_metrics])
            for k in fold_metrics[0]
        }

        results.append(
            {
                "model": name,
                "MAE": avg["MAE"],
                "MAE_std": std["MAE"],
                "RMSE": avg["RMSE"],
                "RMSE_std": std["RMSE"],
                "R2": avg["R2"],
                "R2_std": std["R2"],
                "training_time_s": round(elapsed, 2),
            }
        )

        logger.info(
            "%s — MAE=%.4f, RMSE=%.4f, R²=%.4f (%.1fs)",
            name, avg["MAE"], avg["RMSE"], avg["R2"], elapsed,
        )

    comparison = pd.DataFrame(results)

    # Sort by selection metric
    ascending = metric != "R2"
    comparison_sorted = comparison.sort_values(metric, ascending=ascending).reset_index(
        drop=True
    )
    best_name = comparison_sorted.iloc[0]["model"]

    logger.info(
        "Best model for %s: %s (%s=%.4f)",
        task_name,
        best_name,
        metric,
        comparison_sorted.iloc[0][metric],
    )

    # Save CSV
    if save_csv:
        comparison_sorted.to_csv(save_csv, index=False)
        logger.info("Comparison CSV saved to %s", save_csv)

    # Refit best model on full data
    best_model_cls = models[best_name].__class__
    best_model = best_model_cls(**models[best_name].get_params())
    best_model.fit(X_arr, y_arr)
    logger.info("Best model %s refit on full dataset.", best_name)

    return comparison_sorted, best_name, best_model


def save_model(model: Any, path: Union[str, Path]) -> Path:
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


def load_model(path: Union[str, Path]) -> Any:
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
