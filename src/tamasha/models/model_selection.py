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
from datetime import datetime
from typing import Any, Optional, Union

import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict, RandomizedSearchCV
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

# ── Hyperparameter search spaces for RandomizedSearchCV ──────────────

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
        "max_depth": [4, 6, 8, -1],
        "learning_rate": [0.01, 0.05, 0.1],
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
    tune: bool = False,
    tune_n_iter: int = 10,
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
    tune : bool, default=False
        If True, run ``RandomizedSearchCV`` for models with a defined
        search space (see ``_TUNING_SPACES``) **before** the k-fold
        comparison. The tuned estimator is used in the comparison.
    tune_n_iter : int, default=10
        Number of parameter settings sampled per tuned model.

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

    # ── Optional hyperparameter tuning ──────────────────────────────
    tuned_models: dict[str, Any] = {}
    tuned_params_log: dict[str, dict[str, Any]] = {}

    if tune:
        logger.info("=" * 60)
        logger.info("  HYPERPARAMETER TUNING PHASE (%s)", task_name)
        logger.info("=" * 60)
        for name in models:
            if name in _TUNING_SPACES:
                logger.info("  Tuning %s (n_iter=%d) ...", name, tune_n_iter)
                best_est, best_params = tune_model(
                    name, X_arr, y_arr,
                    cv_folds=cv_folds,
                    n_iter=tune_n_iter,
                    random_state=random_state,
                )
                if best_est is not None:
                    tuned_models[name] = best_est
                    tuned_params_log[name] = best_params
                    logger.info("    ✓ Tuned %s: %s", name, best_params)
                else:
                    tuned_models[name] = models[name]
                    logger.info("    — Using default params for %s (tuning skipped/failed)", name)
            else:
                tuned_models[name] = models[name]
                logger.info("    — No tuning space for %s, using defaults", name)

        # Use tuned models for comparison
        compare_models = tuned_models
    else:
        compare_models = models

    # ── k-fold CV comparison ────────────────────────────────────────
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results: list[dict[str, Any]] = []

    for name, model in compare_models.items():
        if tune and name in tuned_params_log:
            logger.info("Evaluating TUNED %s for %s ...", name, task_name)
        else:
            logger.info("Evaluating %s for %s ...", name, task_name)

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
                "tuned": tune and name in tuned_params_log,
                "MAE": avg["MAE"],
                "MAE_std": std["MAE"],
                "RMSE": avg["RMSE"],
                "RMSE_std": std["RMSE"],
                "R2": avg["R2"],
                "R2_std": std["R2"],
                "training_time_s": round(elapsed, 2),
            }
        )

        tuned_tag = " (TUNED)" if tune and name in tuned_params_log else ""
        logger.info(
            "%s%s — MAE=%.4f, RMSE=%.4f, R²=%.4f (%.1fs)",
            name, tuned_tag, avg["MAE"], avg["RMSE"], avg["R2"], elapsed,
        )

    # ── Significance test between top 2 (out-of-fold predictions) ──
    if len(results) >= 2:
        sorted_idx = np.argsort([r["MAE"] for r in results])
        top2_names = [results[i]["model"] for i in sorted_idx[:2]]
        top2_model_instances = [compare_models[name] for name in top2_names]

        # Use cross_val_predict for genuine out-of-fold predictions
        try:
            oof_preds = []
            for model_obj in top2_model_instances:
                # Create a fresh clone for cross_val_predict
                m = model_obj.__class__(**model_obj.get_params()) if hasattr(model_obj, "get_params") else model_obj.__class__()
                preds = cross_val_predict(
                    m, X_arr, y_arr,
                    cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                    n_jobs=1,
                )
                oof_preds.append(preds)

            sig_result = compare_models_significance(
                y_arr, oof_preds[0], oof_preds[1],
                name_a=top2_names[0], name_b=top2_names[1],
            )

            logger.info("")
            logger.info("  Significance test (out-of-fold): %s vs %s", top2_names[0], top2_names[1])
            logger.info("    OOF MAE %s: %.4f", top2_names[0], sig_result["mae_a"])
            logger.info("    OOF MAE %s: %.4f", top2_names[1], sig_result["mae_b"])
            logger.info("    p-value: %.4f", sig_result["p_value"])
            if sig_result["significant"]:
                logger.info("    → %s is SIGNIFICANTLY better (p < 0.05)", sig_result["better_model"])
            else:
                logger.info("    → Difference is NOT statistically significant (p >= 0.05)")
            logger.info("")
        except Exception as exc:
            logger.warning("  Significance test failed (non-blocking): %s", exc)

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
    if tune and best_name in tuned_params_log:
        # Best model is a tuned estimator — use it directly
        best_model = compare_models[best_name]
        best_model.fit(X_arr, y_arr)
        logger.info("Best model %s (tuned) refit on full dataset.", best_name)
    else:
        best_model_cls = compare_models[best_name].__class__
        best_model = best_model_cls(**compare_models[best_name].get_params())
        best_model.fit(X_arr, y_arr)
        logger.info("Best model %s refit on full dataset.", best_name)

    return comparison_sorted, best_name, best_model


# ── Hyperparameter tuning with RandomizedSearchCV ────────────────────

def tune_model(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    n_iter: int = 10,
    random_state: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Run ``RandomizedSearchCV`` for a given model if a search space exists.

    Parameters
    ----------
    name : str
        Model name (must be in ``_TUNING_SPACES``).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    cv_folds : int, default=5
        Number of CV folds for tuning.
    n_iter : int, default=10
        Number of parameter settings sampled.
    random_state : int, default=42
        Random state.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        ``(best_estimator, best_params)``.  If no search space is defined
        for this model, returns ``(None, {})``.
    """
    if name not in _TUNING_SPACES:
        return None, {}

    # Get the model class and its default params
    if name in _MODEL_REGISTRY:
        cls, default_kwargs = _MODEL_REGISTRY[name]
    elif name in _EXTRA_MODEL_REGISTRY:
        import_path, default_kwargs = _EXTRA_MODEL_REGISTRY[name]
        cls = _import_extra_model(import_path)
        if cls is None:
            return None, {}
    else:
        return None, {}

    try:
        model = cls(**default_kwargs)
        param_dist = _TUNING_SPACES[name]

        search = RandomizedSearchCV(
            model, param_dist, n_iter=n_iter,
            cv=cv_folds, scoring="neg_mean_absolute_error",
            n_jobs=-1, random_state=random_state,
        )
        search.fit(X, y)

        logger.info(
            "Tuned %s: best MAE=%.4f (params: %s)",
            name, -search.best_score_, search.best_params_,
        )
        return search.best_estimator_, search.best_params_
    except Exception as exc:
        logger.warning("Tuning failed for %s: %s", name, exc)
        return None, {}


# ── Statistical significance test ─────────────────────────────────────

def compare_models_significance(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    name_a: str = "Model A",
    name_b: str = "Model B",
) -> dict[str, Any]:
    """Run a Wilcoxon signed-rank test between two models' absolute errors.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth.
    preds_a : np.ndarray
        Predictions from model A.
    preds_b : np.ndarray
        Predictions from model B.
    name_a : str, default="Model A"
        Display name for model A.
    name_b : str, default="Model B"
        Display name for model B.

    Returns
    -------
    dict
        ``{"mae_a": float, "mae_b": float, "p_value": float,
           "significant": bool, "better_model": str}``
    """
    errors_a = np.abs(y_true - preds_a)
    errors_b = np.abs(y_true - preds_b)

    mae_a = float(np.mean(errors_a))
    mae_b = float(np.mean(errors_b))

    # Wilcoxon signed-rank test (paired, two-sided)
    stat, p_value = stats.wilcoxon(errors_a, errors_b)

    significant = p_value < 0.05
    better = name_a if mae_a < mae_b else name_b

    logger.info(
        "Significance test: %s MAE=%.4f vs %s MAE=%.4f, "
        "p=%.4f, significant=%s, winner=%s",
        name_a, mae_a, name_b, mae_b,
        p_value, significant, better,
    )

    return {
        "mae_a": mae_a,
        "mae_b": mae_b,
        "p_value": float(p_value),
        "significant": bool(significant),
        "better_model": better,
    }


# ── Model versioning ──────────────────────────────────────────────────

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
        "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "sha256": model_hash,
    }
    if metadata:
        meta.update(metadata)

    meta_path = version_dir / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info(
        "Model v%d for '%s' saved to %s", next_version, task_name, model_path,
    )
    return {"version": next_version, "path": model_path, "metadata_path": meta_path}


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
