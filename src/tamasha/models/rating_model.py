"""Rating prediction module.

Wraps ``model_selection`` for the rating task. Exports a convenience
``train_rating_model()`` that runs the full comparison, saves both
the comparison CSV and the winning model, and returns the estimator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from tamasha.config import settings
from tamasha.features.movie_features import build_feature_matrix
from tamasha.models.model_selection import (
    get_all_models,
    save_model,
    train_and_compare,
)

logger = logging.getLogger(__name__)


def train_rating_model(
    df: pd.DataFrame,
    rating_column: Optional[str] = None,
    models: Optional[dict[str, Any]] = None,
    metric: str = "MAE",
    cv_folds: int = 5,
    save_dir: Optional[Union[str, Path]] = None,
    tune: bool = False,
    tune_n_iter: int = 10,
) -> tuple[Any, pd.DataFrame]:
    """Run the full rating model comparison and save the winner.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, joined DataFrame.
    rating_column : str, optional
        Target column name. Auto-detected if None.
    models : dict[str, Any], optional
        Models to compare.  Defaults to all available.
    metric : str, default="MAE"
        Selection metric.
    cv_folds : int, default=5
        Number of CV folds.
    save_dir : str or Path, optional
        Directory to save outputs.  Defaults to ``settings.MODELS_DIR``
        for the model and ``settings.REPORTS_DIR`` for the CSV.
    tune : bool, default=False
        If True, run ``RandomizedSearchCV`` for models with search spaces
        defined in ``_TUNING_SPACES``.
    tune_n_iter : int, default=10
        Number of parameter settings sampled per tuned model.

    Returns
    -------
    tuple[Any, pd.DataFrame]
        ``(best_model, comparison_df)``.
    """
    if save_dir is None:
        save_dir = settings.MODELS_DIR
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Detect rating column
    if rating_column is None:
        candidates = [c for c in df.columns if "rating" in c.lower()]
        if not candidates:
            raise ValueError(
                "No rating column found in DataFrame. "
                "Please specify ``rating_column``."
            )
        rating_column = candidates[0]
        logger.info("Auto-detected rating column: %s", rating_column)

    # Build feature matrix
    X, y_rating, _ = build_feature_matrix(
        df,
        target_column_rating=rating_column,
    )

    # Drop rows with missing target
    valid = y_rating.notna()
    X, y_rating = X[valid], y_rating[valid]
    logger.info("Rating model: %d samples with valid target.", len(y_rating))

    if models is None:
        models = get_all_models()

    csv_path = save_dir.parent / "reports" / "model_comparison_rating.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    comparison, best_name, best_model = train_and_compare(
        X=X,
        y=y_rating,
        task_name="rating",
        models=models,
        cv_folds=cv_folds,
        metric=metric,
        save_csv=str(csv_path),
        tune=tune,
        tune_n_iter=tune_n_iter,
    )

    # Save best model
    model_path = save_dir / "best_rating_model.pkl"
    save_model(best_model, model_path)

    logger.info(
        "Rating model training complete. Best: %s. Saved to %s",
        best_name,
        model_path,
    )

    return best_model, comparison
