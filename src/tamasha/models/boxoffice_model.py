"""Box-office prediction module.

Wraps ``model_selection`` for the box-office task. Supports running
the comparison **twice**: once with baseline features, and once with
the Bankability Score added (after Stage 6).
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


def train_boxoffice_model(
    df: pd.DataFrame,
    boxoffice_column: Optional[str] = None,
    bankability_df: Optional[pd.DataFrame] = None,
    models: Optional[dict[str, Any]] = None,
    metric: str = "MAE",
    cv_folds: int = 5,
    save_dir: Optional[Union[str, Path]] = None,
    run_label: str = "boxoffice",
    tune: bool = False,
    tune_n_iter: int = 10,
) -> tuple[Any, pd.DataFrame]:
    """Run the box-office model comparison and save the winner.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, joined DataFrame.
    boxoffice_column : str, optional
        Target column name. Auto-detected if None.
    bankability_df : pd.DataFrame, optional
        Bankability Score per actor.  If provided, the average
        Bankability Score of the cast is added as a feature.
    models : dict[str, Any], optional
        Models to compare.  Defaults to all available.
    metric : str, default="MAE"
        Selection metric.
    cv_folds : int, default=5
        Number of CV folds.
    save_dir : str or Path, optional
        Directory to save outputs.
    run_label : str, default="boxoffice"
        Label for CSV naming.
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

    # Detect box-office column
    if boxoffice_column is None:
        candidates = (
            [c for c in df.columns if "collection_inr" in c.lower()]
            or [c for c in df.columns if "box_office_inr" in c.lower()]
            or [c for c in df.columns if "box_office" in c.lower()]
        )
        if not candidates:
            raise ValueError(
                "No box-office column found. Specify ``boxoffice_column``."
            )
        boxoffice_column = candidates[0]
        logger.info("Auto-detected box-office column: %s", boxoffice_column)

    # Build feature matrix
    X, _, y_boxoffice = build_feature_matrix(
        df,
        target_column_boxoffice=boxoffice_column,
    )

    # Add Bankability Score feature if available
    if bankability_df is not None and not bankability_df.empty:
        cast_col = [c for c in df.columns if c.lower() == "cast"]
        if cast_col:
            avg_bankability = _compute_cast_avg_bankability(
                df, cast_col[0], bankability_df
            )
            X["avg_bankability_score"] = avg_bankability
            logger.info("Added avg_bankability_score feature.")
        run_label = f"{run_label}_with_bankability"
    else:
        run_label = f"{run_label}_baseline"

    # Drop rows with missing target
    valid = y_boxoffice.notna()
    X, y_boxoffice = X[valid], y_boxoffice[valid]
    logger.info("Box-office model: %d samples with valid target.", len(y_boxoffice))

    if models is None:
        models = get_all_models()

    csv_name = f"model_comparison_{run_label}.csv"
    csv_path = save_dir.parent / "reports" / csv_name
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    comparison, best_name, best_model = train_and_compare(
        X=X,
        y=y_boxoffice,
        task_name=run_label,
        models=models,
        cv_folds=cv_folds,
        metric=metric,
        save_csv=str(csv_path),
        tune=tune,
        tune_n_iter=tune_n_iter,
    )

    # Save best model (with bankability version)
    model_name = "best_boxoffice_model.pkl"
    if "bankability" in run_label:
        model_name = "best_boxoffice_model.pkl"
    model_path = save_dir / model_name
    save_model(best_model, model_path)

    logger.info(
        "Box-office model training complete. Best: %s. Saved to %s",
        best_name,
        model_path,
    )

    return best_model, comparison


def _compute_cast_avg_bankability(
    df: pd.DataFrame,
    cast_column: str,
    bankability_df: pd.DataFrame,
) -> pd.Series:
    """Compute the average Bankability Score of each film's cast.

    Parameters
    ----------
    df : pd.DataFrame
        Movie DataFrame.
    cast_column : str
        Column with comma-separated cast.
    bankability_df : pd.DataFrame
        DataFrame with ``actor`` and ``bankability_score`` columns.

    Returns
    -------
    pd.Series
        Average bankability per movie.
    """
    bankability_map = dict(
        zip(
            bankability_df["actor"].str.lower().str.strip(),
            bankability_df["bankability_score"],
        )
    )

    def _avg_bankability(cast_str: str) -> float:
        actors = [a.strip().lower() for a in cast_str.split(",") if a.strip()]
        scores = [bankability_map.get(a, 0.0) for a in actors]
        return sum(scores) / len(scores) if scores else 0.0

    return df[cast_column].apply(_avg_bankability)
