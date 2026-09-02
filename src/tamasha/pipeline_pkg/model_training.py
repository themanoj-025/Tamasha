"""Model training — rating and box-office models (Steps 4, 7)."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from tamasha.config import settings
from tamasha.features.movie_features import build_feature_matrix, save_director_encoder
from tamasha.models.boxoffice_model import train_boxoffice_model
from tamasha.models.rating_model import train_rating_model

logger = logging.getLogger(__name__)


def train_rating(
    df_rating: pd.DataFrame,
) -> tuple[Any, pd.DataFrame, list[str]]:
    """Step 4: Train rating model comparison.

    Returns (best_model, comparison_df, feature_names).
    """
    logger.info("  Training 8 models on %d movies with ratings...", len(df_rating))

    best_rating, comparison_rating = train_rating_model(
        df_rating,
        rating_column="rating",
        tune=True,
        tune_n_iter=15,
    )

    best_name = comparison_rating.iloc[0]["model"]
    best_mae = comparison_rating.iloc[0]["MAE"]
    best_rmse = comparison_rating.iloc[0]["RMSE"]
    best_r2 = comparison_rating.iloc[0]["R2"]

    logger.info("  [RATING] Best model: %s", best_name)
    logger.info("  [RATING] MAE=%.4f | RMSE=%.4f | R²=%.4f", best_mae, best_rmse, best_r2)

    # Save feature column names
    X_train, _, _ = build_feature_matrix(df_rating, target_column_rating="rating")
    rating_features = X_train.columns.tolist()
    (settings.MODELS_DIR / "rating_features.json").write_text(json.dumps(rating_features))
    logger.info("  Saved %d rating feature column names", len(rating_features))

    # Save director LabelEncoder
    try:
        director_col = [c for c in df_rating.columns if c.lower() == "director"]
        if director_col:
            save_director_encoder(df_rating, rating_column=director_col[0])
        else:
            logger.warning("  No director column found; skipping director encoder save.")
    except (OSError, ValueError) as exc:
        logger.warning("  Director encoder save failed (non-blocking): %s", exc)

    return best_rating, comparison_rating, rating_features


def train_boxoffice(
    df_box_model: pd.DataFrame,
    box_target: str | None,
    bankability_scores: pd.DataFrame | None,
    yr_col: str | None,
) -> tuple[Any, Any, pd.DataFrame, pd.DataFrame, float, float]:
    """Step 7: Train box-office models (baseline + with bankability).

    Returns (best_baseline, best_with_bank, comp_baseline, comp_with_bank,
             baseline_mae, bank_mae).
    """
    # Rename suffixed columns
    col_map = {}
    for search, target in [
        ("genre", "genre"), ("cast", "cast"), ("director", "director"),
        ("duration_minutes", "duration_minutes"), ("budget_inr", "budget_inr"),
    ]:
        found = [
            c for c in df_box_model.columns
            if c.lower() == search or (c.lower().endswith("_left") and search in c.lower())
        ]
        if found:
            col_map[found[0]] = target

    year_candidates = [
        c for c in df_box_model.columns if "year" in c.lower() and c not in (yr_col or "year")
    ]
    if year_candidates:
        col_map[year_candidates[0]] = "year"

    rating_candidates = [
        c for c in df_box_model.columns if "rating" in c.lower() and c != "_match_score"
    ]
    if rating_candidates:
        col_map[rating_candidates[0]] = "rating"

    df_box_model = df_box_model.rename(columns=col_map)
    logger.info("  Box office model columns (renamed): %s", list(df_box_model.columns))

    # ── Step 7a: Baseline ──
    logger.info("  [STEP 7a] Baseline (no Bankability)")
    best_baseline, comp_baseline = train_boxoffice_model(
        df_box_model,
        boxoffice_column=box_target,
        bankability_df=None,
        run_label="boxoffice",
        tune=True,
        tune_n_iter=15,
    )
    baseline_mae = comp_baseline.iloc[0]["MAE"]
    logger.info("  [BASELINE] Best: %s (MAE=%.4f)", comp_baseline.iloc[0]["model"], baseline_mae)

    # Save box office feature columns
    X_box_feat, _, _ = build_feature_matrix(df_box_model, target_column_boxoffice=box_target)
    box_features = X_box_feat.columns.tolist() + ["avg_bankability_score"]
    (settings.MODELS_DIR / "boxoffice_features.json").write_text(json.dumps(box_features))
    logger.info("  Saved %d box office feature column names", len(box_features))

    # ── Step 7b: With Bankability ──
    logger.info("  [STEP 7b] With Bankability Score")
    best_with_bank, comp_with_bank = train_boxoffice_model(
        df_box_model,
        boxoffice_column=box_target,
        bankability_df=bankability_scores,
        run_label="boxoffice",
        tune=True,
        tune_n_iter=15,
    )
    bank_mae = comp_with_bank.iloc[0]["MAE"]
    logger.info("  [WITH BANKABILITY] Best: %s (MAE=%.4f)", comp_with_bank.iloc[0]["model"], bank_mae)

    mae_improvement = ((baseline_mae - bank_mae) / abs(baseline_mae) * 100) if baseline_mae != 0 else 0
    logger.info("  MAE Improvement from Bankability Score: %.1f%%", mae_improvement)
    logger.info("    Baseline MAE:      %.4f", baseline_mae)
    logger.info("    With Bankability:  %.4f", bank_mae)

    return best_baseline, best_with_bank, comp_baseline, comp_with_bank, baseline_mae, bank_mae
