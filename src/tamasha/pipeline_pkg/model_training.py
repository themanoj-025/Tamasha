"""Training pipeline — end-to-end model training and selection.

Called by ``make train``.

Flow:
1. Load all three raw datasets (IMDb India, Box Office, year-bridge)
2. Two-step fuzzy join: Box Office → year-bridge → IMDb
3. Clean both the rating dataset (IMDb alone) and the box-office dataset (joined)
4. Rating model comparison (8 models, 5-fold CV, auto-select by MAE)
5. Bankability Scores + Chemistry Pairs (from joined dataset)
6. Box-office model comparison: baseline (no Bankability)
7. Box-office model comparison: with Bankability Score
8. Plot sentiment analysis (genre-conditional, if plot column available)
9. Release timing analysis (festival/clash features, if dates available)
10. Save all comparison CSVs, winning models, and report CSVs
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path before importing tamasha.*
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from tamasha.config import settings
from tamasha.data.enrichment import enrich_dataset
from tamasha.data.joining import fuzzy_join_datasets, generate_join_quality_report
from tamasha.data.loaders import load_bollywood_boxoffice, load_imdb_india
from tamasha.features.movie_features import build_feature_matrix
from tamasha.models.boxoffice_model import (
    _compute_cast_avg_bankability,
    train_boxoffice_model,
)
from tamasha.models.model_selection import get_all_models
from tamasha.models.rating_model import train_rating_model
from tamasha.network.bankability_score import compute_bankability_scores
from tamasha.network.chemistry_pairs import detect_chemistry_pairs
from tamasha.nlp.plot_sentiment import (
    genre_conditional_correlation,
    score_plot_sentiment,
)

logger = logging.getLogger(__name__)




"""
model_training.py — Steps 4-6: Rating model, sentiment, bankability.
"""

    # =====================================================================
    # STEP 4: RATING MODEL COMPARISON (Stage 3)
    # =====================================================================
    _print_separator("STEP 4: Rating Model Comparison (Stage 3)")

    logger.info("  Training 8 models on %d movies with ratings...", len(df_rating))

    best_rating, comparison_rating = train_rating_model(
        df_rating,
        rating_column="rating",
        tune=True,
        tune_n_iter=15,
    )

    best_rating_name = comparison_rating.iloc[0]["model"]
    best_rating_mae = comparison_rating.iloc[0]["MAE"]
    best_rating_rmse = comparison_rating.iloc[0]["RMSE"]
    best_rating_r2 = comparison_rating.iloc[0]["R2"]

    logger.info("  [RATING] Best model: %s", best_rating_name)
    logger.info(
        "  [RATING] MAE=%.4f | RMSE=%.4f | R²=%.4f",
        best_rating_mae,
        best_rating_rmse,
        best_rating_r2,
    )
    logger.info("  [RATING] Full comparison saved to reports/model_comparison_rating.csv")
    logger.info("  [RATING] Model saved to models/best_rating_model.pkl")

    # Save expected feature column names for inference
    X_rating_train, _, _ = build_feature_matrix(df_rating, target_column_rating="rating")
    rating_features = X_rating_train.columns.tolist()
    (settings.MODELS_DIR / "rating_features.json").write_text(json.dumps(rating_features))
    logger.info("  Saved %d rating feature column names", len(rating_features))

    # Save director LabelEncoder for inference
    from tamasha.features.movie_features import save_director_encoder

    try:
        director_col = [c for c in df_rating.columns if c.lower() == "director"]
        if director_col:
            save_director_encoder(df_rating, director_column=director_col[0])
        else:
            logger.warning("  No director column found; skipping director encoder save.")
    except (OSError, ValueError) as exc:
        logger.warning("  Director encoder save failed (non-blocking): %s", exc)

    # =====================================================================
    # STEP 5: PLOT SENTIMENT (Stage 5) — using TMDb-enriched plot summaries
    # =====================================================================
    _print_separator("STEP 5: Plot Sentiment Analysis (Stage 5)")

    # Use the TMDb-enriched plot_summary column from the box office dataset
    plot_col = [c for c in df_box_clean.columns if "plot_summary" in c.lower()]
    if plot_col:
        has_plot = df_box_clean["plot_summary"].str.strip().astype(bool).sum()
        logger.info(
            "  Found plot column: '%s' (%d movies with plot text out of %d)",
            plot_col[0],
            has_plot,
            len(df_box_clean),
        )

        if has_plot >= 20:
            # Limit to movies with plot text
            df_box_plot = df_box_clean[df_box_clean["plot_summary"].str.strip().astype(bool)].copy()
            sentiment_df = score_plot_sentiment(df_box_plot, plot_column=plot_col[0])

            # Genre-conditional correlation: does tone correlate with box office WITHIN each genre?
            logger.info("  Computing genre-conditional correlations with box office...")
            # Auto-detect the collection column (may be suffixed after join)
            box_col_targets = [
                c for c in df_box_plot.columns if "worldwide_collection" in c.lower()
            ]
            target_col = box_col_targets[0] if box_col_targets else None
            genre_corr = (
                genre_conditional_correlation(
                    df_box_plot,
                    sentiment_df,
                    target_column=target_col,
                    genre_column="genre",
                )
                if target_col
                else pd.DataFrame()
            )
            if len(genre_corr) > 0:
                genre_corr.to_csv(settings.REPORTS_DIR / "genre_tone_correlation.csv", index=False)
                logger.info("  Genre-tone correlations saved. Top findings:")
                for _, row in genre_corr.iterrows():
                    logger.info(
                        "    %s: corr=%.4f (n=%d movies)",
                        row["genre"],
                        row["correlation"],
                        row["n_movies"],
                    )
            else:
                logger.info(
                    "  No genre conditional correlations found (insufficient samples per genre)."
                )

            # Also check correlation with IMDB rating (if available)
            rating_col = [
                c for c in df_box_plot.columns if "rating" in c.lower() and c != "_match_score"
            ]
            if rating_col:
                genre_corr_rating = genre_conditional_correlation(
                    df_box_plot,
                    sentiment_df,
                    target_column=rating_col[0],
                    genre_column="genre",
                )
                if len(genre_corr_rating) > 0:
                    genre_corr_rating.to_csv(
                        settings.REPORTS_DIR / "genre_tone_correlation_rating.csv", index=False
                    )
                    logger.info("  Genre-tone vs RATING correlations also saved.")
        else:
            logger.info(
                "  Only %d movies have plot text — insufficient for genre-conditional analysis.",
                has_plot,
            )
            logger.info(
                "  (Need at least 20 movies with plot summaries to compute meaningful correlations per genre.)"
            )
    else:
        logger.info("  No plot_summary column found. Skipping Stage 5.")

    # =====================================================================
    # STEP 6: BANKABILITY SCORES & CHEMISTRY PAIRS (Stage 6)
    # =====================================================================
    _print_separator("STEP 6: Bankability Scores & Chemistry Pairs (Stage 6)")

    # Use joined dataset which has box office data for weighted scoring
    # Find the right columns for rating and collection
    box_rating_col = [
        c for c in df_box_clean.columns if "rating" in c.lower() and c != "_match_score"
    ]
    box_collection_col = [
        c
        for c in df_box_clean.columns
        if "worldwide_collection" in c.lower() or "collection_inr" in c.lower()
    ]
    box_cast_col = [c for c in df_box_clean.columns if c.lower() == "cast" or "cast" in c.lower()]
    box_dir_col = [c for c in df_box_clean.columns if "director" in c.lower()]
    box_year_col = [c for c in df_box_clean.columns if "year" in c.lower()]

    logger.info("  Columns for Bankability:")
    logger.info("    Rating: %s", box_rating_col[0] if box_rating_col else "NONE")
    logger.info("    Collection: %s", box_collection_col[0] if box_collection_col else "NONE")
    logger.info("    Cast: %s", box_cast_col[0] if box_cast_col else "NONE")
    logger.info("    Director: %s", box_dir_col[0] if box_dir_col else "NONE")

    # Use descriptive column names for bankability computation
    bankability_scores = compute_bankability_scores(
        df_box_clean,
        cast_column=box_cast_col[0] if box_cast_col else "cast",
        director_column=box_dir_col[0] if box_dir_col else "director",
        year_column=box_year_col[0] if box_year_col else "year",
        rating_column=box_rating_col[0] if box_rating_col else None,
        boxoffice_column=box_collection_col[0] if box_collection_col else None,
    )

    logger.info("  Bankability scores computed for %d individuals.", len(bankability_scores))
    logger.info("  Top 5:")
    for _, row in bankability_scores.head(5).iterrows():
        logger.info(
            "    %s [%s]: score=%.4f (%d films)",
            row["actor"],
            row["type"],
            row["bankability_score"],
            row["film_count"],
        )

    # Save Bankability scores
    bankability_scores.to_csv(settings.REPORTS_DIR / "bankability_scores.csv", index=False)

    # Detect chemistry pairs
    chemistry_pairs = detect_chemistry_pairs(
        df_box_clean,
        cast_column=box_cast_col[0] if box_cast_col else "cast",
        rating_column=box_rating_col[0] if box_rating_col else None,
        boxoffice_column=box_collection_col[0] if box_collection_col else None,
        min_joint_films=2,
        top_n=10,
    )

    if len(chemistry_pairs) > 0:
        logger.info("  Top 10 chemistry pairs identified:")
        for _, row in chemistry_pairs.iterrows():
            logger.info(
                "    %s & %s: uplift=%.4f (%d joint films)",
                row["actor_1"],
                row["actor_2"],
                row["uplift"],
                row["joint_films"],
            )
        chemistry_pairs.to_csv(settings.REPORTS_DIR / "chemistry_pairs.csv", index=False)
    else:
        logger.info("  No chemistry pairs found (insufficient joint appearances >= 2).")

    # =====================================================================
    # STEP 7: BOX-OFFICE MODEL COMPARISONS (Stage 4)
    # =====================================================================

    # Rename suffixed columns to standard form for feature builder
    df_box_model = df_box_clean.copy()
    col_map = {}
    for search, target in [
        ("genre", "genre"),
        ("cast", "cast"),
        ("director", "director"),
        ("duration_minutes", "duration_minutes"),
        ("budget_inr", "budget_inr"),
    ]:
        found = [
            c
            for c in df_box_model.columns
            if c.lower() == search or c.lower().endswith("_left") and search in c.lower()
        ]
        if found:
            col_map[found[0]] = target

    year_candidates = [
        c for c in df_box_model.columns if "year" in c.lower() and c not in (yr_col or "year")
    ]
    if year_candidates:
        col_map[year_candidates[0]] = "year"

    # Also keep rating column for context
    rating_candidates = [
        c for c in df_box_model.columns if "rating" in c.lower() and c != "_match_score"
    ]
    if rating_candidates:
        col_map[rating_candidates[0]] = "rating"

    df_box_model = df_box_model.rename(columns=col_map)
    logger.info("  Box office model columns (renamed): %s", list(df_box_model.columns))

    # Determine target column
    collection_target = [c for c in df_box_model.columns if "worldwide_collection_inr" in c.lower()]
    box_target = collection_target[0] if collection_target else None
    if not box_target:
        logger.warning("  No worldwide_collection_inr column found!")

    # ── STEP 7a: Baseline (no Bankability) ───────────────────────
    _print_separator("STEP 7a: Box-Office Model — Baseline (no Bankability)")
