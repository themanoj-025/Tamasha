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
data_loading.py — Step 1: Load datasets, Step 2: Fuzzy join.
"""

    # =====================================================================
    # STEP 1: LOAD DATA
    # =====================================================================
    _print_separator("STEP 1: Loading Datasets")

    df_imdb = load_imdb_india()
    df_box = load_bollywood_boxoffice()
    df_extra = pd.read_csv(settings.DATA_RAW / "bollywood_movies.csv")

    logger.info("  IMDb India:           %d rows x %d cols", df_imdb.shape[0], df_imdb.shape[1])
    logger.info("  Box Office:           %d rows x %d cols", df_box.shape[0], df_box.shape[1])
    logger.info("  Year Bridge (extra):  %d rows x %d cols", df_extra.shape[0], df_extra.shape[1])

    # =====================================================================
    # STEP 2: TWO-STEP FUZZY JOIN
    # =====================================================================
    _print_separator("STEP 2: Two-Step Fuzzy Join")

    # Step 2a: Box Office → year-bridge (add year info to box office movies)
    logger.info("  Step 2a: Box Office → year-bridge (adding year info)...")
    box_with_years = fuzzy_join_datasets(
        df_box,
        df_extra,
        left_title_col="title",
        right_title_col="title",
        score_cutoff=80.0,
    )
    logger.info(
        "    Matched: %d / %d box office movies now have year info",
        len(box_with_years),
        len(df_box),
    )

    # Extract the year column from the year-bridge side
    year_col = [c for c in box_with_years.columns if c.lower() == "year" or c == "year_right"]
    if year_col:
        yr_col = year_col[0]
        logger.info("    Year column: %s", yr_col)
    else:
        yr_col = None
        logger.warning("    No year column found in bridge join!")

    # Step 2b: Enriched Box Office → IMDb (match on title + year)
    logger.info("  Step 2b: Box Office (with years) → IMDb...")
    joined = fuzzy_join_datasets(
        box_with_years,
        df_imdb,
        left_title_col="title_left",
        right_title_col="title",
        left_year_col=yr_col,
        right_year_col="year",
        score_cutoff=80.0,
        year_tolerance=2,
    )

    logger.info("    High-quality joined dataset: %d movies", len(joined))
    logger.info("    Coverage of box office: %.1f%%", len(joined) / len(df_box) * 100)

    if len(joined) == 0:
        logger.error("No movies joined! Check the join logic.")
        return

    score_high = len(joined[joined["_match_score"] >= 95])
    logger.info(
        "    Score >= 95: %d | 90-94: %d | 80-89: %d",
        score_high,
        len(joined[(joined["_match_score"] >= 90) & (joined["_match_score"] < 95)]),
        len(joined[(joined["_match_score"] >= 80) & (joined["_match_score"] < 90)]),
    )

    # =====================================================================
    # STEP 3: CLEAN
    # =====================================================================
    _print_separator("STEP 3: Cleaning Datasets")
