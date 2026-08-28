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
data_processing.py — Step 3: Clean datasets, Step 3.5: TMDb enrichment.
"""

    # =====================================================================
    # STEP 3: CLEAN
    # =====================================================================
    _print_separator("STEP 3: Cleaning Datasets")

    # For rating model: clean IMDb data
    # For box office model: clean joined data
    # We clean the joined data; IMDb data is already clean enough from loaders

    # ── Prepare box-office focused dataset ────────────────────────
    keep_patterns = [
        "title_left",
        "title_right",
        "genre",
        "rating",
        "director",
        "year",
        "cast",
        "duration_minutes",
        "worldwide_collection_inr",
        "india_net_collection_inr",
        "india_gross_collection_inr",
        "overseas_collection_inr",
        "budget_inr",
        "verdict",
        "_match_score",
        "year_right",
    ]
    if yr_col:
        keep_patterns.append(yr_col)

    available_cols = [c for c in keep_patterns if c in joined.columns]
    df_box_clean = joined[available_cols].copy()

    # Drop rows with missing collection target
    box_col = [c for c in df_box_clean.columns if "worldwide_collection" in c]
    if box_col:
        before = len(df_box_clean)
        df_box_clean = df_box_clean.dropna(subset=[box_col[0]])
        logger.info(
            "  Box office data: %d rows (dropped %d with missing collection)",
            len(df_box_clean),
            before - len(df_box_clean),
        )

    # Budget is already numeric; fill zeros for NaN
    budget_col = [c for c in df_box_clean.columns if "budget_inr" in c]
    if budget_col:
        df_box_clean[budget_col[0]] = pd.to_numeric(
            df_box_clean[budget_col[0]], errors="coerce"
        ).fillna(0)

    # Clean string columns
    for col in df_box_clean.select_dtypes(include="object").columns:
        df_box_clean[col] = df_box_clean[col].astype(str).str.strip()

    logger.info("  Box office clean shape: %s", df_box_clean.shape)
    logger.info("  Columns: %s", list(df_box_clean.columns))

    # Save cleaned dataset
    df_box_clean.to_parquet(settings.DATA_PROCESSED / "boxoffice_clean.parquet")
    logger.info("  Saved cleaned box office dataset")

    # ── Inflation adjustment decision ────────────────────────────
    logger.info("  [DECISION] Budget/collection NOT inflation-adjusted.")
    logger.info("    Rationale: The Bollywood box office data (1,000 movies)")
    logger.info("    is heavily concentrated in 2010–2023. With only ~8% of")
    logger.info("    movies from before 2010, inflation adjustment would add")
    logger.info("    complexity for minimal benefit. If needed, use")
    logger.info("    cleaning.inflation_adjust() with a CPI deflator.")

    # ── Prepare IMDb-only dataset for rating model ──────────────
    df_rating = df_imdb.dropna(subset=["rating"]).copy()
    for col in df_rating.select_dtypes(include="object").columns:
        df_rating[col] = df_rating[col].astype(str).str.strip()
    df_rating.to_parquet(settings.DATA_PROCESSED / "imdb_clean.parquet")
    logger.info("  Rating dataset: %d movies with ratings", len(df_rating))

    # ── Generate join quality report ─────────────────────────────
    report = generate_join_quality_report(joined, sample_size=15)
    (settings.REPORTS_DIR / "join_quality_report.md").write_text(report)

    # =====================================================================
    # STEP 3.5: TMDb ENRICHMENT (Focus 2)
    # =====================================================================
    _print_separator("STEP 3.5: TMDb Enrichment — Plot Summaries & Release Dates")

    # Enrich the BOX OFFICE dataset (the smaller, focused dataset)
    # This gets us plot summaries and release dates via TMDb API
    logger.info("  Enriching box office dataset (%d movies) from TMDb...", len(df_box_clean))
    title_col = [c for c in df_box_clean.columns if "title" in c.lower() and "left" in c.lower()]
    if not title_col:
        title_col = [c for c in df_box_clean.columns if "title" in c.lower()]
    year_col_enrich = [c for c in df_box_clean.columns if "year" in c.lower()]

    enrich_title = title_col[0] if title_col else "title"
    enrich_year = year_col_enrich[0] if year_col_enrich else None

    logger.info("  Using title column: '%s', year column: %s", enrich_title, enrich_year)

    try:
        coverage, df_box_enriched = enrich_dataset(
            df_box_clean,
            title_column=enrich_title,
            year_column=enrich_year,
            max_movies=len(df_box_clean),
        )

        # Merge enriched columns back into box office model dataframe
        df_box_clean["plot_summary"] = df_box_enriched["plot_summary"].values
        df_box_clean["release_date"] = df_box_enriched["release_date"].values

        plot_coverage = len(coverage["plots"]) / len(df_box_clean) * 100
        date_coverage = len(coverage["dates"]) / len(df_box_clean) * 100
        logger.info(
            "  TMDb enrichment complete. Plot coverage: %.1f%%, Date coverage: %.1f%%",
            plot_coverage,
            date_coverage,
        )

        # Save enrichment report
        (settings.REPORTS_DIR / "tmdb_enrichment_coverage.md").write_text(
            f"# TMDb Enrichment Coverage\n\n"
            f"- Movies attempted: {len(df_box_clean)}\n"
            f"- Plot coverage: {plot_coverage:.1f}% ({len(coverage['plots'])} movies)\n"
            f"- Date coverage: {date_coverage:.1f}% ({len(coverage['dates'])} movies)\n"
            f"- Attempted on: {enrich_title} + {enrich_year or 'N/A'}\n"
        )
    except (OSError, ValueError) as exc:
        logger.warning("TMDb enrichment failed: %s. Proceeding without enrichment.", exc)
        df_box_clean["plot_summary"] = ""
        df_box_clean["release_date"] = ""

    # =====================================================================
    # STEP 4: RATING MODEL COMPARISON (Stage 3)
    # =====================================================================
    _print_separator("STEP 4: Rating Model Comparison (Stage 3)")
