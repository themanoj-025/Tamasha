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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tamasha.config import settings  # noqa: E402
from tamasha.data.loaders import (  # noqa: E402
    load_imdb_india,
    load_bollywood_boxoffice,
)
from tamasha.data.joining import fuzzy_join_datasets, generate_join_quality_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tamasha.features.movie_features import build_feature_matrix  # noqa: E402
from tamasha.models.model_selection import get_all_models  # noqa: E402
from tamasha.models.rating_model import train_rating_model  # noqa: E402
from tamasha.models.boxoffice_model import train_boxoffice_model, _compute_cast_avg_bankability  # noqa: E402
from tamasha.network.bankability_score import compute_bankability_scores  # noqa: E402
from tamasha.network.chemistry_pairs import detect_chemistry_pairs  # noqa: E402
from tamasha.data.enrichment import enrich_dataset  # noqa: E402
from tamasha.nlp.plot_sentiment import score_plot_sentiment, genre_conditional_correlation  # noqa: E402

logger = logging.getLogger(__name__)


def _print_separator(title: str) -> None:
    """Print a section separator with title."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  %s", title)
    logger.info("=" * 70)


def main() -> None:
    """Run the full training pipeline."""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Tamasha Training Pipeline — Starting")
    logger.info("Config: PROJECT_ROOT=%s", settings.PROJECT_ROOT)

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
    logger.info("    Matched: %d / %d box office movies now have year info", len(box_with_years), len(df_box))

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
    logger.info("    Score >= 95: %d | 90-94: %d | 80-89: %d",
                score_high,
                len(joined[(joined["_match_score"] >= 90) & (joined["_match_score"] < 95)]),
                len(joined[(joined["_match_score"] >= 80) & (joined["_match_score"] < 90)]))

    # =====================================================================
    # STEP 3: CLEAN
    # =====================================================================
    _print_separator("STEP 3: Cleaning Datasets")

    # For rating model: clean IMDb data
    # For box office model: clean joined data
    # We clean the joined data; IMDb data is already clean enough from loaders

    # ── Prepare box-office focused dataset ────────────────────────
    keep_patterns = [
        "title_left", "title_right",
        "genre", "rating", "director", "year", "cast",
        "duration_minutes",
        "worldwide_collection_inr", "india_net_collection_inr",
        "india_gross_collection_inr", "overseas_collection_inr",
        "budget_inr", "verdict",
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
        logger.info("  Box office data: %d rows (dropped %d with missing collection)",
                     len(df_box_clean), before - len(df_box_clean))

    # Budget is already numeric; fill zeros for NaN
    budget_col = [c for c in df_box_clean.columns if "budget_inr" in c]
    if budget_col:
        df_box_clean[budget_col[0]] = (
            pd.to_numeric(df_box_clean[budget_col[0]], errors="coerce").fillna(0)
        )

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
        logger.info("  TMDb enrichment complete. Plot coverage: %.1f%%, Date coverage: %.1f%%",
                     plot_coverage, date_coverage)

        # Save enrichment report
        (settings.REPORTS_DIR / "tmdb_enrichment_coverage.md").write_text(
            f"# TMDb Enrichment Coverage\n\n"
            f"- Movies attempted: {len(df_box_clean)}\n"
            f"- Plot coverage: {plot_coverage:.1f}% ({len(coverage['plots'])} movies)\n"
            f"- Date coverage: {date_coverage:.1f}% ({len(coverage['dates'])} movies)\n"
            f"- Attempted on: {enrich_title} + {enrich_year or 'N/A'}\n"
        )
    except Exception as exc:
        logger.warning("TMDb enrichment failed: %s. Proceeding without enrichment.", exc)
        df_box_clean["plot_summary"] = ""
        df_box_clean["release_date"] = ""

    # =====================================================================
    # STEP 4: RATING MODEL COMPARISON (Stage 3)
    # =====================================================================
    _print_separator("STEP 4: Rating Model Comparison (Stage 3)")

    logger.info("  Training 8 models on %d movies with ratings...", len(df_rating))

    best_rating, comparison_rating = train_rating_model(
        df_rating,
        rating_column="rating",
    )

    best_rating_name = comparison_rating.iloc[0]["model"]
    best_rating_mae = comparison_rating.iloc[0]["MAE"]
    best_rating_rmse = comparison_rating.iloc[0]["RMSE"]
    best_rating_r2 = comparison_rating.iloc[0]["R2"]

    logger.info("  [RATING] Best model: %s", best_rating_name)
    logger.info("  [RATING] MAE=%.4f | RMSE=%.4f | R²=%.4f",
                best_rating_mae, best_rating_rmse, best_rating_r2)
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
    except Exception as exc:
        logger.warning("  Director encoder save failed (non-blocking): %s", exc)

    # =====================================================================
    # STEP 5: PLOT SENTIMENT (Stage 5) — using TMDb-enriched plot summaries
    # =====================================================================
    _print_separator("STEP 5: Plot Sentiment Analysis (Stage 5)")

    # Use the TMDb-enriched plot_summary column from the box office dataset
    plot_col = [c for c in df_box_clean.columns if "plot_summary" in c.lower()]
    if plot_col:
        has_plot = df_box_clean["plot_summary"].str.strip().astype(bool).sum()
        logger.info("  Found plot column: '%s' (%d movies with plot text out of %d)",
                     plot_col[0], has_plot, len(df_box_clean))

        if has_plot >= 20:
            # Limit to movies with plot text
            df_box_plot = df_box_clean[df_box_clean["plot_summary"].str.strip().astype(bool)].copy()
            sentiment_df = score_plot_sentiment(df_box_plot, plot_column=plot_col[0])

            # Genre-conditional correlation: does tone correlate with box office WITHIN each genre?
            logger.info("  Computing genre-conditional correlations with box office...")
            # Auto-detect the collection column (may be suffixed after join)
            box_col_targets = [c for c in df_box_plot.columns if "worldwide_collection" in c.lower()]
            target_col = box_col_targets[0] if box_col_targets else None
            genre_corr = genre_conditional_correlation(
                df_box_plot, sentiment_df,
                target_column=target_col, genre_column="genre",
            ) if target_col else pd.DataFrame()
            if len(genre_corr) > 0:
                genre_corr.to_csv(settings.REPORTS_DIR / "genre_tone_correlation.csv", index=False)
                logger.info("  Genre-tone correlations saved. Top findings:")
                for _, row in genre_corr.iterrows():
                    logger.info("    %s: corr=%.4f (n=%d movies)", row["genre"], row["correlation"], row["n_movies"])
            else:
                logger.info("  No genre conditional correlations found (insufficient samples per genre).")

            # Also check correlation with IMDB rating (if available)
            rating_col = [c for c in df_box_plot.columns if "rating" in c.lower() and c != "_match_score"]
            if rating_col:
                genre_corr_rating = genre_conditional_correlation(
                    df_box_plot, sentiment_df,
                    target_column=rating_col[0], genre_column="genre",
                )
                if len(genre_corr_rating) > 0:
                    genre_corr_rating.to_csv(settings.REPORTS_DIR / "genre_tone_correlation_rating.csv", index=False)
                    logger.info("  Genre-tone vs RATING correlations also saved.")
        else:
            logger.info("  Only %d movies have plot text — insufficient for genre-conditional analysis.", has_plot)
            logger.info("  (Need at least 20 movies with plot summaries to compute meaningful correlations per genre.)")
    else:
        logger.info("  No plot_summary column found. Skipping Stage 5.")

    # =====================================================================
    # STEP 6: BANKABILITY SCORES & CHEMISTRY PAIRS (Stage 6)
    # =====================================================================
    _print_separator("STEP 6: Bankability Scores & Chemistry Pairs (Stage 6)")

    # Use joined dataset which has box office data for weighted scoring
    # Find the right columns for rating and collection
    box_rating_col = [c for c in df_box_clean.columns if "rating" in c.lower() and c != "_match_score"]
    box_collection_col = [c for c in df_box_clean.columns if "worldwide_collection" in c.lower() or "collection_inr" in c.lower()]
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
        logger.info("    %s [%s]: score=%.4f (%d films)", row["actor"], row["type"], row["bankability_score"], row["film_count"])

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
            logger.info("    %s & %s: uplift=%.4f (%d joint films)", row["actor_1"], row["actor_2"], row["uplift"], row["joint_films"])
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
        found = [c for c in df_box_model.columns if c.lower() == search or c.lower().endswith("_left") and search in c.lower()]
        if found:
            col_map[found[0]] = target

    year_candidates = [c for c in df_box_model.columns if "year" in c.lower() and c not in (yr_col or "year")]
    if year_candidates:
        col_map[year_candidates[0]] = "year"

    # Also keep rating column for context
    rating_candidates = [c for c in df_box_model.columns if "rating" in c.lower() and c != "_match_score"]
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

    best_boxoffice_baseline, comparison_boxoffice_baseline = train_boxoffice_model(
        df_box_model,
        boxoffice_column=box_target,
        bankability_df=None,
        run_label="boxoffice",  # Function appends "_baseline"
    )

    baseline_best_name = comparison_boxoffice_baseline.iloc[0]["model"]
    baseline_mae = comparison_boxoffice_baseline.iloc[0]["MAE"]
    logger.info("  [BASELINE] Best: %s (MAE=%.4f)", baseline_best_name, baseline_mae)

    # Save box office feature columns (before bankability is added)
    X_box_feat, _, _ = build_feature_matrix(df_box_model, target_column_boxoffice=box_target)
    box_features_base = X_box_feat.columns.tolist()
    box_features = box_features_base + ["avg_bankability_score"]
    (settings.MODELS_DIR / "boxoffice_features.json").write_text(json.dumps(box_features))
    logger.info("  Saved %d box office feature column names", len(box_features))

    # ── STEP 7b: With Bankability Score ──────────────────────────
    _print_separator("STEP 7b: Box-Office Model — with Bankability Score")

    # Pass the real bankability_scores — train_boxoffice_model will
    # compute avg_bankability_score internally via _compute_cast_avg_bankability
    best_boxoffice_with_bank, comparison_boxoffice_with_bank = train_boxoffice_model(
        df_box_model,
        boxoffice_column=box_target,
        bankability_df=bankability_scores,
        run_label="boxoffice",  # Function appends "_with_bankability"
    )

    bank_best_name = comparison_boxoffice_with_bank.iloc[0]["model"]
    bank_mae = comparison_boxoffice_with_bank.iloc[0]["MAE"]
    logger.info("  [WITH BANKABILITY] Best: %s (MAE=%.4f)", bank_best_name, bank_mae)

    # Compare
    mae_improvement = ((baseline_mae - bank_mae) / abs(baseline_mae) * 100) if baseline_mae != 0 else 0
    logger.info("")
    logger.info("  MAE Improvement from Bankability Score: %.1f%%", mae_improvement)
    logger.info("    Baseline MAE:      %.4f", baseline_mae)
    logger.info("    With Bankability:  %.4f", bank_mae)

    # ── Generate evaluation charts ───────────────────────────────
    _print_separator("Generating Evaluation Charts")
    from tamasha.evaluation.metrics import plot_model_comparison, plot_predicted_vs_actual

    def _generate_scatter_plots(
        comp_csv: Path,
        X_all: pd.DataFrame,
        y_all: pd.Series,
        prefix: str,
        n_top: int = 3,
    ) -> None:
        """Generate predicted-vs-actual scatter plots for top N models."""
        if not comp_csv.exists() or len(X_all) < 10:
            return
        comp_df = pd.read_csv(comp_csv)
        top_models = comp_df.head(n_top)["model"].tolist()
        logger.info("  Top %d models for %s: %s", n_top, prefix, top_models)

        # Single train/test split for consistent comparison
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=42
        )

        all_models = get_all_models()
        for model_name in top_models:
            if model_name not in all_models:
                logger.warning("  Model %s not available for scatter plot", model_name)
                continue
            try:
                model = all_models[model_name].__class__(**all_models[model_name].get_params())
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                save_path = settings.FIGURES_DIR / f"{prefix}_pred_vs_actual_{model_name.lower()}.png"
                plot_predicted_vs_actual(y_test, y_pred, model_name, save_path=save_path)
                logger.info("  Scatter plot saved: %s", save_path)
            except Exception as exc:
                logger.warning("  Scatter plot failed for %s: %s", model_name, exc)

    # ── Bar charts ────────────────────────────────────────────
    for csv_name, prefix in [
        ("model_comparison_rating.csv", "rating"),
        ("model_comparison_boxoffice_baseline.csv", "boxoffice_baseline"),
        ("model_comparison_boxoffice_with_bankability.csv", "boxoffice_with_bank"),
    ]:
        csv_path = settings.REPORTS_DIR / csv_name
        if csv_path.exists():
            plot_model_comparison(
                csv_path,
                save_path=settings.FIGURES_DIR / f"{prefix}_comparison.png",
            )
            logger.info("  Bar chart saved: %s", settings.FIGURES_DIR / f"{prefix}_comparison.png")

    # ── Scatter plots: Rating ─────────────────────────────────
    logger.info("  Generating predicted-vs-actual scatter plots...")
    X_rating_scatter, y_rating_scatter, _ = build_feature_matrix(
        df_rating, target_column_rating="rating"
    )
    X_rating_scatter = X_rating_scatter.select_dtypes(include=[np.number])
    y_rating_scatter = pd.to_numeric(y_rating_scatter, errors="coerce")
    valid = y_rating_scatter.notna() & ~X_rating_scatter.isna().any(axis=1)
    _generate_scatter_plots(
        settings.REPORTS_DIR / "model_comparison_rating.csv",
        X_rating_scatter[valid], y_rating_scatter[valid],
        "rating",
    )

    # ── Scatter plots: Box Office ─────────────────────────────
    X_box_scatter, _, y_box_scatter = build_feature_matrix(
        df_box_model, target_column_boxoffice=box_target
    )
    X_box_scatter = X_box_scatter.select_dtypes(include=[np.number])
    y_box_scatter = pd.to_numeric(y_box_scatter, errors="coerce")
    if "cast" in df_box_model.columns and len(bankability_scores) > 0:
        X_box_scatter["avg_bankability_score"] = _compute_cast_avg_bankability(
            df_box_model, "cast", bankability_scores
        ).loc[X_box_scatter.index]
    valid = y_box_scatter.notna() & ~X_box_scatter.isna().any(axis=1)
    _generate_scatter_plots(
        settings.REPORTS_DIR / "model_comparison_boxoffice_with_bankability.csv",
        X_box_scatter[valid], y_box_scatter[valid],
        "boxoffice_with_bank",
    )

    # ── SHAP analysis (Stage 9) ──────────────────────────────────
    _print_separator("STEP 9: SHAP Explainability")
    try:
        import shap  # noqa: F401
        from tamasha.evaluation.metrics import plot_shap_summary

        def _ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
            return X.select_dtypes(include=[np.number])

        # Rating model SHAP
        X_rating, _, _ = build_feature_matrix(
            df_rating, target_column_rating="rating"
        )
        X_rating = _ensure_numeric(X_rating)
        y_rating_sub = pd.to_numeric(df_rating["rating"], errors="coerce").loc[X_rating.index]
        valid = y_rating_sub.notna() & ~X_rating.isna().any(axis=1)
        X_rating, y_rating_v = X_rating[valid], y_rating_sub[valid]
        if len(X_rating) > 0:
            best_rating.fit(X_rating, y_rating_v)
            X_sample = X_rating.sample(min(100, len(X_rating)), random_state=42)
            plot_shap_summary(
                best_rating, X_sample,
                save_path=settings.FIGURES_DIR / "shap_rating.png",
            )

        # Box office model SHAP
        X_box, _, _ = build_feature_matrix(
            df_box_model, target_column_boxoffice=box_target
        )
        X_box = _ensure_numeric(X_box)
        if "cast" in df_box_model.columns and len(bankability_scores) > 0:
            X_box["avg_bankability_score"] = _compute_cast_avg_bankability(
                df_box_model, "cast", bankability_scores
            ).loc[X_box.index]
        y_box = pd.to_numeric(df_box_model[box_target], errors="coerce") if box_target in df_box_model.columns else None
        if y_box is not None:
            valid = y_box.notna() & ~X_box.isna().any(axis=1)
            X_box, y_box_v = X_box[valid], y_box[valid]
            if len(X_box) > 0:
                best_boxoffice_with_bank.fit(X_box, y_box_v)
                X_sample = X_box.sample(min(100, len(X_box)), random_state=42)
                plot_shap_summary(
                    best_boxoffice_with_bank, X_sample,
                    save_path=settings.FIGURES_DIR / "shap_boxoffice.png",
                )
        logger.info("  SHAP analysis complete.")
    except ImportError as exc:
        logger.info("  SKIP SHAP: %s", exc)
    except Exception as exc:
        logger.info("  SHAP error (non-blocking): %s", exc)

    # =====================================================================
    # STEP 8: RELEASE TIMING (Stage 7) — using TMDb-enriched release dates
    # =====================================================================
    _print_separator("STEP 8: Release Timing Analysis (Stage 7)")

    # Use the TMDb-enriched release_date column from df_box_clean
    # Also need to merge into df_box_model since that's what model features use
    date_cols = [c for c in df_box_clean.columns if "release_date" in c.lower()]
    if date_cols:
        has_dates = df_box_clean["release_date"].str.strip().astype(bool).sum()
        logger.info("  Found release_date column: %s (%d movies with dates out of %d)",
                     date_cols[0], has_dates, len(df_box_clean))

        if has_dates >= 30:
            # Compute festival features
            try:
                from tamasha.timing.festival_calendar import compute_festival_features

                # Auto-detect the year column (may be suffixed after join)
                year_col_fest = [c for c in df_box_clean.columns if "year" in c.lower()]
                fest_year_col = year_col_fest[0] if year_col_fest else "year"

                df_festival = compute_festival_features(
                    df_box_clean,
                    date_column="release_date",
                    year_column=fest_year_col,
                )
                festival_count = df_festival["is_festival_release"].sum() if "is_festival_release" in df_festival.columns else 0
                logger.info("  Festival releases identified: %d / %d",
                             festival_count, len(df_festival))

                # Analyze: do festival releases outperform?
                box_col_fest = [c for c in df_festival.columns if "worldwide_collection" in c.lower()]
                if box_col_fest and festival_count >= 5:
                    fest_mean = df_festival[df_festival["is_festival_release"]][box_col_fest[0]].mean()
                    non_fest_mean = df_festival[~df_festival["is_festival_release"]][box_col_fest[0]].mean()
                    logger.info("  Avg BOX OFFICE: Festival=₹%.0f, Non-festival=₹%.0f", fest_mean, non_fest_mean)
                    if fest_mean > non_fest_mean:
                        logger.info("  → Festival releases outperform by %.1f%%",
                                     (fest_mean - non_fest_mean) / non_fest_mean * 100)
                    else:
                        logger.info("  → Non-festival releases outperform by %.1f%%",
                                     (non_fest_mean - fest_mean) / fest_mean * 100)

                # Compute clash features
                from tamasha.timing.festival_calendar import compute_clash_feature
                df_clash = compute_clash_feature(df_festival, date_column="release_date")
                clash_count = df_clash["has_clash"].sum() if "has_clash" in df_clash.columns else 0
                logger.info("  Clashes identified: %d movies", clash_count)

                # Save festival analysis report
                report_lines = [
                    "# Release Timing Analysis (Stage 7)",
                    "",
                    f"Movies analyzed: {len(df_festival)}",
                    f"Movies with valid dates: {has_dates}",
                    "",
                    "## Festival Releases",
                    f"Total festival releases: {festival_count}",
                ]
                if box_col_fest and festival_count >= 5:
                    report_lines.append(f"Average box office (festival): ₹{fest_mean:,.0f}")
                    report_lines.append(f"Average box office (non-festival): ₹{non_fest_mean:,.0f}")
                    pct = (fest_mean - non_fest_mean) / non_fest_mean * 100
                    report_lines.append(f"Difference: {pct:+.1f}%")
                report_lines.append("")
                report_lines.append("## Clashes")
                report_lines.append(f"Movies with direct clash: {clash_count}")

                (settings.REPORTS_DIR / "release_timing_analysis.md").write_text("\n".join(report_lines))
                logger.info("  Festival/clash analysis saved to reports/release_timing_analysis.md")

                # Also merge festival columns back into df_box_model for scenario simulator
                df_box_clean["is_festival_release"] = df_festival["is_festival_release"] if "is_festival_release" in df_festival.columns else False
                df_box_clean["has_clash"] = df_clash["has_clash"] if "has_clash" in df_clash.columns else False

            except Exception as exc:
                logger.warning("  Festival analysis failed: %s", exc)
        else:
            logger.info("  Only %d movies have release dates — insufficient for festival analysis.", has_dates)
            logger.info("  (Need at least 30 movies with dates for meaningful festival/clash analysis.)")
    else:
        logger.info("  No release_date column found. Skipping release timing analysis.")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    _print_separator("TRAINING PIPELINE COMPLETE")

    logger.info("  Rating Model:")
    logger.info("    Algorithm: %s", best_rating_name)
    logger.info("    MAE: %.4f | RMSE: %.4f | R²: %.4f", best_rating_mae, best_rating_rmse, best_rating_r2)
    logger.info("    Saved: models/best_rating_model.pkl")
    logger.info("")
    logger.info("  Box Office Model (Baseline):")
    logger.info("    Algorithm: %s", baseline_best_name)
    logger.info("    MAE: %.4f", baseline_mae)
    logger.info("")
    logger.info("  Box Office Model (with Bankability):")
    logger.info("    Algorithm: %s", bank_best_name)
    logger.info("    MAE: %.4f", bank_mae)
    logger.info("    MAE Improvement: %.1f%%", mae_improvement)
    logger.info("    Saved: models/best_boxoffice_model.pkl")
    logger.info("")
    logger.info("  Reports:")
    logger.info("    reports/model_comparison_rating.csv")
    logger.info("    reports/model_comparison_boxoffice_baseline.csv")
    logger.info("    reports/model_comparison_boxoffice_with_bankability.csv")
    logger.info("    reports/bankability_scores.csv")
    logger.info("    reports/chemistry_pairs.csv")
    logger.info("")
    logger.info("★ Pipeline finished. Run 'make test' to verify.")


if __name__ == "__main__":
    main()
