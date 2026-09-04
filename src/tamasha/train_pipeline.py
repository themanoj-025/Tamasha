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

Implementation delegates to :mod:`tamasha.pipeline_pkg` modules:
- :mod:`data_loading` — Steps 1-3.5
- :mod:`model_training` — Steps 4, 7
- :mod:`evaluation` — reusable ModelEvaluator
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from tamasha.config import settings
from tamasha.models.boxoffice_model import (
    _compute_cast_avg_bankability,
)
from tamasha.models.model_selection import get_all_models
from tamasha.network.bankability_score import compute_bankability_scores
from tamasha.network.chemistry_pairs import detect_chemistry_pairs
from tamasha.nlp.plot_sentiment import (
    genre_conditional_correlation,
    score_plot_sentiment,
)

# --- New pipeline modules ---
from tamasha.pipeline_pkg.data_loading import (
    clean_datasets,
    enrich_with_tmdb,
    load_datasets,
    two_step_fuzzy_join,
)
from tamasha.pipeline_pkg.model_training import train_boxoffice, train_rating

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
    df_imdb, df_box, df_extra = load_datasets()

    # =====================================================================
    # STEP 2: TWO-STEP FUZZY JOIN
    # =====================================================================
    _print_separator("STEP 2: Two-Step Fuzzy Join")
    joined, yr_col = two_step_fuzzy_join(df_imdb, df_box, df_extra)

    if len(joined) == 0:
        logger.error("No movies joined! Check the join logic.")
        return

    # =====================================================================
    # STEP 3: CLEAN
    # =====================================================================
    _print_separator("STEP 3: Cleaning Datasets")
    df_box_clean, df_rating = clean_datasets(joined, df_imdb, yr_col)

    # ── Inflation adjustment decision ────────────────────────────
    logger.info("  [DECISION] Budget/collection NOT inflation-adjusted.")
    logger.info("    Rationale: The Bollywood box office data (1,000 movies)")
    logger.info("    is heavily concentrated in 2010–2023. With only ~8% of")
    logger.info("    movies from before 2010, inflation adjustment would add")
    logger.info("    complexity for minimal benefit. If needed, use")
    logger.info("    cleaning.inflation_adjust() with a CPI deflator.")

    # =====================================================================
    # STEP 3.5: TMDb ENRICHMENT
    # =====================================================================
    _print_separator("STEP 3.5: TMDb Enrichment — Plot Summaries & Release Dates")
    df_box_clean = enrich_with_tmdb(df_box_clean)

    # =====================================================================
    # STEP 4: RATING MODEL COMPARISON (Stage 3)
    # =====================================================================
    _print_separator("STEP 4: Rating Model Comparison (Stage 3)")
    best_rating, comparison_rating, rating_features = train_rating(df_rating)

    best_rating_name = comparison_rating.iloc[0]["model"]
    best_rating_mae = comparison_rating.iloc[0]["MAE"]
    best_rating_rmse = comparison_rating.iloc[0]["RMSE"]
    best_rating_r2 = comparison_rating.iloc[0]["R2"]

    # =====================================================================
    # STEP 5: PLOT SENTIMENT (Stage 5)
    # =====================================================================
    _print_separator("STEP 5: Plot Sentiment Analysis (Stage 5)")
    _run_plot_sentiment(df_box_clean)

    # =====================================================================
    # STEP 6: BANKABILITY SCORES & CHEMISTRY PAIRS (Stage 6)
    # =====================================================================
    _print_separator("STEP 6: Bankability Scores & Chemistry Pairs (Stage 6)")
    bankability_scores, chemistry_pairs = _run_bankability_analysis(df_box_clean)

    # =====================================================================
    # STEP 7: BOX-OFFICE MODEL COMPARISONS (Stage 4)
    # =====================================================================
    (
        best_boxoffice_baseline,
        best_boxoffice_with_bank,
        comparison_boxoffice_baseline,
        comparison_boxoffice_with_bank,
        baseline_mae,
        bank_mae,
    ) = train_boxoffice(df_box_clean, _get_box_target(df_box_clean), bankability_scores, yr_col)

    baseline_best_name = comparison_boxoffice_baseline.iloc[0]["model"]
    bank_best_name = comparison_boxoffice_with_bank.iloc[0]["model"]
    mae_improvement = ((baseline_mae - bank_mae) / abs(baseline_mae) * 100) if baseline_mae != 0 else 0

    # ── Generate evaluation charts ───────────────────────────────
    _print_separator("Generating Evaluation Charts")
    _generate_all_charts(
        df_rating, df_box_clean, bankability_scores,
        best_rating, best_boxoffice_with_bank,
        comparison_boxoffice_baseline, comparison_boxoffice_with_bank,
        _get_box_target(df_box_clean),
    )

    # =====================================================================
    # STEP 8: RELEASE TIMING (Stage 7)
    # =====================================================================
    _print_separator("STEP 8: Release Timing Analysis (Stage 7)")
    _run_release_timing(df_box_clean)

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


# ── Helper functions ───────────────────────────────────────────────


def _get_box_target(df_box: pd.DataFrame) -> str | None:
    """Find the worldwide collection target column."""
    candidates = [c for c in df_box.columns if "worldwide_collection_inr" in c.lower()]
    return candidates[0] if candidates else None


def _run_plot_sentiment(df_box_clean: pd.DataFrame) -> None:
    """Stage 5: Plot sentiment analysis."""
    plot_col = [c for c in df_box_clean.columns if "plot_summary" in c.lower()]
    if not plot_col:
        logger.info("  No plot_summary column found. Skipping Stage 5.")
        return

    has_plot = df_box_clean["plot_summary"].str.strip().astype(bool).sum()
    logger.info(
        "  Found plot column: '%s' (%d movies with plot text out of %d)",
        plot_col[0], has_plot, len(df_box_clean),
    )

    if has_plot < 20:
        logger.info("  Only %d movies have plot text — insufficient for analysis.", has_plot)
        return

    df_box_plot = df_box_clean[df_box_clean["plot_summary"].str.strip().astype(bool)].copy()
    sentiment_df = score_plot_sentiment(df_box_plot, plot_column=plot_col[0])

    # Genre-conditional correlation
    logger.info("  Computing genre-conditional correlations with box office...")
    box_col_targets = [c for c in df_box_plot.columns if "worldwide_collection" in c.lower()]
    target_col = box_col_targets[0] if box_col_targets else None

    if target_col:
        genre_corr = genre_conditional_correlation(df_box_plot, sentiment_df, target_column=target_col, genre_column="genre")
        if len(genre_corr) > 0:
            genre_corr.to_csv(settings.REPORTS_DIR / "genre_tone_correlation.csv", index=False)
            logger.info("  Genre-tone correlations saved. Top findings:")
            for _, row in genre_corr.iterrows():
                logger.info("    %s: corr=%.4f (n=%d movies)", row["genre"], row["correlation"], row["n_movies"])
        else:
            logger.info("  No genre conditional correlations found (insufficient samples per genre).")

    # Correlation with rating
    rating_col = [c for c in df_box_plot.columns if "rating" in c.lower() and c != "_match_score"]
    if rating_col:
        genre_corr_rating = genre_conditional_correlation(df_box_plot, sentiment_df, target_column=rating_col[0], genre_column="genre")
        if len(genre_corr_rating) > 0:
            genre_corr_rating.to_csv(settings.REPORTS_DIR / "genre_tone_correlation_rating.csv", index=False)
            logger.info("  Genre-tone vs RATING correlations also saved.")


def _run_bankability_analysis(df_box_clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 6: Bankability scores and chemistry pairs."""
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

    bankability_scores.to_csv(settings.REPORTS_DIR / "bankability_scores.csv", index=False)

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

    return bankability_scores, chemistry_pairs


def _generate_all_charts(
    df_rating: pd.DataFrame,
    df_box_clean: pd.DataFrame,
    bankability_scores: pd.DataFrame,
    best_rating: object,
    best_boxoffice_with_bank: object,
    comparison_boxoffice_baseline: pd.DataFrame,
    comparison_boxoffice_with_bank: pd.DataFrame,
    box_target: str | None,
) -> None:
    """Generate evaluation charts and SHAP analysis."""
    from tamasha.evaluation.metrics import plot_model_comparison, plot_predicted_vs_actual
    from tamasha.features.movie_features import build_feature_matrix as _bld

    def _generate_scatter_plots(
        comp_csv: Path,
        X_all: pd.DataFrame,
        y_all: pd.Series,
        prefix: str,
        n_top: int = 3,
    ) -> None:
        if not comp_csv.exists() or len(X_all) < 10:
            return
        comp_df = pd.read_csv(comp_csv)
        top_models = comp_df.head(n_top)["model"].tolist()
        logger.info("  Top %d models for %s: %s", n_top, prefix, top_models)

        from sklearn.model_selection import KFold
        from sklearn.model_selection import cross_val_predict as cvp

        all_models = get_all_models()
        for model_name in top_models:
            if model_name not in all_models:
                logger.warning("  Model %s not available for scatter plot", model_name)
                continue
            try:
                model = all_models[model_name].__class__(**all_models[model_name].get_params())
                y_pred = cvp(model, X_all, y_all, cv=KFold(n_splits=5, shuffle=True, random_state=42), n_jobs=1)
                save_path = settings.FIGURES_DIR / f"{prefix}_pred_vs_actual_{model_name.lower()}.png"
                plot_predicted_vs_actual(y_all, y_pred, model_name, save_path=save_path)
                scatter_mae = float(np.mean(np.abs(y_all.values - y_pred)))
                reported_mae = float(comp_df[comp_df["model"] == model_name]["MAE"].iloc[0])
                rel_diff = abs(scatter_mae - reported_mae) / max(reported_mae, 1e-8)
                if rel_diff > 0.05:
                    logger.warning(
                        "  Scatter plot MAE (%.4f) differs from reported CV MAE (%.4f) by %.1f%% for %s",
                        scatter_mae, reported_mae, rel_diff * 100, model_name,
                    )
                else:
                    logger.info("  Scatter plot MAE=%.4f matches reported CV MAE for %s", scatter_mae, model_name)
                logger.info("  Scatter plot saved: %s", save_path)
            except (OSError, ValueError) as exc:
                logger.warning("  Scatter plot failed for %s: %s", model_name, exc)

    # ── Bar charts ──
    for csv_name, prefix in [
        ("model_comparison_rating.csv", "rating"),
        ("model_comparison_boxoffice_baseline.csv", "boxoffice_baseline"),
        ("model_comparison_boxoffice_with_bankability.csv", "boxoffice_with_bank"),
    ]:
        csv_path = settings.REPORTS_DIR / csv_name
        if csv_path.exists():
            plot_model_comparison(csv_path, save_path=settings.FIGURES_DIR / f"{prefix}_comparison.png")
            logger.info("  Bar chart saved: %s", settings.FIGURES_DIR / f"{prefix}_comparison.png")

    # ── Scatter plots: Rating ──
    logger.info("  Generating predicted-vs-actual scatter plots...")
    X_rating_scatter, y_rating_scatter, _ = _bld(df_rating, target_column_rating="rating")
    X_rating_scatter = X_rating_scatter.select_dtypes(include=[np.number])
    y_rating_scatter = pd.to_numeric(y_rating_scatter, errors="coerce")
    valid = y_rating_scatter.notna() & ~X_rating_scatter.isna().any(axis=1)
    _generate_scatter_plots(settings.REPORTS_DIR / "model_comparison_rating.csv", X_rating_scatter[valid], y_rating_scatter[valid], "rating")

    # ── Scatter plots: Box Office ──
    X_box_scatter, _, y_box_scatter = _bld(df_box_clean, target_column_boxoffice=box_target)
    X_box_scatter = X_box_scatter.select_dtypes(include=[np.number])
    y_box_scatter = pd.to_numeric(y_box_scatter, errors="coerce")
    if "cast" in df_box_clean.columns and len(bankability_scores) > 0:
        X_box_scatter["avg_bankability_score"] = _compute_cast_avg_bankability(
            df_box_clean, "cast", bankability_scores
        ).loc[X_box_scatter.index]
    valid = y_box_scatter.notna() & ~X_box_scatter.isna().any(axis=1)
    _generate_scatter_plots(
        settings.REPORTS_DIR / "model_comparison_boxoffice_with_bankability.csv",
        X_box_scatter[valid], y_box_scatter[valid], "boxoffice_with_bank",
    )

    # ── SHAP analysis (Stage 9) ──
    _run_shap_analysis(df_rating, df_box_clean, bankability_scores, best_rating, best_boxoffice_with_bank, box_target)


def _run_shap_analysis(
    df_rating: pd.DataFrame,
    df_box_clean: pd.DataFrame,
    bankability_scores: pd.DataFrame,
    best_rating: object,
    best_boxoffice: object,
    box_target: str | None,
) -> None:
    """Stage 9: SHAP Explainability."""
    _print_separator("STEP 9: SHAP Explainability")
    try:
        import shap  # noqa: F401

        from tamasha.evaluation.metrics import plot_shap_summary
        from tamasha.features.movie_features import build_feature_matrix as _bld

        def _ensure_numeric(X: pd.DataFrame) -> pd.DataFrame:
            return X.select_dtypes(include=[np.number])

        # Rating model SHAP
        X_rating, _, _ = _bld(df_rating, target_column_rating="rating")
        X_rating = _ensure_numeric(X_rating)
        y_rating_sub = pd.to_numeric(df_rating["rating"], errors="coerce").loc[X_rating.index]
        valid = y_rating_sub.notna() & ~X_rating.isna().any(axis=1)
        X_rating_v, y_rating_v = X_rating[valid], y_rating_sub[valid]
        if len(X_rating_v) > 0:
            best_rating.fit(X_rating_v, y_rating_v)
            X_sample = X_rating_v.sample(min(100, len(X_rating_v)), random_state=42)
            plot_shap_summary(best_rating, X_sample, save_path=settings.FIGURES_DIR / "shap_rating.png")

        # Box office model SHAP
        X_box, _, _ = _bld(df_box_clean, target_column_boxoffice=box_target)
        X_box = _ensure_numeric(X_box)
        if "cast" in df_box_clean.columns and len(bankability_scores) > 0:
            X_box["avg_bankability_score"] = _compute_cast_avg_bankability(
                df_box_clean, "cast", bankability_scores
            ).loc[X_box.index]
        y_box = pd.to_numeric(df_box_clean[box_target], errors="coerce") if box_target in df_box_clean.columns else None
        if y_box is not None:
            valid = y_box.notna() & ~X_box.isna().any(axis=1)
            X_box_v, y_box_v = X_box[valid], y_box[valid]
            if len(X_box_v) > 0:
                best_boxoffice.fit(X_box_v, y_box_v)
                X_sample = X_box_v.sample(min(100, len(X_box_v)), random_state=42)
                plot_shap_summary(best_boxoffice, X_sample, save_path=settings.FIGURES_DIR / "shap_boxoffice.png")
        logger.info("  SHAP analysis complete.")
    except ImportError as exc:
        logger.info("  SKIP SHAP: %s", exc)
    except (OSError, ValueError) as exc:
        logger.info("  SHAP error (non-blocking): %s", exc)


def _run_release_timing(df_box_clean: pd.DataFrame) -> None:
    """Stage 7: Release timing analysis."""
    date_cols = [c for c in df_box_clean.columns if "release_date" in c.lower()]
    if not date_cols:
        logger.info("  No release_date column found. Skipping release timing analysis.")
        return

    has_dates = df_box_clean["release_date"].str.strip().astype(bool).sum()
    logger.info("  Found release_date column: %s (%d movies with dates out of %d)", date_cols[0], has_dates, len(df_box_clean))

    if has_dates < 30:
        logger.info("  Only %d movies have release dates — insufficient for festival analysis.", has_dates)
        logger.info("  (Need at least 30 movies with dates for meaningful festival/clash analysis.)")
        return

    try:
        from tamasha.timing.festival_calendar import (
            compute_clash_feature,
            compute_festival_features,
        )

        year_col_fest = [c for c in df_box_clean.columns if "year" in c.lower()]
        fest_year_col = year_col_fest[0] if year_col_fest else "year"

        df_festival = compute_festival_features(df_box_clean, date_column="release_date", year_column=fest_year_col)
        festival_count = df_festival["is_festival_release"].sum() if "is_festival_release" in df_festival.columns else 0
        logger.info("  Festival releases identified: %d / %d", festival_count, len(df_festival))

        box_col_fest = [c for c in df_festival.columns if "worldwide_collection" in c.lower()]
        if box_col_fest and festival_count >= 5:
            fest_mean = df_festival[df_festival["is_festival_release"]][box_col_fest[0]].mean()
            non_fest_mean = df_festival[~df_festival["is_festival_release"]][box_col_fest[0]].mean()
            logger.info("  Avg BOX OFFICE: Festival=₹%.0f, Non-festival=₹%.0f", fest_mean, non_fest_mean)
            if fest_mean > non_fest_mean:
                logger.info("  → Festival releases outperform by %.1f%%", (fest_mean - non_fest_mean) / non_fest_mean * 100)
            else:
                logger.info("  → Non-festival releases outperform by %.1f%%", (non_fest_mean - fest_mean) / fest_mean * 100)

        df_clash = compute_clash_feature(df_festival, date_column="release_date")
        clash_count = df_clash["has_clash"].sum() if "has_clash" in df_clash.columns else 0
        logger.info("  Clashes identified: %d movies", clash_count)

        # Save festival analysis report
        report_lines = [
            "# Release Timing Analysis (Stage 7)", "",
            f"Movies analyzed: {len(df_festival)}", f"Movies with valid dates: {has_dates}", "",
            "## Festival Releases", f"Total festival releases: {festival_count}",
        ]
        if box_col_fest and festival_count >= 5:
            report_lines.append(f"Average box office (festival): ₹{fest_mean:,.0f}")
            report_lines.append(f"Average box office (non-festival): ₹{non_fest_mean:,.0f}")
            pct = (fest_mean - non_fest_mean) / non_fest_mean * 100
            report_lines.append(f"Difference: {pct:+.1f}%")
        report_lines += ["", "## Clashes", f"Movies with direct clash: {clash_count}"]

        (settings.REPORTS_DIR / "release_timing_analysis.md").write_text("\n".join(report_lines))
        logger.info("  Festival/clash analysis saved to reports/release_timing_analysis.md")

    except (OSError, ValueError) as exc:
        logger.warning("  Festival analysis failed: %s", exc)


if __name__ == "__main__":
    main()
