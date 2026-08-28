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
evaluation.py — Steps 7-9: Box office model, release timing, SHAP.
"""

        logger.warning("  No worldwide_collection_inr column found!")

    # ── STEP 7a: Baseline (no Bankability) ───────────────────────
    _print_separator("STEP 7a: Box-Office Model — Baseline (no Bankability)")

    best_boxoffice_baseline, comparison_boxoffice_baseline = train_boxoffice_model(
        df_box_model,
        boxoffice_column=box_target,
        bankability_df=None,
        run_label="boxoffice",  # Function appends "_baseline"
        tune=True,
        tune_n_iter=15,
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
        tune=True,
        tune_n_iter=15,
    )

    bank_best_name = comparison_boxoffice_with_bank.iloc[0]["model"]
    bank_mae = comparison_boxoffice_with_bank.iloc[0]["MAE"]
    logger.info("  [WITH BANKABILITY] Best: %s (MAE=%.4f)", bank_best_name, bank_mae)

    # Compare
    mae_improvement = (
        ((baseline_mae - bank_mae) / abs(baseline_mae) * 100) if baseline_mae != 0 else 0
    )
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
        """Generate predicted-vs-actual scatter plots for top N models.

        Uses ``cross_val_predict`` to produce **out-of-fold** predictions,
        ensuring the scatter plot visually represents the SAME evaluation
        methodology as the headline MAE in the comparison tables.
        """
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
                # Out-of-fold predictions — matches the CV comparison
                y_pred = cvp(
                    model,
                    X_all,
                    y_all,
                    cv=KFold(n_splits=5, shuffle=True, random_state=42),
                    n_jobs=1,
                )
                save_path = (
                    settings.FIGURES_DIR / f"{prefix}_pred_vs_actual_{model_name.lower()}.png"
                )
                plot_predicted_vs_actual(y_all, y_pred, model_name, save_path=save_path)
                # Verify scatter-plot MAE matches reported CV MAE
                scatter_mae = float(np.mean(np.abs(y_all.values - y_pred)))
                reported_mae = float(comp_df[comp_df["model"] == model_name]["MAE"].iloc[0])
                # Relative tolerance to avoid false positives on low-MAE models
                rel_diff = abs(scatter_mae - reported_mae) / max(reported_mae, 1e-8)
                if rel_diff > 0.05:  # 5% relative tolerance
                    logger.warning(
                        "  Scatter plot MAE (%.4f) differs from reported CV MAE (%.4f) by %.1f%% for %s",
                        scatter_mae,
                        reported_mae,
                        rel_diff * 100,
                        model_name,
                    )
                else:
                    logger.info(
                        "  Scatter plot MAE=%.4f matches reported CV MAE for %s",
                        scatter_mae,
                        model_name,
                    )
                logger.info("  Scatter plot saved: %s", save_path)
            except (OSError, ValueError) as exc:
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
        X_rating_scatter[valid],
        y_rating_scatter[valid],
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
        X_box_scatter[valid],
        y_box_scatter[valid],
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
        X_rating, _, _ = build_feature_matrix(df_rating, target_column_rating="rating")
        X_rating = _ensure_numeric(X_rating)
        y_rating_sub = pd.to_numeric(df_rating["rating"], errors="coerce").loc[X_rating.index]
        valid = y_rating_sub.notna() & ~X_rating.isna().any(axis=1)
        X_rating, y_rating_v = X_rating[valid], y_rating_sub[valid]
        if len(X_rating) > 0:
            best_rating.fit(X_rating, y_rating_v)
            X_sample = X_rating.sample(min(100, len(X_rating)), random_state=42)
            plot_shap_summary(
                best_rating,
                X_sample,
                save_path=settings.FIGURES_DIR / "shap_rating.png",
            )

        # Box office model SHAP
        X_box, _, _ = build_feature_matrix(df_box_model, target_column_boxoffice=box_target)
        X_box = _ensure_numeric(X_box)
        if "cast" in df_box_model.columns and len(bankability_scores) > 0:
            X_box["avg_bankability_score"] = _compute_cast_avg_bankability(
                df_box_model, "cast", bankability_scores
            ).loc[X_box.index]
        y_box = (
            pd.to_numeric(df_box_model[box_target], errors="coerce")
            if box_target in df_box_model.columns
            else None
        )
        if y_box is not None:
            valid = y_box.notna() & ~X_box.isna().any(axis=1)
            X_box, y_box_v = X_box[valid], y_box[valid]
            if len(X_box) > 0:
                best_boxoffice_with_bank.fit(X_box, y_box_v)
                X_sample = X_box.sample(min(100, len(X_box)), random_state=42)
                plot_shap_summary(
                    best_boxoffice_with_bank,
                    X_sample,
                    save_path=settings.FIGURES_DIR / "shap_boxoffice.png",
                )
        logger.info("  SHAP analysis complete.")
    except ImportError as exc:
        logger.info("  SKIP SHAP: %s", exc)
    except (OSError, ValueError) as exc:
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
        logger.info(
            "  Found release_date column: %s (%d movies with dates out of %d)",
            date_cols[0],
            has_dates,
            len(df_box_clean),
        )

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
                festival_count = (
                    df_festival["is_festival_release"].sum()
                    if "is_festival_release" in df_festival.columns
                    else 0
                )
                logger.info(
                    "  Festival releases identified: %d / %d", festival_count, len(df_festival)
                )

                # Analyze: do festival releases outperform?
                box_col_fest = [
                    c for c in df_festival.columns if "worldwide_collection" in c.lower()
                ]
                if box_col_fest and festival_count >= 5:
                    fest_mean = df_festival[df_festival["is_festival_release"]][
                        box_col_fest[0]
                    ].mean()
                    non_fest_mean = df_festival[~df_festival["is_festival_release"]][
                        box_col_fest[0]
                    ].mean()
                    logger.info(
                        "  Avg BOX OFFICE: Festival=₹%.0f, Non-festival=₹%.0f",
                        fest_mean,
                        non_fest_mean,
                    )
                    if fest_mean > non_fest_mean:
                        logger.info(
                            "  → Festival releases outperform by %.1f%%",
                            (fest_mean - non_fest_mean) / non_fest_mean * 100,
                        )
                    else:
                        logger.info(
                            "  → Non-festival releases outperform by %.1f%%",
                            (non_fest_mean - fest_mean) / fest_mean * 100,
                        )

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

                (settings.REPORTS_DIR / "release_timing_analysis.md").write_text(
                    "\n".join(report_lines)
                )
                logger.info("  Festival/clash analysis saved to reports/release_timing_analysis.md")

                # Also merge festival columns back into df_box_model for scenario simulator
                df_box_clean["is_festival_release"] = (
                    df_festival["is_festival_release"]
                    if "is_festival_release" in df_festival.columns
                    else False
                )
                df_box_clean["has_clash"] = (
                    df_clash["has_clash"] if "has_clash" in df_clash.columns else False
                )

            except (OSError, ValueError) as exc:
                logger.warning("  Festival analysis failed: %s", exc)
        else:
            logger.info(
                "  Only %d movies have release dates — insufficient for festival analysis.",
                has_dates,
            )
            logger.info(
                "  (Need at least 30 movies with dates for meaningful festival/clash analysis.)"
            )
    else:
        logger.info("  No release_date column found. Skipping release timing analysis.")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    _print_separator("TRAINING PIPELINE COMPLETE")

    logger.info("  Rating Model:")
    logger.info("    Algorithm: %s", best_rating_name)
    logger.info(
        "    MAE: %.4f | RMSE: %.4f | R²: %.4f", best_rating_mae, best_rating_rmse, best_rating_r2
    )
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
