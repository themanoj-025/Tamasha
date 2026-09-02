"""Data loading, joining, cleaning, and TMDb enrichment (Steps 1-3.5)."""

from __future__ import annotations

import logging

import pandas as pd

from tamasha.config import settings
from tamasha.data.enrichment import enrich_dataset
from tamasha.data.joining import fuzzy_join_datasets, generate_join_quality_report
from tamasha.data.loaders import load_bollywood_boxoffice, load_imdb_india

logger = logging.getLogger(__name__)


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 1: Load all three raw datasets."""
    df_imdb = load_imdb_india()
    df_box = load_bollywood_boxoffice()
    df_extra = pd.read_csv(settings.DATA_RAW / "bollywood_movies.csv")

    logger.info("  IMDb India:           %d rows x %d cols", df_imdb.shape[0], df_imdb.shape[1])
    logger.info("  Box Office:           %d rows x %d cols", df_box.shape[0], df_box.shape[1])
    logger.info("  Year Bridge (extra):  %d rows x %d cols", df_extra.shape[0], df_extra.shape[1])

    return df_imdb, df_box, df_extra


def two_step_fuzzy_join(
    df_imdb: pd.DataFrame,
    df_box: pd.DataFrame,
    df_extra: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """Step 2: Two-step fuzzy join (Box Office → year-bridge → IMDb).

    Returns the joined DataFrame and the year column name.
    """
    # Step 2a: Box Office → year-bridge
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

    year_col = [c for c in box_with_years.columns if c.lower() == "year" or c == "year_right"]
    yr_col = year_col[0] if year_col else None
    if yr_col:
        logger.info("    Year column: %s", yr_col)
    else:
        logger.warning("    No year column found in bridge join!")

    # Step 2b: Enriched Box Office → IMDb
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
        return joined, yr_col

    score_high = len(joined[joined["_match_score"] >= 95])
    logger.info(
        "    Score >= 95: %d | 90-94: %d | 80-89: %d",
        score_high,
        len(joined[(joined["_match_score"] >= 90) & (joined["_match_score"] < 95)]),
        len(joined[(joined["_match_score"] >= 80) & (joined["_match_score"] < 90)]),
    )

    return joined, yr_col


def clean_datasets(
    joined: pd.DataFrame,
    df_imdb: pd.DataFrame,
    yr_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Step 3: Clean both box-office and rating datasets."""
    # Box-office focused dataset
    keep_patterns = [
        "title_left", "title_right", "genre", "rating", "director", "year",
        "cast", "duration_minutes", "worldwide_collection_inr",
        "india_net_collection_inr", "india_gross_collection_inr",
        "overseas_collection_inr", "budget_inr", "verdict", "_match_score", "year_right",
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
            len(df_box_clean), before - len(df_box_clean),
        )

    # Budget numeric
    budget_col = [c for c in df_box_clean.columns if "budget_inr" in c]
    if budget_col:
        df_box_clean[budget_col[0]] = pd.to_numeric(
            df_box_clean[budget_col[0]], errors="coerce"
        ).fillna(0)

    # Clean string columns
    for col in df_box_clean.select_dtypes(include="object").columns:
        df_box_clean[col] = df_box_clean[col].astype(str).str.strip()

    logger.info("  Box office clean shape: %s", df_box_clean.shape)
    df_box_clean.to_parquet(settings.DATA_PROCESSED / "boxoffice_clean.parquet")

    # IMDb-only dataset for rating model
    df_rating = df_imdb.dropna(subset=["rating"]).copy()
    for col in df_rating.select_dtypes(include="object").columns:
        df_rating[col] = df_rating[col].astype(str).str.strip()
    df_rating.to_parquet(settings.DATA_PROCESSED / "imdb_clean.parquet")
    logger.info("  Rating dataset: %d movies with ratings", len(df_rating))

    # Join quality report
    report = generate_join_quality_report(joined, sample_size=15)
    (settings.REPORTS_DIR / "join_quality_report.md").write_text(report)

    return df_box_clean, df_rating


def enrich_with_tmdb(df_box_clean: pd.DataFrame) -> pd.DataFrame:
    """Step 3.5: TMDb enrichment — plot summaries & release dates."""
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

        df_box_clean["plot_summary"] = df_box_enriched["plot_summary"].values
        df_box_clean["release_date"] = df_box_enriched["release_date"].values

        plot_coverage = len(coverage["plots"]) / len(df_box_clean) * 100
        date_coverage = len(coverage["dates"]) / len(df_box_clean) * 100
        logger.info(
            "  TMDb enrichment complete. Plot coverage: %.1f%%, Date coverage: %.1f%%",
            plot_coverage, date_coverage,
        )

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

    return df_box_clean
