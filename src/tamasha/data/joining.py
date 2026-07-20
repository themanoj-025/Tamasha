"""Fuzzy-join logic for merging IMDB India and Bollywood Box Office datasets.

Uses ``rapidfuzz`` to match on movie title + release year, logs the
match rate, and provides a convenience function to export a quality
report.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


def _normalise_title(title: str) -> str:
    """Lower-case, strip, and normalise whitespace in a movie title.

    Parameters
    ----------
    title : str
        Raw movie title.

    Returns
    -------
    str
        Normalised title.
    """
    return " ".join(title.lower().split())


def _fuzzy_match_title(
    query: str,
    choices: list[str],
    score_cutoff: float = 60.0,
) -> tuple[str, float]:
    """Find the best fuzzy match for ``query`` among ``choices``.

    Parameters
    ----------
    query : str
        Title to match.
    choices : list[str]
        Candidate titles.
    score_cutoff : float, default=60.0
        Minimum similarity score (0-100).

    Returns
    -------
    tuple[str, float]
        Best-matching title and its score.
    """
    result = process.extractOne(
        query,
        choices,
        scorer=fuzz.WRatio,
        score_cutoff=score_cutoff,
    )
    if result is None:
        return ("", 0.0)
    return result[0], result[1]


def fuzzy_join_datasets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_title_col: str = "title",
    right_title_col: str = "title",
    left_year_col: Optional[str] = None,
    right_year_col: Optional[str] = None,
    score_cutoff: float = 60.0,
    year_tolerance: int = 2,
) -> pd.DataFrame:
    """Fuzzy-join two movie DataFrames on title (+ optional year).

    For every row in ``df_left``, the best fuzzy match in ``df_right``
    is found.  If an optional year column is provided, a candidate must
    also fall within ``year_tolerance`` years of the reference year.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left-side DataFrame (e.g., IMDB India).
    df_right : pd.DataFrame
        Right-side DataFrame (e.g., Bollywood Box Office).
    left_title_col : str, default="title"
        Column in ``df_left`` containing movie titles.
    right_title_col : str, default="title"
        Column in ``df_right`` containing movie titles.
    left_year_col : str, optional
        Column in ``df_left`` with release year.
    right_year_col : str, optional
        Column in ``df_right`` with release year.
    score_cutoff : float, default=60.0
        Minimum ``rapidfuzz`` score.
    year_tolerance : int, default=2
        Maximum absolute year difference.

    Returns
    -------
    pd.DataFrame
        Joined DataFrame with a ``_match_score`` column indicating
        the fuzzy-match quality.
        Duplicate columns are suffixed ``_left`` / ``_right``.
    """
    # Identify common columns that need suffixes
    common_cols = set(df_left.columns) & set(df_right.columns)

    # Rename duplicate columns (copy to avoid mutating inputs)
    left_rename = {c: c + "_left" for c in common_cols}
    right_rename = {c: c + "_right" for c in common_cols}
    df_left = df_left.copy().rename(columns=left_rename)
    df_right = df_right.copy().rename(columns=right_rename)

    # Adjust title column names if they were suffixed
    left_title = left_title_col + "_left" if left_title_col in common_cols else left_title_col
    right_title = right_title_col + "_right" if right_title_col in common_cols else right_title_col

    # Adjust year column names if they were suffixed
    left_year = (
        (left_year_col + "_left" if left_year_col in common_cols else left_year_col)
        if left_year_col
        else None
    )
    right_year = (
        (right_year_col + "_right" if right_year_col in common_cols else right_year_col)
        if right_year_col
        else None
    )

    # Normalise titles in right frame
    right_titles_norm = [_normalise_title(t) for t in df_right[right_title].astype(str)]
    right_titles_orig = df_right[right_title].tolist()
    right_lookup = dict(zip(right_titles_norm, right_titles_orig))

    matched_rows: list[pd.Series] = []
    for _, left_row in df_left.iterrows():
        left_title_norm = _normalise_title(str(left_row[left_title]))
        best_title_norm, score = _fuzzy_match_title(
            left_title_norm, right_titles_norm, score_cutoff
        )

        if best_title_norm:
            best_title_orig = right_lookup[best_title_norm]
            right_candidate = df_right[df_right[right_title] == best_title_orig]

            if right_candidate.empty:
                continue

            right_idx = right_candidate.index[0]
            right_row = df_right.loc[right_idx]

            # Year filter (if available)
            if left_year and right_year:
                left_val = pd.to_numeric(left_row.get(left_year), errors="coerce")
                right_val = pd.to_numeric(right_row.get(right_year), errors="coerce")
                if pd.notna(left_val) and pd.notna(right_val):
                    if abs(left_val - right_val) > year_tolerance:
                        continue

            merged = pd.concat([left_row, right_row], axis=0)
            merged["_match_score"] = score
            matched_rows.append(merged)

    if not matched_rows:
        logger.warning("Fuzzy join produced zero matches.")
        return pd.DataFrame()

    result = pd.DataFrame(matched_rows)
    match_rate = len(result) / len(df_left) * 100
    logger.info(
        "Fuzzy join: %d / %d rows matched (%.1f%%)",
        len(result),
        len(df_left),
        match_rate,
    )
    return result


def generate_join_quality_report(
    joined_df: pd.DataFrame,
    report_path: Optional[str] = None,
    sample_size: int = 15,
) -> str:
    """Generate a Markdown quality report for the fuzzy join.

    Parameters
    ----------
    joined_df : pd.DataFrame
        Result of :func:`fuzzy_join_datasets`.
    report_path : str, optional
        If provided, write the report to this file.
    sample_size : int, default=15
        Number of random matched pairs to include for manual inspection.

    Returns
    -------
    str
        The report as a Markdown string.
    """
    total_rows = len(joined_df)
    match_rate = total_rows  # caller already knows denominator

    # Find left/right title columns (before _match_score split)
    title_cols = [c for c in joined_df.columns if "title" in c.lower()]
    left_title_candidates = [c for c in title_cols if c != "_match_score"]
    # Take first two title-like columns as left/right if possible
    l_col = left_title_candidates[0] if len(left_title_candidates) > 0 else "title_left"
    r_col = left_title_candidates[1] if len(left_title_candidates) > 1 else "title_right"

    sample = joined_df.sample(n=min(sample_size, total_rows), random_state=42)

    lines = [
        "# Join Quality Report",
        "",
        f"**Total matched rows:** {total_rows}",
        f"**Match rate:** {match_rate} rows matched (denominator = left DataFrame size)",
        "",
        "## Random Sample of 15 Matched Pairs",
        "",
        "| # | Left Title | Right Title | Match Score |",
        "|---|-----------|-------------|-------------|",
    ]

    for i, (_, row) in enumerate(sample.iterrows(), 1):
        left_title = str(row.get(l_col, "N/A"))
        right_title = str(row.get(r_col, "N/A"))
        score = row.get("_match_score", "N/A")
        lines.append(f"| {i} | {left_title} | {right_title} | {score} |")

    lines.extend(
        [
            "",
            "## Manual Verification Instructions",
            "",
            "1. Review the 15 samples above.",
            "2. Confirm that each left-right pair is indeed the same movie.",
            "3. If mismatches are found, adjust `score_cutoff` or `year_tolerance`",
            "   and re-run the join.",
            "4. Record any observations below:",
            "",
            "### Observations",
            "",
            "(To be filled after manual inspection)",
        ]
    )

    report = "\n".join(lines)

    if report_path:
        Path(report_path).write_text(report, encoding="utf-8")
        logger.info("Join quality report written to %s", report_path)

    return report
