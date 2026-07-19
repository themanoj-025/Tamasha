"""Tests for the fuzzy-joining module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tamasha.data.joining import fuzzy_join_datasets, generate_join_quality_report


def test_fuzzy_join_exact_match(sample_imdb_df, sample_boxoffice_df):
    """Test that exact title matches produce correct joins."""
    result = fuzzy_join_datasets(
        sample_imdb_df,
        sample_boxoffice_df,
        left_title_col="Title",
        right_title_col="title",
        left_year_col="Year",
        right_year_col="year",
        score_cutoff=80.0,
    )
    # Should match at least Movie A, Movie B, Movie D
    assert len(result) >= 3
    assert "_match_score" in result.columns


def test_fuzzy_join_similar_title(sample_imdb_df, sample_boxoffice_df):
    """Test fuzzy matching with slightly different titles."""
    # Add a movie with close-but-not-identical title
    imdb_extra = sample_imdb_df.copy()
    boxoffice_extra = sample_boxoffice_df.copy()

    imdb_extra = pd.concat(
        [imdb_extra, pd.DataFrame({"Title": ["Mov E"], "Year": [2023], "Rating": [7.0]})],
        ignore_index=True,
    )
    boxoffice_extra = pd.concat(
        [boxoffice_extra, pd.DataFrame({"title": ["Movie E"], "year": [2023], "box_office": ["₹200 Cr"]})],
        ignore_index=True,
    )

    result = fuzzy_join_datasets(
        imdb_extra,
        boxoffice_extra,
        left_title_col="Title",
        right_title_col="title",
        score_cutoff=60.0,
    )
    assert len(result) >= 1


def test_generate_join_report_creates_sample(sample_movie_df):
    """Test that the quality report contains the expected sample."""
    # Use the same df as left and right (trivially matches)
    joined = fuzzy_join_datasets(
        sample_movie_df,
        sample_movie_df,
        left_title_col="title",
        right_title_col="title",
        score_cutoff=80.0,
    )
    report = generate_join_quality_report(joined, sample_size=3)
    assert "Join Quality Report" in report
    assert "Matched Pairs" in report
