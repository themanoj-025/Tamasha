"""Tests for feature engineering modules."""

from __future__ import annotations

import pandas as pd

from tamasha.features.movie_features import (
    build_feature_matrix,
    extract_cast_features,
    extract_genre_features,
    extract_runtime_feature,
)


class TestExtractGenreFeatures:
    """Tests for extract_genre_features."""

    def test_one_hot_encodes_genres(self):
        df = pd.DataFrame({"genre": ["Action, Drama", "Comedy"]})
        result = extract_genre_features(df)
        assert "genre_Action" in result.columns
        assert "genre_Drama" in result.columns
        assert "genre_Comedy" in result.columns
        assert result.shape[1] == 3

    def test_handles_empty_genres(self):
        df = pd.DataFrame({"genre": ["", "Action"]})
        result = extract_genre_features(df)
        assert "genre_Action" in result.columns


class TestExtractCastFeatures:
    """Tests for extract_cast_features."""

    def test_cast_size(self):
        df = pd.DataFrame({"cast": ["Actor A, Actor B, Actor C"], "director": ["Dir A"]})
        result = extract_cast_features(df)
        assert result["cast_size"].iloc[0] == 3

    def test_director_encoded(self):
        df = pd.DataFrame({"cast": ["Actor A", "Actor B"], "director": ["Dir A", "Dir B"]})
        result = extract_cast_features(df)
        assert "director_encoded" in result.columns
        assert result["director_encoded"].nunique() == 2


class TestExtractRuntimeFeature:
    """Tests for extract_runtime_feature."""

    def test_fills_nan_with_median(self):
        df = pd.DataFrame({"runtime": [120, None, 150]})
        result = extract_runtime_feature(df)
        assert result["runtime_minutes"].isna().sum() == 0


class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix."""

    def test_returns_all_components(self, sample_movie_df):
        X, y_rating, y_boxoffice = build_feature_matrix(
            sample_movie_df,
            target_column_rating="rating",
            target_column_boxoffice="collection_inr",
        )
        assert X.shape[0] == 5
        assert y_rating is not None
        assert y_boxoffice is not None
        assert len(y_rating) == 5
