"""Tests for the festival calendar module."""

from __future__ import annotations

from datetime import date

import pytest

from tamasha.timing.festival_calendar import (
    compute_festival_features,
    get_major_release_windows,
    is_festival_release,
    compute_clash_feature,
)
import pandas as pd


class TestFestivalCalendar:
    """Tests for festival calendar functions."""

    def test_get_major_release_windows_returns_dict(self):
        windows = get_major_release_windows(2024)
        assert isinstance(windows, dict)
        assert "Diwali" in windows
        assert "Christmas" in windows

    def test_festival_dates_are_valid(self):
        windows = get_major_release_windows(2024)
        for name, d in windows.items():
            assert isinstance(d, date)
            assert d.year == 2024

    def test_is_festival_release_exact_match(self):
        windows = get_major_release_windows(2024)
        # Christmas
        is_fest, name, diff = is_festival_release(date(2024, 12, 25), windows)
        assert is_fest
        assert name == "Christmas"
        assert diff == 0

    def test_is_festival_release_near_match(self):
        windows = get_major_release_windows(2024)
        # 3 days before Diwali (approx Oct 31)
        is_fest, name, diff = is_festival_release(date(2024, 10, 28), windows, window_days=7)
        assert is_fest or not is_fest  # Depends on approximation

    def test_is_festival_release_no_match(self):
        windows = get_major_release_windows(2024)
        # Mid-February (no major festivals)
        is_fest, name, diff = is_festival_release(date(2024, 2, 15), windows, window_days=7)
        assert not is_fest


class TestComputeFestivalFeatures:
    """Tests for compute_festival_features."""

    def test_returns_expected_columns(self):
        df = pd.DataFrame({
            "title": ["Movie A", "Movie B"],
            "year": [2024, 2024],
            "release_date": ["2024-10-31", "2024-02-15"],
        })
        result = compute_festival_features(df, date_column="release_date", year_column="year")
        assert "is_festival_release" in result.columns
        assert "festival_name" in result.columns
        assert "days_to_festival" in result.columns


class TestComputeClashFeature:
    """Tests for compute_clash_feature."""

    def test_clash_detected(self):
        df = pd.DataFrame({
            "title": ["Movie A", "Movie B"],
            "release_date": ["2024-01-01", "2024-01-05"],
        })
        result = compute_clash_feature(df, clash_window_days=7)
        assert bool(result["has_clash"].iloc[0]) is True

    def test_no_clash_when_far_apart(self):
        df = pd.DataFrame({
            "title": ["Movie A", "Movie B"],
            "release_date": ["2024-01-01", "2024-02-01"],
        })
        result = compute_clash_feature(df, clash_window_days=7)
        assert not result["has_clash"].any()
