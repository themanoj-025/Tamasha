"""Tests for festival calendar utility."""

import pytest

from tamasha.train_pipeline import get_festival_multiplier, get_nearest_festival


class TestGetFestivalMultiplier:
    """Tests for get_festival_multiplier."""

    def test_normal_release(self):
        mult = get_festival_multiplier("2024-06-15")
        assert mult >= 1.0

    def test_diwali_window(self):
        # Diwali 2024 is around Nov 1
        mult = get_festival_multiplier("2024-11-01")
        assert mult >= 1.0


class TestGetNearestFestival:
    """Tests for get_nearest_festival."""

    def test_returns_string(self):
        result = get_nearest_festival("2024-01-01")
        assert isinstance(result, str)

    def test_diwali_period(self):
        result = get_nearest_festival("2024-11-01")
        assert result.lower() in ["diwali", "normal"]
