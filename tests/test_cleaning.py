"""Tests for the data-cleaning module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tamasha.data.cleaning import clean_joined_dataset, inflation_adjust, parse_inr_value


class TestParseInrValue:
    """Tests for parse_inr_value."""

    def test_crore_format(self):
        assert parse_inr_value("₹100 Cr") == 100 * 1e7

    def test_lakh_format(self):
        assert parse_inr_value("Rs. 50 lakh") == 50 * 1e5

    def test_crore_abbreviation(self):
        assert parse_inr_value("₹25 cr") == 25 * 1e7

    def test_billion_format(self):
        assert parse_inr_value("1.5 billion") == 1.5 * 1e9

    def test_no_currency_symbol(self):
        assert parse_inr_value("100 crore") == 100 * 1e7

    def test_nan_input(self):
        assert parse_inr_value(np.nan) is None

    def test_empty_string(self):
        assert parse_inr_value("") is None

    def test_unparseable(self):
        assert parse_inr_value("N/A") is None


class TestInflationAdjust:
    """Tests for inflation_adjust."""

    def test_adjust_older_year(self):
        df = pd.DataFrame({"value": [100_000_000], "year": [2014]})
        result = inflation_adjust(df, "value", "year", base_year=2024)
        # 2024 - 2014 = 10 years, deflator = 1 + 0.06*10 = 1.6
        # Adjusted = 100M / 1.6 ≈ 62.5M
        assert abs(result.iloc[0] - 62_500_000) < 1_000

    def test_current_year_no_adjustment(self):
        df = pd.DataFrame({"value": [100_000_000], "year": [2024]})
        result = inflation_adjust(df, "value", "year", base_year=2024)
        assert abs(result.iloc[0] - 100_000_000) < 1


class TestCleanJoinedDataset:
    """Tests for clean_joined_dataset."""

    def test_drops_all_nan_columns(self):
        df = pd.DataFrame(
            {
                "title": ["A", "B"],
                "all_nan": [np.nan, np.nan],
                "rating": [7.0, 8.0],
            }
        )
        result = clean_joined_dataset(df)
        assert "all_nan" not in result.columns

    def test_drops_duplicate_rows(self):
        df = pd.DataFrame(
            {
                "title": ["A", "A", "B"],
                "rating": [7.0, 7.0, 8.0],
            }
        )
        result = clean_joined_dataset(df)
        assert len(result) == 2
