"""Tests for bankability score computation."""

import math

import numpy as np
import pytest

from tamasha.train_pipeline import compute_bankability_score, time_decay_weight


class TestTimeDecayWeight:
    """Tests for time_decay_weight."""

    def test_recent_film(self):
        # Film from 2024, reference 2024 → weight ≈ 1.0
        w = time_decay_weight(2024, 2024, halflife=3.0)
        assert w == pytest.approx(1.0, abs=0.01)

    def test_old_film(self):
        # Film from 2018, reference 2024 (6 years, 2 halflives) → weight ≈ 0.25
        w = time_decay_weight(2018, 2024, halflife=3.0)
        assert w == pytest.approx(0.25, abs=0.05)

    def test_half_life(self):
        # After one halflife, weight ≈ 0.5
        w = time_decay_weight(2021, 2024, halflife=3.0)
        assert w == pytest.approx(0.5, abs=0.05)

    def test_weight_range(self):
        for years_ago in range(20):
            w = time_decay_weight(2024 - years_ago, 2024, halflife=3.0)
            assert 0.0 <= w <= 1.0


class TestComputeBankabilityScore:
    """Tests for compute_bankability_score."""

    def test_basic_computation(self):
        scores = [8.0, 7.0, 6.0]
        years = [2024, 2023, 2022]
        ref_year = 2024
        result = compute_bankability_score(scores, years, ref_year)
        assert isinstance(result, float)
        assert 0.0 <= result <= 10.0

    def test_empty_scores(self):
        result = compute_bankability_score([], [], 2024)
        assert result == 0.0

    def test_single_film(self):
        result = compute_bankability_score([8.0], [2024], 2024)
        assert result == pytest.approx(8.0, abs=0.1)

    def test_recent_films_score_higher(self):
        recent = compute_bankability_score([8.0], [2024], 2024)
        old = compute_bankability_score([8.0], [2015], 2024)
        assert recent > old
