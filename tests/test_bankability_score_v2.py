"""Tests for bankability score computation."""


import pytest

from tamasha.network.bankability_score import _decay_weight, compute_bankability_scores


class TestTimeDecayWeight:
    """Tests for _decay_weight (exponential time-decay)."""

    def test_recent_film(self) -> None:
        # Film from 2024, reference 2024 → weight ≈ 1.0
        w = _decay_weight(2024, 2024, half_life=3.0)
        assert w == pytest.approx(1.0, abs=0.01)

    def test_old_film(self) -> None:
        # Film from 2018, reference 2024 (6 years, 2 halflives) → weight ≈ 0.25
        w = _decay_weight(2018, 2024, half_life=3.0)
        assert w == pytest.approx(0.25, abs=0.05)

    def test_half_life(self) -> None:
        # After one halflife, weight ≈ 0.5
        w = _decay_weight(2021, 2024, half_life=3.0)
        assert w == pytest.approx(0.5, abs=0.05)

    def test_weight_range(self) -> None:
        """All weights should be between 0 and 1."""
        for year_offset in range(20):
            w = _decay_weight(2024 - year_offset, 2024, half_life=3.0)
            assert 0.0 <= w <= 1.0

    def test_future_film(self) -> None:
        """Film from the future should have weight > 1."""
        w = _decay_weight(2026, 2024, half_life=3.0)
        assert w > 1.0


class TestComputeBankabilityScores:
    """Tests for compute_bankability_scores (requires real data)."""

    def test_is_callable(self) -> None:
        """compute_bankability_scores should be a callable function."""
        assert callable(compute_bankability_scores)
