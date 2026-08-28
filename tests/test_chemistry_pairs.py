"""Tests for chemistry pairs feature engineering."""

import pytest

from tamasha.train_pipeline import compute_chemistry_score


class TestComputeChemistryScore:
    """Tests for compute_chemistry_score."""

    def test_same_actor_zero(self):
        score = compute_chemistry_score("Actor A", "Actor A")
        assert score == 0.0

    def test_different_actors_positive(self):
        score = compute_chemistry_score("Shah Rukh Khan", "Salman Khan")
        assert score >= 0.0

    def test_empty_names(self):
        score = compute_chemistry_score("", "")
        assert score == 0.0
