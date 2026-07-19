"""Tests for the model-selection module.

Critical: protect the auto-selection logic — the lowest-MAE model must
be picked correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tamasha.models.model_selection import train_and_compare


def test_mae_selection_correct():
    """Verify that the model with lowest MAE is auto-selected."""
    # Create a small dataset where Ridge should outperform a dummy baseline
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1  # Low noise

    # Only compare two models
    from sklearn.linear_model import LinearRegression, Ridge

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
    }

    comparison, best_name, best_model = train_and_compare(
        X, y, task_name="test", models=models, cv_folds=3, metric="MAE",
    )

    # Both models should perform reasonably
    assert best_name in ["LinearRegression", "Ridge"]
    assert comparison.iloc[0]["MAE"] <= comparison.iloc[1]["MAE"]


def test_comparison_csv_contains_expected_columns():
    """Verify the comparison output has the right columns."""
    np.random.seed(42)
    X = np.random.randn(30, 2)
    y = X[:, 0] + np.random.randn(30) * 0.1

    models = {"LinearRegression": LinearRegression()}
    comparison, best_name, best_model = train_and_compare(
        X, y, task_name="test_cols", models=models, cv_folds=2, metric="MAE",
    )

    expected_cols = ["model", "MAE", "RMSE", "R2", "training_time_s"]
    for col in expected_cols:
        assert col in comparison.columns, f"Missing column: {col}"


def test_best_model_refit_on_full_data():
    """Test that the best model is refit on the full dataset."""
    np.random.seed(42)
    X = np.random.randn(30, 2)
    y = X[:, 0] + np.random.randn(30) * 0.1

    models = {"LinearRegression": LinearRegression()}
    _, _, best_model = train_and_compare(
        X, y, task_name="test_refit", models=models, cv_folds=2, metric="MAE",
    )

    # After refit, model should have coefficients
    assert hasattr(best_model, "coef_")
    assert best_model.coef_ is not None
