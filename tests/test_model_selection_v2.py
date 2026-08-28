"""Tests for model selection logic."""

import numpy as np
import pytest

from tamasha.train_pipeline import select_best_model


class TestSelectBestModel:
    """Tests for select_best_model."""

    def test_returns_best_model(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=10, random_state=42),
        }
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        result = select_best_model(models, X, y, metric="MAE")
        assert result in models

    def test_different_metrics(self):
        from sklearn.linear_model import LinearRegression

        models = {"LR": LinearRegression()}
        X = np.random.rand(100, 5)
        y = np.random.rand(100)
        for metric in ["MAE", "RMSE", "R2"]:
            result = select_best_model(models, X, y, metric=metric)
            assert result == "LR"
