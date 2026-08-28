"""Tests for prediction service."""

import pytest
from unittest.mock import MagicMock, patch

from tamasha.predict import predict_bankability


class TestPredictBankability:
    """Tests for predict_bankability."""

    def test_predict_returns_dict(self):
        # Mock the model and scaler
        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0]
        mock_scaler = MagicMock()
        mock_scaler.transform.return_value = [[1.0, 2.0, 3.0]]

        result = predict_bankability(
            title="Test Movie",
            year=2024,
            model=mock_model,
            scaler=mock_scaler,
            feature_names=["f1", "f2", "f3"],
        )
        assert isinstance(result, dict)
        assert "bankability_score" in result
