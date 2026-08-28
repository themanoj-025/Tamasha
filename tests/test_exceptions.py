"""Tests for custom exceptions."""

import pytest

from tamasha.exceptions import (
    ModelNotFoundError,
    PredictionError,
    DataValidationError,
    CacheError,
    EnrichmentError,
)


class TestCustomExceptions:
    """Tests for custom exception classes."""

    def test_model_not_found(self):
        exc = ModelNotFoundError("model.joblib")
        assert "model.joblib" in str(exc)
        assert isinstance(exc, Exception)

    def test_prediction_error(self):
        exc = PredictionError("prediction failed")
        assert "prediction failed" in str(exc)

    def test_data_validation_error(self):
        exc = DataValidationError("invalid data")
        assert "invalid data" in str(exc)

    def test_cache_error(self):
        exc = CacheError("cache miss")
        assert "cache miss" in str(exc)

    def test_enrichment_error(self):
        exc = EnrichmentError("TMDB failed")
        assert "TMDB failed" in str(exc)

    def test_all_are_exception_subclasses(self):
        for cls in [ModelNotFoundError, PredictionError, DataValidationError, CacheError, EnrichmentError]:
            assert issubclass(cls, Exception)
