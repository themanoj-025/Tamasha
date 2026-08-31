"""Tests for model selection logic."""

from tamasha.models.model_selection import get_all_models


class TestGetAllModels:
    """Tests for get_all_models."""

    def test_returns_dict(self) -> None:
        """get_all_models should return a dict of model name → model instance."""
        models = get_all_models()
        assert isinstance(models, dict)

    def test_has_models(self) -> None:
        """Should return at least one model."""
        models = get_all_models()
        assert len(models) > 0

    def test_keys_are_strings(self) -> None:
        """All keys should be model name strings."""
        models = get_all_models()
        for name in models:
            assert isinstance(name, str)

    def test_values_are_sklearn_compatible(self) -> None:
        """All values should have a predict method (sklearn-compatible)."""
        models = get_all_models()
        for name, model in models.items():
            assert hasattr(model, "predict"), f"{name} has no predict method"
