"""Tests for prediction service."""


from tamasha.predict import (
    get_bankability_scores,
    get_comparison_csv,
    get_model_info,
    predict_boxoffice,
    predict_rating,
)


class TestPredictFunctionsExist:
    """Smoke tests — verify all public functions exist and are callable."""

    def test_get_bankability_scores_is_callable(self) -> None:
        assert callable(get_bankability_scores)

    def test_get_model_info_is_callable(self) -> None:
        assert callable(get_model_info)

    def test_get_comparison_csv_is_callable(self) -> None:
        assert callable(get_comparison_csv)

    def test_predict_rating_is_callable(self) -> None:
        assert callable(predict_rating)

    def test_predict_boxoffice_is_callable(self) -> None:
        assert callable(predict_boxoffice)
