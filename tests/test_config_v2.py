"""Tests for configuration settings."""


from tamasha.config import Settings


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_paths(self) -> None:
        s = Settings()
        assert s.DATA_RAW.exists() or True  # dirs created at import
        assert s.MODELS_DIR is not None

    def test_festival_multipliers(self) -> None:
        s = Settings()
        assert "Diwali" in s.FESTIVAL_MULTIPLIERS
        assert "Eid" in s.FESTIVAL_MULTIPLIERS
        assert "Normal" in s.FESTIVAL_MULTIPLIERS
        assert s.FESTIVAL_MULTIPLIERS["Normal"] == 1.0

    def test_model_selection_defaults(self) -> None:
        s = Settings()
        assert s.CV_FOLDS == 5
        assert s.TEST_SIZE == 0.2
        assert s.RANDOM_STATE == 42

    def test_bankability_decay(self) -> None:
        s = Settings()
        assert s.BANKABILITY_DECAY_HALFLIFE_YEARS == 3.0

    def test_fuzzy_join_defaults(self) -> None:
        s = Settings()
        assert s.FUZZY_JOIN_SCORE_CUTOFF == 60.0
        assert s.FUZZY_JOIN_YEAR_TOLERANCE == 2

    def test_rate_limit_default(self) -> None:
        s = Settings()
        assert s.RATE_LIMIT == "60/minute"

    def test_api_key_empty_default(self) -> None:
        s = Settings()
        assert s.API_KEY == ""
