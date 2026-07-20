"""Thread-safety and edge-case tests for ``PredictionService``.

Verifies:
- Concurrent ``predict_rating()`` calls don't corrupt shared state.
- Graceful degradation when model files are missing.
- Bankability fallback for unknown actors.
- Empty / extreme inputs are handled without crashes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from tamasha.config import settings
from tamasha.predict import PredictionService


# ── Helpers ───────────────────────────────────────────────────────────

def _ensure_dummy_artifacts() -> Path:
    """Create minimal dummy model artifacts in the models dir.

    Returns the models dir path.
    """
    models_dir = settings.MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # Rating model + features
    rating_cols = [
        "genre_Action", "genre_Drama", "genre_Romance",
        "director_encoded", "cast_size", "runtime_minutes",
        "budget_inr", "decade_2020",
    ]
    model_r = DummyRegressor(strategy="constant", constant=7.0)
    X = np.zeros((1, len(rating_cols)))
    model_r.fit(X, np.array([7.0]))
    joblib.dump(model_r, models_dir / "best_rating_model.pkl")
    (models_dir / "rating_features.json").write_text(json.dumps(rating_cols))

    # Box office model + features
    box_cols = rating_cols + ["avg_bankability_score"]
    model_b = DummyRegressor(strategy="constant", constant=500_000_000.0)
    X2 = np.zeros((1, len(box_cols)))
    model_b.fit(X2, np.array([500_000_000.0]))
    joblib.dump(model_b, models_dir / "best_boxoffice_model.pkl")
    (models_dir / "boxoffice_features.json").write_text(json.dumps(box_cols))

    # Director encoder
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(["Director A", "Director B", "Director C"])
    joblib.dump(le, models_dir / "director_encoder.pkl")

    # Bankability scores
    bank_df = pd.DataFrame({
        "actor": ["Actor Known", "Actor AlsoKnown"],
        "type": ["actor", "actor"],
        "bankability_score": [1.5, 0.8],
        "film_count": [5, 3],
    })
    bank_df.to_csv(settings.REPORTS_DIR / "bankability_scores.csv", index=False)

    return models_dir


# ── Tests ─────────────────────────────────────────────────────────────

class TestPredictionService:
    """Tests for the PredictionService class."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        """Create dummy artifacts in a temp dir for each test."""
        # Use tmp_path as models dir so tests don't clobber real models
        self.models_dir = tmp_path / "models"
        self.reports_dir = tmp_path / "reports"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Temporarily override settings paths
        self._orig_models = settings.MODELS_DIR
        self._orig_reports = settings.REPORTS_DIR
        settings.MODELS_DIR = self.models_dir
        settings.REPORTS_DIR = self.reports_dir

        _ensure_dummy_artifacts()

        # Build service
        self.svc = PredictionService(models_dir=self.models_dir, reports_dir=self.reports_dir)
        self.svc.load()

    def teardown_method(self) -> None:
        """Restore original settings paths."""
        settings.MODELS_DIR = self._orig_models
        settings.REPORTS_DIR = self._orig_reports

    def test_healthy_when_models_present(self) -> None:
        assert self.svc.healthy is True

    def test_healthy_when_model_missing(self) -> None:
        """Remove rating model -> healthy should be False."""
        (self.models_dir / "best_rating_model.pkl").unlink()
        # Recreate service without the model
        svc2 = PredictionService(models_dir=self.models_dir, reports_dir=self.reports_dir)
        svc2.load()
        assert svc2.healthy is False

    def test_predict_rating_happy_path(self) -> None:
        result = self.svc.predict_rating(
            genres=["Drama", "Romance"],
            cast=["Actor Known", "Actor AlsoKnown"],
            director="Director A",
            budget_inr=50_000_000,
            runtime_minutes=150,
            year=2024,
        )
        assert result["predicted_rating"] is not None
        assert 0 <= result["predicted_rating"] <= 10
        assert "model_name" in result

    def test_predict_rating_without_model(self) -> None:
        """Should degrade gracefully, not crash."""
        svc = PredictionService(models_dir=self.models_dir, reports_dir=self.reports_dir)
        # Don't call load() -> no model loaded
        result = svc.predict_rating(
            genres=["Drama"], cast=["Actor A"],
            director="Dir", budget_inr=0, runtime_minutes=120, year=2024,
        )
        assert result["predicted_rating"] is None

    def test_predict_boxoffice_unknown_actor_fallback(self) -> None:
        """Unknown actors should not crash — fallback to mean score."""
        result = self.svc.predict_boxoffice(
            genres=["Action"],
            cast=["Nobody HasEverHeardOfMe"],
            director="Director A",
            budget_inr=100_000_000,
            runtime_minutes=140,
            year=2024,
        )
        assert result["predicted_boxoffice_cr"] is not None
        # Fallback should be noted
        assert result.get("fallback_actors", False) is True

    def test_predict_boxoffice_empty_cast(self) -> None:
        """Empty cast list should not crash."""
        result = self.svc.predict_boxoffice(
            genres=["Drama"],
            cast=[],
            director="Director A",
            budget_inr=50_000_000,
            runtime_minutes=120,
            year=2024,
        )
        assert result["predicted_boxoffice_cr"] is not None

    def test_predict_negative_budget(self) -> None:
        """Negative budget should not cause issues."""
        result = self.svc.predict_rating(
            genres=["Comedy"],
            cast=["Actor Known"],
            director="Director B",
            budget_inr=-100_000_000,  # negative
            runtime_minutes=120,
            year=2024,
        )
        assert result["predicted_rating"] is not None

    def test_unknown_genre_ignored(self) -> None:
        """Unknown genre string should be silently ignored."""
        result = self.svc.predict_rating(
            genres=["ThisGenreDoesNotExist"],
            cast=["Actor Known"],
            director="Director A",
            budget_inr=50_000_000,
            runtime_minutes=120,
            year=2024,
        )
        assert result["predicted_rating"] is not None

    def test_get_actor_info_known(self) -> None:
        info = self.svc.get_actor_info("Actor Known")
        assert info["found"] is True
        assert info["bankability_score"] == 1.5
        assert info["film_count"] == 5

    def test_get_actor_info_unknown(self) -> None:
        info = self.svc.get_actor_info("Nobody")
        assert info["found"] is False
        assert info["bankability_score"] is None

    def test_get_model_info(self) -> None:
        info = self.svc.get_model_info()
        assert "rating_model" in info
        assert "boxoffice_model" in info
        assert info["rating_model"]["algorithm"] == "DummyRegressor"

    def test_load_idempotent(self) -> None:
        """Calling load() twice should not raise or lose state."""
        self.svc.load()  # second call
        assert self.svc.healthy is True

    def test_load_after_unload(self) -> None:
        """Create a fresh service, don't load, verify degraded."""
        svc = PredictionService(models_dir=self.models_dir, reports_dir=self.reports_dir)
        assert svc.healthy is False


class TestConcurrency:
    """Thread-safety tests — the core guarantee of PredictionService."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.models_dir = tmp_path / "models_conc"
        self.reports_dir = tmp_path / "reports_conc"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self._orig_models = settings.MODELS_DIR
        self._orig_reports = settings.REPORTS_DIR
        settings.MODELS_DIR = self.models_dir
        settings.REPORTS_DIR = self.reports_dir

        _ensure_dummy_artifacts()

        self.svc = PredictionService(models_dir=self.models_dir, reports_dir=self.reports_dir)
        self.svc.load()

    def teardown_method(self) -> None:
        settings.MODELS_DIR = self._orig_models
        settings.REPORTS_DIR = self._orig_reports

    def test_20_concurrent_predictions_no_crash(self) -> None:
        """Fire 20 threads at predict_rating simultaneously — no crashes or state corruption."""

        results: list[dict] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _predict() -> None:
            try:
                r = self.svc.predict_rating(
                    genres=["Drama", "Action"],
                    cast=["Actor Known", "Actor AlsoKnown"],
                    director="Director A",
                    budget_inr=50_000_000,
                    runtime_minutes=150,
                    year=2024,
                )
                with lock:
                    results.append(r)
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=_predict) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent predictions failed: {errors}"
        assert len(results) == 20
        # All results should have the same predicted value (DummyRegressor)
        ratings = {r["predicted_rating"] for r in results if r["predicted_rating"] is not None}
        assert len(ratings) == 1, f"Expected same prediction from DummyRegressor, got: {ratings}"

    def test_concurrent_mixed_calls_no_leak(self) -> None:
        """Mix predict_rating and predict_boxoffice calls concurrently.

        Verify no state leaks between calls (e.g., boxoffice state
        polluting rating state).
        """
        errors: list[Exception] = []
        lock = threading.Lock()

        def _predict_rating() -> None:
            try:
                r = self.svc.predict_rating(
                    genres=["Drama"], cast=["Actor Known"],
                    director="Director A", budget_inr=50_000_000,
                    runtime_minutes=150, year=2024,
                )
                # Should have rating-specific fields, not boxoffice fields
                assert "predicted_rating" in r
                assert "predicted_boxoffice_cr" not in r
            except Exception as e:
                with lock:
                    errors.append(e)

        def _predict_boxoffice() -> None:
            try:
                r = self.svc.predict_boxoffice(
                    genres=["Action"], cast=["Actor Known"],
                    director="Director A", budget_inr=100_000_000,
                    runtime_minutes=140, year=2024,
                )
                assert "predicted_boxoffice_cr" in r
                assert "predicted_rating" not in r
            except Exception as e:
                with lock:
                    errors.append(e)

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=_predict_rating))
            threads.append(threading.Thread(target=_predict_boxoffice))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Mixed concurrent calls failed: {errors}"
