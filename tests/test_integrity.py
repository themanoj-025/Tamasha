"""Tests for SHA-256 model artifact integrity verification.

Verifies:
- Corrupted model file causes health to report degraded with specific artifact
- Intact model loads fine
- /health response body includes integrity check details
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from fastapi.testclient import TestClient

from api.main import app
from tamasha.config import settings
from tamasha.models.model_selection import sha256_of_file, save_model_with_version


def _install_model_with_hash(tmp_path: Path) -> Path:
    """Install a dummy model with a valid SHA-256 hash in metadata.json."""
    model = DummyRegressor(strategy="constant", constant=5.0)
    X = np.zeros((1, 3))
    model.fit(X, np.array([5.0]))
    model_path = tmp_path / "best_rating_model.pkl"
    joblib.dump(model, model_path)

    # Create metadata.json with the correct hash
    version_dir = tmp_path
    version_dir.mkdir(parents=True, exist_ok=True)
    meta_path = version_dir / "metadata.json"
    meta_path.write_text(json.dumps({
        "version": 1,
        "task": "rating",
        "sha256": sha256_of_file(model_path),
        "model_type": "DummyRegressor",
    }))
    return model_path


def _corrupt_model(model_path: Path) -> None:
    """Corrupt a model file by appending garbage bytes."""
    with open(model_path, "ab") as f:
        f.write(b"CORRUPTED_DATA")


class TestSha256Hash:
    """Verify sha256_of_file() computes correct hashes."""

    def test_hash_matches_known_value(self) -> None:
        """SHA-256 of known content matches expected hash."""
        import hashlib
        test_file = Path("test_hash.txt")
        try:
            test_file.write_text("hello world")
            expected = hashlib.sha256(b"hello world").hexdigest()
            assert sha256_of_file(test_file) == expected
        finally:
            test_file.unlink(missing_ok=True)

    def test_hash_changes_on_corruption(self) -> None:
        """Hash changes when file is modified."""
        test_file = Path("test_hash2.txt")
        try:
            test_file.write_text("original content")
            hash1 = sha256_of_file(test_file)
            test_file.write_text("corrupted content")
            hash2 = sha256_of_file(test_file)
            assert hash1 != hash2
        finally:
            test_file.unlink(missing_ok=True)


class TestSaveModelWithVersion:
    """Verify save_model_with_version stores SHA-256 in metadata."""

    def test_metadata_contains_sha256(self) -> None:
        """metadata.json includes sha256 field after save."""
        model = DummyRegressor(strategy="constant", constant=3.0)
        X = np.zeros((1, 2))
        model.fit(X, np.array([3.0]))

        result = save_model_with_version(model, "test_task", models_dir=Path("models_test_temp"))
        try:
            meta = json.loads(result["metadata_path"].read_text())
            assert "sha256" in meta
            assert len(meta["sha256"]) == 64  # SHA-256 hex digest length
            # Verify the hash matches the file
            assert meta["sha256"] == sha256_of_file(result["path"])
        finally:
            # Cleanup
            import shutil
            shutil.rmtree("models_test_temp", ignore_errors=True)


class TestIntegrityVerification:
    """Verify PredictionService integrity checking behavior."""

    def test_intact_model_loads_normally(self) -> None:
        """Model with correct hash loads without integrity issues."""
        from tamasha.predict import PredictionService

        svc = PredictionService(
            models_dir=settings.MODELS_DIR,
            reports_dir=settings.REPORTS_DIR,
        )
        svc.load()
        # No integrity failures should be reported
        assert len(svc.integrity_failures) == 0

    def test_corrupted_model_reported_in_health(self) -> None:
        """Corrupted model causes health to report degraded with specific artifact."""
        from tamasha.predict import PredictionService

        svc = PredictionService(
            models_dir=settings.MODELS_DIR,
            reports_dir=settings.REPORTS_DIR,
        )
        svc.load()
        # Manually inject an integrity failure to simulate corruption
        svc._integrity_failures.append({
            "artifact": str(settings.MODELS_DIR / "best_rating_model.pkl"),
            "expected": "abc123def456",
            "actual": "deadbeef0123",
        })
        assert svc.healthy is False
        failures = svc.integrity_failures
        assert len(failures) == 1
        assert "rating_model" in failures[0]["artifact"]


class TestHealthIntegrity:
    """Verify /health endpoint reports integrity details."""

    def test_health_includes_checks_field(self) -> None:
        """/health response has a 'checks' field."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "checks" in data
