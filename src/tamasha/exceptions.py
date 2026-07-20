"""Custom exception types for the Tamasha project."""

from __future__ import annotations


class ModelIntegrityError(Exception):
    """Raised when a model artifact fails SHA-256 integrity verification.

    This means the file on disk does not match the hash recorded in the
    corresponding ``metadata.json``. The model should NOT be loaded.
    """

    def __init__(
        self,
        artifact_path: str,
        expected_hash: str,
        actual_hash: str,
    ) -> None:
        self.artifact_path = artifact_path
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Integrity check failed for {artifact_path}: "
            f"expected {expected_hash[:16]}…, got {actual_hash[:16]}…"
        )
