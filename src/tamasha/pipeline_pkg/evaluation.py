"""
Evaluation Module — Model performance evaluation and metrics.

Handles:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrix generation
- Model comparison and selection
- Report generation
- Performance visualization
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluates trained ML models and generates performance reports."""

    def __init__(self, model_name: str = "model"):
        """Initialize evaluator.

        Args:
            model_name: Name identifier for the model being evaluated.
        """
        self.model_name = model_name
        self.metrics: dict[str, Any] = {}
        self.confusion_mat: np.ndarray | None = None

    def evaluate_classification(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        average: str = "weighted",
    ) -> dict[str, float]:
        """Calculate classification metrics.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            average: Averaging method for multi-class (weighted, macro, micro).

        Returns:
            Dictionary with accuracy, precision, recall, f1 scores.
        """
        self.metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        }

        self.confusion_mat = confusion_matrix(y_true, y_pred)

        logger.info(
            "Model '%s' evaluation: acc=%.3f, prec=%.3f, rec=%.3f, f1=%.3f",
            self.model_name,
            self.metrics["accuracy"],
            self.metrics["precision"],
            self.metrics["recall"],
            self.metrics["f1"],
        )

        return self.metrics

    def get_confusion_matrix(self) -> np.ndarray | None:
        """Return the confusion matrix from last evaluation."""
        return self.confusion_mat

    def get_classification_report(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        target_names: list[str] | None = None,
    ) -> str:
        """Generate detailed classification report.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.
            target_names: Optional class names for the report.

        Returns:
            String representation of classification report.
        """
        return classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )

    def compare_models(
        self,
        results: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """Compare multiple models side by side.

        Args:
            results: Dict mapping model names to their metrics.

        Returns:
            DataFrame with models as rows and metrics as columns.
        """
        df = pd.DataFrame(results).T
        if "f1" in df.columns:
            df = df.sort_values("f1", ascending=False)
        return df

    def get_best_model(
        self,
        results: dict[str, dict[str, float]],
        metric: str = "f1",
    ) -> tuple[str, float]:
        """Find the best performing model.

        Args:
            results: Dict mapping model names to their metrics.
            metric: Which metric to use for comparison.

        Returns:
            Tuple of (model_name, metric_value).
        """
        best_name = max(results, key=lambda k: results[k].get(metric, 0))
        return best_name, results[best_name].get(metric, 0)

    def generate_report(self) -> dict[str, Any]:
        """Generate a summary report of evaluation results.

        Returns:
            Dictionary with model name, metrics, and confusion matrix.
        """
        return {
            "model_name": self.model_name,
            "metrics": self.metrics.copy(),
            "confusion_matrix": self.confusion_mat.tolist() if self.confusion_mat is not None else None,
        }
