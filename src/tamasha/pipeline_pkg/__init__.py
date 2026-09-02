"""
Pipeline Package — ML training pipeline split into focused modules.

Modules:
- data_loading: Data fetching, preprocessing, feature engineering
- model_training: Model training, hyperparameter tuning
- evaluation: Model evaluation, metrics, reporting
"""

from .data_loading import engineer_features, load_and_preprocess
from .evaluation import ModelEvaluator, evaluate_model
from .model_training import train_model, tune_hyperparameters

__all__ = [
    "ModelEvaluator",
    "engineer_features",
    "evaluate_model",
    "load_and_preprocess",
    "train_model",
    "tune_hyperparameters",
]
