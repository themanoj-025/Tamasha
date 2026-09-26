"""
Pipeline Package — ML training pipeline split into focused modules.

Modules:
- data_loading: Data fetching, preprocessing, feature engineering
- model_training: Model training, hyperparameter tuning
- evaluation: Model evaluation, metrics, reporting
"""

from .data_loading import clean_datasets, enrich_with_tmdb, load_datasets, two_step_fuzzy_join
from .evaluation import ModelEvaluator
from .model_training import train_boxoffice, train_rating

__all__ = [
    "ModelEvaluator",
    "clean_datasets",
    "enrich_with_tmdb",
    "load_datasets",
    "train_boxoffice",
    "train_rating",
    "two_step_fuzzy_join",
]
