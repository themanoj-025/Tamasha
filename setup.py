"""Setup script for tamasha package.

Usage:
    pip install -e .             # base install (no extras)
    pip install -e .[ml]         # + ML-heavy models (xgboost, shap, ...)
    pip install -e .[dev]        # + dev tooling (pytest, pre-commit, ...)
    pip install -e .[all]        # everything
"""

from setuptools import find_packages, setup

EXTRAS_ML = [
    "xgboost>=2.0,<3.0",
    "lightgbm>=4.0,<5.0",
    "catboost>=1.2,<2.0",
    "shap>=0.42,<1.0",
]

EXTRAS_DEV = [
    "pytest>=7.4,<8.0",
    "pytest-cov>=4.1,<5.0",
    "httpx>=0.24,<1.0",
    "hypothesis>=6.0,<7.0",
    "pre-commit>=3.5,<4.0",
    "black>=23.11,<24.0",
    "isort>=5.12,<6.0",
    "ruff>=0.1,<1.0",
    "structlog>=24.0,<25.0",
    "prometheus-fastapi-instrumentator>=6.0,<7.0",
    "pip-audit>=2.0,<3.0",
]

setup(
    name="tamasha",
    version="0.1.0",
    description="Bollywood movie intelligence platform",
    author="Tamasha Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    extras_require={
        "ml": EXTRAS_ML,
        "dev": EXTRAS_DEV,
        "all": EXTRAS_ML + EXTRAS_DEV,
    },
)
