"""Custom evaluation metrics and visualization helpers.

Provides functions to generate:
- Grouped bar charts comparing MAE/RMSE across models
- Predicted-vs-actual scatter plots for top models
- SHAP summary plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for saving figures

logger = logging.getLogger(__name__)

# Try importing optional viz libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


def plot_model_comparison(
    comparison_csv: Union[str, Path],
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[Any]:
    """Generate a grouped bar chart comparing MAE/RMSE across models.

    Parameters
    ----------
    comparison_csv : str or Path
        Path to the model comparison CSV.
    save_path : str or Path, optional
        Path to save the figure (PNG/HTML).  If not provided, the
        figure is displayed.

    Returns
    -------
    matplotlib.figure.Figure or plotly.Figure or None
        The figure object if Plotly is available; otherwise None.
    """
    df = pd.read_csv(comparison_csv)

    if _HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="MAE",
                x=df["model"],
                y=df["MAE"],
                error_y=dict(type="data", array=df.get("MAE_std", None)),
            )
        )
        fig.add_trace(
            go.Bar(
                name="RMSE",
                x=df["model"],
                y=df["RMSE"],
                error_y=dict(type="data", array=df.get("RMSE_std", None)),
            )
        )
        fig.update_layout(
            title="Model Comparison: MAE & RMSE",
            xaxis_title="Model",
            yaxis_title="Error",
            barmode="group",
            template="plotly_dark",
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix == ".html":
                fig.write_html(str(save_path))
            else:
                fig.write_image(str(save_path))
            logger.info("Comparison chart saved to %s", save_path)

        return fig
    else:
        # Fallback to matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width / 2, df["MAE"], width, label="MAE")
        ax.bar(x + width / 2, df["RMSE"], width, label="RMSE")
        ax.set_xlabel("Model")
        ax.set_ylabel("Error")
        ax.set_title("Model Comparison: MAE & RMSE")
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"], rotation=45, ha="right")
        ax.legend()
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path))
            logger.info("Comparison chart saved to %s", save_path)

        plt.close(fig)
        return fig


def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[Any]:
    """Generate a predicted-vs-actual scatter plot.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    model_name : str
        Model name for the title.
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or plotly.Figure or None
    """
    if _HAS_PLOTLY:
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            labels={"x": "Actual", "y": "Predicted"},
            title=f"{model_name}: Predicted vs Actual",
            template="plotly_dark",
        )
        # Add perfect-prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            )
        )

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.suffix == ".html":
                fig.write_html(str(save_path))
            else:
                fig.write_image(str(save_path))

        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.7)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{model_name}: Predicted vs Actual")
        ax.set_aspect("equal")
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            fig.savefig(str(save_path))

        plt.close(fig)
        return fig


def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[Any]:
    """Generate a SHAP summary plot for a model.

    Parameters
    ----------
    model : Any
        Trained model (must be compatible with SHAP).
    X : pd.DataFrame
        Feature matrix.
    save_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _HAS_SHAP:
        logger.warning("SHAP not installed. Skipping SHAP plot.")
        return None

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X, check_additivity=False)

        fig = shap.summary_plot(shap_values, X, show=False)
        fig = plt.gcf()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(save_path), bbox_inches="tight")
            logger.info("SHAP summary plot saved to %s", save_path)

        plt.close(fig)
        return fig
    except Exception as exc:
        logger.warning("SHAP plot failed: %s", exc)
        return None
