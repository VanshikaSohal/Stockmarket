"""
Model interpretability using SHAP (SHapley Additive exPlanations).

Provides utilities to explain predictions of any trained model,
produce beeswarm, dependence, force, and waterfall plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import shap
except ImportError:
    shap = None  # graceful degradation

logger = logging.getLogger(__name__)


# =============================================================================
# Explainer factory
# =============================================================================


def create_explainer(
    model: Any,
    X_background: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    method: str = "auto",
    nsamples: int = 100,
):
    """
    Create a SHAP explainer appropriate for the given model type.

    Args:
        model: A fitted model (sklearn, xgboost, lightgbm, catboost, pytorch).
        X_background: Background dataset for TreeSHAP / KernelSHAP.
            If None and model is tree-based, a small random sample is used.
        feature_names: Names for each feature column.
        method: ``"auto"``, ``"tree"``, ``"kernel"``, or ``"deep"``.
        nsamples: Number of samples for KernelSHAP.

    Returns:
        A SHAP ``Explainer`` object.

    Raises:
        ImportError: If SHAP is not installed.
    """
    if shap is None:
        raise ImportError(
            "SHAP is required. Install with: pip install shap"
        )

    if method == "auto":
        # Auto-detect based on model type
        _model_type = type(model).__name__.lower()
        if any(t in _model_type for t in ["xgb", "lgbm", "catboost", "randomforest", "gradientboosting", "tree"]):
            method = "tree"
        elif any(t in _model_type for t in ["lstm", "gru", "transformer", "sequential", "module"]):
            method = "deep"
        else:
            method = "kernel"

    if method == "tree":
        if X_background is None and hasattr(model, "feature_importances_"):
            X_background = np.random.randn(100, len(model.feature_importances_))
        explainer = shap.TreeExplainer(model)
    elif method == "deep":
        explainer = shap.DeepExplainer(model, data=X_background)
    elif method == "kernel":
        if X_background is None:
            raise ValueError("KernelSHAP requires a background dataset (X_background).")
        explainer = shap.KernelExplainer(
            model.predict if hasattr(model, "predict") else model,
            X_background,
            nsamples=nsamples,
        )
    else:
        raise ValueError(f"Unknown SHAP method: {method}. Use 'tree', 'kernel', 'deep', or 'auto'.")

    if feature_names is not None:
        explainer.feature_names = feature_names

    return explainer


# =============================================================================
# Plotting utilities
# =============================================================================


def plot_shap_summary(
    shap_values: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    max_display: int = 20,
    show: bool = False,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Beeswarm summary plot of SHAP values.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=show,
        ax=ax,
    )
    plt.tight_layout()
    return fig


def plot_shap_dependence(
    shap_values: Any,
    X: np.ndarray,
    feature_idx: int,
    interaction_idx: Optional[str] = "auto",
    feature_names: Optional[List[str]] = None,
    show: bool = False,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """
    SHAP dependence plot for a single feature.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    shap.dependence_plot(
        feature_idx,
        shap_values,
        X,
        interaction_index=interaction_idx,
        feature_names=feature_names,
        show=show,
        ax=ax,
    )
    plt.tight_layout()
    return fig


def plot_shap_force(
    explainer: Any,
    X_instance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    matplotlib: bool = True,
    figsize: tuple = (12, 3),
) -> plt.Figure:
    """
    Force plot for a single prediction.

    Returns:
        Matplotlib figure (if matplotlib=True) or raw SHAP force plot object.
    """
    shap_values = explainer.shap_values(X_instance)
    if matplotlib:
        fig, ax = plt.subplots(figsize=figsize)
        shap.force_plot(
            explainer.expected_value,
            shap_values[0] if shap_values.ndim > 1 else shap_values,
            X_instance,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
            ax=ax,
        )
        plt.tight_layout()
        return fig
    return shap.force_plot(
        explainer.expected_value,
        shap_values,
        X_instance,
        feature_names=feature_names,
    )


def plot_shap_waterfall(
    shap_values: Any,
    X_instance: pd.DataFrame,
    max_display: int = 10,
    show: bool = False,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Waterfall plot showing how each feature contributes to the prediction.
    """
    fig, ax = plt.subplots(figsize=figsize)
    shap.plots.waterfall(
        shap_values,
        max_display=max_display,
        show=show,
    )
    plt.tight_layout()
    return fig


def explain_model(
    model: Any,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    X_background: Optional[np.ndarray] = None,
    method: str = "auto",
    max_display: int = 20,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full model explanation: computes SHAP values and produces summary plots.

    Args:
        model: Fitted model.
        X: Feature matrix to explain.
        feature_names: Feature column names.
        X_background: Background dataset.
        method: SHAP method.
        max_display: Max features in summary plot.
        output_dir: If given, saves plots to this directory.

    Returns:
        Dict with keys:
          - ``explainer``: the SHAP explainer object.
          - ``shap_values``: computed SHAP values.
          - ``expected_value``: base (expected) value.
          - ``summary_fig``: matplotlib figure for summary plot.
    """
    explainer = create_explainer(
        model, X_background=X_background, feature_names=feature_names, method=method
    )

    shap_values = explainer.shap_values(X)

    expected = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, list) and len(explainer.expected_value) > 1
        else explainer.expected_value
    )

    fig = plot_shap_summary(shap_values, X, feature_names, max_display)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / "shap_summary.png", dpi=150, bbox_inches="tight")

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "expected_value": expected,
        "summary_fig": fig,
    }
