"""
MLflow experiment tracking integration.

Provides context managers and utility functions to auto-log parameters,
metrics, and model artifacts for the portfolio risk pipeline.

Usage:
    with mlflow_run("rf_classification", tags={"task": "direction"}):
        model.fit(X, y)
        mlflow.log_metric("accuracy", 0.85)

Or use the decorator:
    @auto_log("my_experiment")
    def train_model(X, y):
        ...
        return {"accuracy": 0.85}
"""

from __future__ import annotations

import logging
import os
import tempfile
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mlflow
import mlflow.sklearn
import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_TRACKING_URI = os.getenv(
    "PRA_MLFLOW_URI",
    os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"),
)


# =============================================================================
# Configuration
# =============================================================================


def configure_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "portfolio_risk_analyzer",
    nested: bool = True,
) -> None:
    """
    Configure the MLflow tracking server and active experiment.

    Args:
        tracking_uri: URI for the tracking server (file:/, http://, databricks).
            Defaults to ``PRA_MLFLOW_URI`` env var or ``./mlruns``.
        experiment_name: Name of the MLflow experiment.
        nested: Allow nested runs.
    """
    mlflow.set_tracking_uri(tracking_uri or _DEFAULT_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    if nested:
        mlflow.set_nested_experiment(experiment_name)


# =============================================================================
# Context manager
# =============================================================================


@contextmanager
def mlflow_run(
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = True,
):
    """
    Context manager that wraps code in an MLflow run.

    Args:
        run_name: Name for this run (auto-generated if None).
        tags: Dict of tags to attach to the run.
        nested: Allow this run to be nested inside another active run.

    Yields:
        active ``mlflow.ActiveRun`` object.
    """
    if nested and mlflow.active_run():
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run
    else:
        with mlflow.start_run(run_name=run_name) as run:
            if tags:
                mlflow.set_tags(tags)
            yield run


# =============================================================================
# Decorator
# =============================================================================


def auto_log(experiment_name: str = None):
    """
    Decorator that auto-logs function parameters (if dict-like return) as MLflow metrics.

    The decorated function should return a dict of metric_name -> value,
    OR a dict with keys ``metrics``, ``params``, ``model``, ``artifacts``.

    Usage:
        @auto_log("rf_tuning")
        def train(X, y, lr=0.1):
            model = ...
            return {"accuracy": 0.85, "model": model}
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if experiment_name:
                mlflow.set_experiment(experiment_name)

            with mlflow_start_run():
                mlflow.log_params(
                    {k: str(v) for k, v in kwargs.items() if _is_loggable(v)}
                )

                result = func(*args, **kwargs)

                if isinstance(result, dict):
                    _log_result_dict(result)

                return result

        return wrapper

    return decorator


# =============================================================================
# Logging helpers
# =============================================================================


def log_sklearn_model(model, artifact_path: str = "model") -> None:
    """
    Log a scikit-learn compatible model to the current MLflow run.
    """
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)


def log_metrics_from_dict(
    metrics: Dict[str, float], step: Optional[int] = None
) -> None:
    """
    Log a flat dict of metric_name -> numeric value.
    """
    sanitized = {k: float(v) for k, v in metrics.items() if _is_loggable(v)}
    mlflow.log_metrics(sanitized, step=step)


def log_params_from_dict(params: Dict[str, Any]) -> None:
    """
    Log a flat dict of param_name -> value.
    """
    sanitized = {k: str(v) for k, v in params.items() if _is_loggable(v)}
    mlflow.log_params(sanitized)


def log_dataframe(df: pd.DataFrame, artifact_name: str = "data") -> None:
    """
    Log a pandas DataFrame as a CSV artifact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{artifact_name}.csv")
        df.to_csv(path, index=False)
        mlflow.log_artifact(path)


def log_figure(fig, artifact_name: str = "figure") -> None:
    """
    Log a matplotlib figure as a PNG artifact.
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"{artifact_name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        mlflow.log_artifact(path)


# =============================================================================
# Internal helpers
# =============================================================================


def _is_loggable(value: Any) -> bool:
    """Check if value is MLflow-loggable (primitive or None)."""
    if value is None:
        return False
    return isinstance(value, (str, int, float, bool))


def _log_result_dict(result: Dict[str, Any]) -> None:
    """Parse a result dict and log all MLflow-trackable components."""
    if "metrics" in result and isinstance(result["metrics"], dict):
        log_metrics_from_dict(result["metrics"])

    if "params" in result and isinstance(result["params"], dict):
        log_params_from_dict(result["params"])

    if "model" in result and result["model"] is not None:
        try:
            log_sklearn_model(result["model"])
        except Exception as exc:
            logger.warning("Could not log model to MLflow: %s", exc)

    if "artifacts" in result and isinstance(result["artifacts"], dict):
        for name, artifact in result["artifacts"].items():
            if isinstance(artifact, pd.DataFrame):
                log_dataframe(artifact, artifact_name=name)
            elif hasattr(artifact, "savefig"):
                log_figure(artifact, artifact_name=name)


# Convenience alias
mlflow_start_run = mlflow_run
