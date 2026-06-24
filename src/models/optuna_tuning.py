"""
Hyperparameter optimization with Optuna.

Optimizes Random Forest, XGBoost, LightGBM, and CatBoost models using
Bayesian search with optional pruning. Supports both classification and
regression tasks with purged walk-forward cross-validation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial

logger = logging.getLogger(__name__)

# =============================================================================
# Search space definitions per model
# =============================================================================


def _rf_params(trial: Trial) -> Dict[str, Any]:
    """Random Forest hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }


def _xgb_params(trial: Trial) -> Dict[str, Any]:
    """XGBoost hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }


def _lgbm_params(trial: Trial) -> Dict[str, Any]:
    """LightGBM hyperparameter search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 256),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }


def _catboost_params(trial: Trial) -> Dict[str, Any]:
    """CatBoost hyperparameter search space."""
    return {
        "iterations": trial.suggest_int("iterations", 50, 500, step=50),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
    }


# Map model names to their param-suggestion functions
_PARAM_SPACES = {
    "random_forest": _rf_params,
    "xgboost": _xgb_params,
    "lightgbm": _lgbm_params,
    "catboost": _catboost_params,
}


def _build_model(model_name: str, params: Dict[str, Any], task: str, random_state: int):
    """Instantiate a model with the given hyperparameters."""
    params = params.copy()
    params["random_state"] = random_state
    params["verbosity"] = 0

    if model_name == "random_forest":
        if task == "classification":
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(**params, n_jobs=-1)
        else:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(**params, n_jobs=-1)

    elif model_name == "xgboost":
        params.pop("verbosity", None)
        if task == "classification":
            from xgboost import XGBClassifier

            return XGBClassifier(
                **params, eval_metric="logloss", verbosity=0, n_jobs=-1
            )
        else:
            from xgboost import XGBRegressor

            return XGBRegressor(**params, verbosity=0, n_jobs=-1)

    elif model_name == "lightgbm":
        params.pop("verbosity", None)
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("Install lightgbm: pip install lightgbm")

        if task == "classification":
            return lgb.LGBMClassifier(**params, verbose=-1, n_jobs=-1)
        else:
            return lgb.LGBMRegressor(**params, verbose=-1, n_jobs=-1)

    elif model_name == "catboost":
        params.pop("verbosity", None)
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError:
            raise ImportError("Install catboost: pip install catboost")

        if task == "classification":
            return CatBoostClassifier(**params, verbose=0)
        else:
            return CatBoostRegressor(**params, verbose=0)

    else:
        raise ValueError(f"Unknown model: {model_name}")


# =============================================================================
# Objective function
# =============================================================================


def _objective(
    trial: Trial,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    cv_fn: Callable,
    random_state: int,
) -> float:
    """Optuna objective: returns a single scalar to maximize."""
    from sklearn.preprocessing import StandardScaler

    params = _PARAM_SPACES[model_name](trial)
    model = _build_model(model_name, params, task, random_state)

    scaler = StandardScaler()
    scores = []

    for train_idx, test_idx in cv_fn(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        if task == "classification":
            from sklearn.metrics import accuracy_score
            scores.append(accuracy_score(y_test, y_pred))
        else:
            from sklearn.metrics import mean_absolute_error
            scores.append(-mean_absolute_error(y_test, y_pred))

    if not scores:
        return 0.0
    return float(np.mean(scores))


# =============================================================================
# Public API
# =============================================================================


def optimize_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "xgboost",
    task: str = "classification",
    n_trials: int = 50,
    cv_fn: Optional[Callable] = None,
    random_state: int = 42,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    direction: str = "maximize",
    timeout: Optional[int] = None,
) -> optuna.Study:
    """
    Run a hyperparameter optimization study with Optuna.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: One of ``"random_forest"``, ``"xgboost"``, ``"lightgbm"``, ``"catboost"``.
        task: ``"classification"`` or ``"regression"``.
        n_trials: Number of optimisation trials.
        cv_fn: Callable that yields ``(train_idx, test_idx)`` tuples.
            Defaults to a simple 3-fold split.
        random_state: Random seed.
        study_name: Optional name for the study (for persistence).
        storage: Optional storage URL for the study.
        direction: ``"maximize"`` or ``"minimize"``.
        timeout: Stop study after this many seconds.

    Returns:
        ``optuna.Study`` object with ``best_params`` and ``best_value``.

    Example:
        >>> study = optimize_hyperparameters(
        ...     X, y, model_name="xgboost", task="classification", n_trials=30
        ... )
        >>> study.best_params
        {'n_estimators': 200, 'max_depth': 6, ...}
    """
    if cv_fn is None:

        def _default_cv(X):
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=3, shuffle=False)
            return kf.split(X)

        cv_fn = _default_cv

    if model_name not in _PARAM_SPACES:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(_PARAM_SPACES.keys())}"
        )

    sampler = TPESampler(seed=random_state, multivariate=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        sampler=sampler,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: _objective(
            trial, model_name, X, y, task, cv_fn, random_state
        ),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True,
    )

    logger.info(
        "Optuna %s/%s: best %s = %.4f",
        model_name,
        task,
        direction,
        study.best_value,
    )
    logger.info("Best params: %s", study.best_params)

    return study


def get_best_model(
    study: optuna.Study,
    model_name: str,
    task: str,
    random_state: int = 42,
) -> Any:
    """
    Build a model instance with the best hyperparameters from an Optuna study.
    """
    return _build_model(model_name, study.best_params, task, random_state)
