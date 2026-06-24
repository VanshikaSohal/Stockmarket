"""
Stacking ensemble models combining RF, XGBoost, LightGBM, CatBoost, and Ridge.

Uses scikit-learn's ``StackingClassifier`` and ``StackingRegressor`` with
a meta-learner trained via cross-validated predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# =============================================================================
# Base estimator factories
# =============================================================================


def _get_base_estimators(
    task: str,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs,
) -> List[Tuple[str, Any]]:
    """
    Return a list of (name, estimator) tuples for ensemble construction.

    Skips any model that fails to import (graceful degradation).
    """
    estimators = []

    # Random Forest
    if task == "classification":
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            n_jobs=n_jobs,
            **{k: v for k, v in kwargs.items() if k.startswith("rf_")},
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            n_jobs=n_jobs,
            **{k: v for k, v in kwargs.items() if k.startswith("rf_")},
        )
    estimators.append(("random_forest", rf))

    # XGBoost
    try:
        from xgboost import XGBClassifier, XGBRegressor

        if task == "classification":
            xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                eval_metric="logloss",
                verbosity=0,
                n_jobs=n_jobs,
            )
        else:
            xgb = XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbosity=0,
                n_jobs=n_jobs,
            )
        estimators.append(("xgboost", xgb))
    except ImportError:
        logger.warning("XGBoost not available, skipping.")

    # LightGBM
    try:
        import lightgbm as lgb

        if task == "classification":
            lgbm = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
                n_jobs=n_jobs,
            )
        else:
            lgbm = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=64,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
                n_jobs=n_jobs,
            )
        estimators.append(("lightgbm", lgbm))
    except ImportError:
        logger.warning("LightGBM not available, skipping.")

    # CatBoost
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        if task == "classification":
            cb = CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=random_state,
                verbose=0,
            )
        else:
            cb = CatBoostRegressor(
                iterations=200,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                random_seed=random_state,
                verbose=0,
            )
        estimators.append(("catboost", cb))
    except ImportError:
        logger.warning("CatBoost not available, skipping.")

    return estimators


# =============================================================================
# Stacking ensemble
# =============================================================================


def build_stacking_ensemble(
    task: str = "classification",
    meta_learner: Optional[Any] = None,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs,
):
    """
    Build a stacking ensemble of RF, XGBoost, LightGBM, and CatBoost.

    Args:
        task: ``"classification"`` or ``"regression"``.
        meta_learner: Estimator for the final meta-learner layer.
            Defaults to LogisticRegression (classification) or Ridge (regression).
        cv: Number of folds for the cross-validated predictions.
        random_state: Random seed.
        n_jobs: Number of parallel jobs.
        **kwargs: Additional kwargs passed to base estimators (prefix with rf_, etc.).

    Returns:
        ``StackingClassifier`` or ``StackingRegressor`` instance.

    Example:
        >>> ensemble = build_stacking_ensemble(task="classification")
        >>> ensemble.fit(X_train, y_train)
        >>> y_pred = ensemble.predict(X_test)
    """
    estimators = _get_base_estimators(task, random_state, n_jobs, **kwargs)

    if len(estimators) < 2:
        logger.warning(
            "Fewer than 2 base estimators available; using a single Random Forest."
        )
        if task == "classification":
            return RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=random_state, n_jobs=n_jobs
            )
        else:
            return RandomForestRegressor(
                n_estimators=200, max_depth=8, random_state=random_state, n_jobs=n_jobs
            )

    if meta_learner is None:
        if task == "classification":
            meta_learner = LogisticRegression(
                C=1.0, max_iter=1000, random_state=random_state, n_jobs=n_jobs
            )
        else:
            meta_learner = Ridge(alpha=1.0, random_state=random_state)

    if task == "classification":
        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv,
            n_jobs=n_jobs,
            passthrough=False,
            verbose=0,
        )
    else:
        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=cv,
            n_jobs=n_jobs,
            passthrough=False,
            verbose=0,
        )

    return ensemble


# =============================================================================
# Training wrapper
# =============================================================================


def train_stacking_ensemble(
    X,
    y,
    task: str = "classification",
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train a stacking ensemble and return evaluation metrics.

    Args:
        X: Feature matrix.
        y: Target vector.
        task: ``"classification"`` or ``"regression"``.
        test_size: Fraction held out for evaluation.
        random_state: Random seed.
        **kwargs: Passed to ``build_stacking_ensemble``.

    Returns:
        Dict with keys ``model``, ``metrics``, ``X_test``, ``y_test``, ``y_pred``.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        mean_absolute_error,
        r2_score,
    )
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = build_stacking_ensemble(
        task=task, random_state=random_state, **kwargs
    )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    if task == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
    else:
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }

    logger.info("Stacking ensemble (%s) metrics: %s", task, metrics)
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
