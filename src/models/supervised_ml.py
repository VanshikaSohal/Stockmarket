"""
Supervised ML models for return/direction forecasting: Random Forest and XGBoost.

Extended with MLflow auto-logging wrappers and additional model support
(LightGBM, CatBoost) for ensemble and Optuna integration.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

logger = logging.getLogger(__name__)


def build_features_and_target(returns, ticker, lags=5, target="direction"):
    """
    Build a supervised learning dataset from a return series.

    Args:
        returns (pd.DataFrame): Return DataFrame with tickers as columns.
        ticker (str): Target ticker to forecast.
        lags (int): Number of lag features to create. Default 5.
        target (str): 'direction' for binary classification (up/down),
            or 'return' for regression. Default 'direction'.

    Returns:
        tuple: (X, y, feature_names) where X is a numpy array, y is a numpy array,
            and feature_names is a list of column names.
    """
    df = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        for lag in range(1, lags + 1):
            df[f"{col}_lag{lag}"] = returns[col].shift(lag)

    if target == "direction":
        df["target"] = (returns[ticker] > 0).astype(int)
    else:
        df["target"] = returns[ticker]

    df = df.dropna()
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values
    return X, y, feature_cols


def train_random_forest(X, y, task="classification", test_size=0.2, random_state=42):
    """
    Train a Random Forest model.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        task (str): 'classification' or 'regression'. Default 'classification'.
        test_size (float): Proportion held out for evaluation. Default 0.2.
        random_state (int): Random seed. Default 42.

    Returns:
        dict: Contains keys 'model', 'scaler', 'metrics', 'X_test', 'y_test',
            'y_pred'.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task == "classification":
        model = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
    else:
        model = RandomForestRegressor(
            n_estimators=50, max_depth=5, random_state=random_state, n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    logger.info("Random Forest (%s) metrics: %s", task, metrics)
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def train_xgboost(X, y, task="classification", test_size=0.2, random_state=42):
    """
    Train an XGBoost model.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        task (str): 'classification' or 'regression'. Default 'classification'.
        test_size (float): Proportion held out for evaluation. Default 0.2.
        random_state (int): Random seed. Default 42.

    Returns:
        dict: Contains keys 'model', 'scaler', 'metrics', 'X_test', 'y_test',
            'y_pred'.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task == "classification":
        model = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
    else:
        model = XGBRegressor(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    logger.info("XGBoost (%s) metrics: %s", task, metrics)
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def get_feature_importances(result, feature_names):
    """
    Extract feature importances from a trained model result dictionary.

    Args:
        result (dict): Result dict as returned by train_random_forest or train_xgboost.
        feature_names (list of str): Feature names corresponding to columns.

    Returns:
        pd.Series: Feature importances sorted descending, indexed by feature name.
    """
    importances = result["model"].feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# =============================================================================
# MLflow-auto-logged wrappers (additive layer — original functions unchanged)
# =============================================================================


def train_random_forest_with_mlflow(
    X, y, task="classification", test_size=0.2, random_state=42, experiment_name=None
):
    """
    Train a Random Forest with MLflow auto-logging of params, metrics, and model.

    Args: same as :func:`train_random_forest`.

    Returns:
        dict: Same as :func:`train_random_forest` but the run is also logged to MLflow.
    """
    try:
        from src.models.mlflow_tracking import mlflow_run, log_sklearn_model, log_metrics_from_dict
        from sklearn.model_selection import train_test_split

        with mlflow_run(
            run_name=f"rf_{task}_{random_state}",
            tags={"model": "random_forest", "task": task},
        ):
            import mlflow

            mlflow.log_params({
                "task": task,
                "test_size": test_size,
                "random_state": random_state,
            })
            result = train_random_forest(X, y, task=task, test_size=test_size, random_state=random_state)
            log_metrics_from_dict(result["metrics"])
            log_sklearn_model(result["model"])
            return result
    except ImportError:
        logger.warning("MLflow not available, falling back to standard training.")
        return train_random_forest(X, y, task=task, test_size=test_size, random_state=random_state)


def train_xgboost_with_mlflow(
    X, y, task="classification", test_size=0.2, random_state=42, experiment_name=None
):
    """
    Train an XGBoost model with MLflow auto-logging.

    Args: same as :func:`train_xgboost`.

    Returns:
        dict: Same as :func:`train_xgboost` but with MLflow logging.
    """
    try:
        from src.models.mlflow_tracking import mlflow_run, log_sklearn_model, log_metrics_from_dict

        with mlflow_run(
            run_name=f"xgb_{task}_{random_state}",
            tags={"model": "xgboost", "task": task},
        ):
            import mlflow

            mlflow.log_params({
                "task": task,
                "test_size": test_size,
                "random_state": random_state,
            })
            result = train_xgboost(X, y, task=task, test_size=test_size, random_state=random_state)
            log_metrics_from_dict(result["metrics"])
            log_sklearn_model(result["model"])
            return result
    except ImportError:
        logger.warning("MLflow not available, falling back to standard training.")
        return train_xgboost(X, y, task=task, test_size=test_size, random_state=random_state)


# =============================================================================
# LightGBM and CatBoost trainers (optionally imported)
# =============================================================================


def train_lightgbm(X, y, task="classification", test_size=0.2, random_state=42):
    """
    Train a LightGBM model (classification or regression).

    Requires ``lightgbm``. Install with ``pip install lightgbm``.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("LightGBM is required. Install: pip install lightgbm")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task == "classification":
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=64,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
    else:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=64,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    logger.info("LightGBM (%s) metrics: %s", task, metrics)
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def train_catboost(X, y, task="classification", test_size=0.2, random_state=42):
    """
    Train a CatBoost model (classification or regression).

    Requires ``catboost``. Install with ``pip install catboost``.
    """
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError:
        raise ImportError("CatBoost is required. Install: pip install catboost")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    if task == "classification":
        model = CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            verbose=0,
        )
        model.fit(X_train_s, y_train, verbose=False)
        y_pred = model.predict(X_test_s)
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }
    else:
        model = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=random_state,
            verbose=0,
        )
        model.fit(X_train_s, y_train, verbose=False)
        y_pred = model.predict(X_test_s)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    logger.info("CatBoost (%s) metrics: %s", task, metrics)
    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
