"""
Realized volatility measures and HAR-RV (Heterogeneous Auto-Regressive) model.

Provides estimators for daily, weekly, and monthly realized volatility
and a HAR-RV forecasting model.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.tools import add_constant


def realized_volatility(
    prices: pd.Series,
    window: int = 1,
    annualize: bool = False,
    periods: int = 252,
) -> pd.Series:
    """
    Compute realized volatility from a price series.

    Uses the square root of the sum of squared log-returns over the window.

    Args:
        prices: Price series with DatetimeIndex.
        window: Aggregation window in days. Default 1 (daily RV).
        annualize: Whether to annualize. Default False.
        periods: Trading periods per year. Default 252.

    Returns:
        Series of realized volatility estimates.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rv = log_ret**2
    if window > 1:
        rv = rv.rolling(window).sum()
    vol = np.sqrt(rv)
    if annualize:
        vol *= np.sqrt(periods)
    return vol


def realized_variance(prices: pd.Series, window: int = 1) -> pd.Series:
    """
    Compute realized variance (sum of squared log-returns).

    Args:
        prices: Price series.
        window: Aggregation window.

    Returns:
        Series of realized variances.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rv = log_ret**2
    if window > 1:
        rv = rv.rolling(window).sum()
    return rv


def bipower_variation(
    prices: pd.Series,
    window: int = 1,
) -> pd.Series:
    """
    Compute bipower variation (robust to jumps).

    BV_t = mu_1^{-2} * sum_{j=2}^{n_t} |r_{t,j}| * |r_{t,j-1}|

    where mu_1 = sqrt(2/pi).

    Args:
        prices: Price series.
        window: Aggregation window.

    Returns:
        Series of bipower variation estimates.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    mu_1 = np.sqrt(2.0 / np.pi)
    bv = (np.pi / 2.0) * (
        log_ret.abs() * log_ret.shift(1).abs()
    ).dropna()
    if window > 1:
        bv = bv.rolling(window).sum()
    return bv


def realized_quarticity(prices: pd.Series, window: int = 1) -> pd.Series:
    """
    Compute realized quarticity (used for jump detection).

    RQ_t = (n_t / 3) * sum_{j=1}^{n_t} r_{t,j}^4

    Args:
        prices: Price series.
        window: Aggregation window.

    Returns:
        Series of realized quarticity.
    """
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rq = (1.0 / 3.0) * (log_ret**4)
    if window > 1:
        rq = rq.rolling(window).sum()
    return rq


def jump_component(
    prices: pd.Series,
    window: int = 1,
    threshold: float = 2.0,
) -> pd.Series:
    """
    Estimate the jump component of volatility.

    J_t = max(RV_t - BV_t, 0)

    Args:
        prices: Price series.
        window: Aggregation window.
        threshold: Not used directly (returns raw jump estimate).

    Returns:
        Series of jump components.
    """
    rv = realized_variance(prices, window=window)
    bv = bipower_variation(prices, window=window)
    jumps = pd.Series(
        np.maximum(rv.values - bv.values, 0.0),
        index=bv.index,
        name="jumps",
    )
    return jumps


def _compute_rv_components(
    prices: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute daily, weekly, and monthly realized volatilities.

    Args:
        prices: Price series.

    Returns:
        Tuple of (rv_daily, rv_weekly, rv_monthly) DataFrames.
    """
    rv_d = realized_variance(prices, window=1)
    rv_w = realized_variance(prices, window=5)
    rv_m = realized_variance(prices, window=22)
    return rv_d, rv_w, rv_m


def prepare_har_features(
    prices: pd.Series,
    lags: int = 5,
) -> pd.DataFrame:
    """
    Prepare feature matrix for HAR-RV model.

    Features:
        - RV_daily (lagged 1 day)
        - RV_weekly (lagged 1 week, 5-day average)
        - RV_monthly (lagged 1 month, 22-day average)
        - Optional: additional lags of daily RV

    Args:
        prices: Price series.
        lags: Number of daily RV lags to include. Default 5.

    Returns:
        DataFrame of HAR features with aligned index.
    """
    rv_d, rv_w, rv_m = _compute_rv_components(prices)

    features = pd.DataFrame(index=rv_d.index)
    features["RV_daily"] = rv_d
    features["RV_weekly"] = rv_w
    features["RV_monthly"] = rv_m

    for lag in range(1, lags + 1):
        features[f"RV_lag_{lag}"] = rv_d.shift(lag)

    features = features.dropna()
    return features


def fit_har_rv(
    prices: pd.Series,
    lags: int = 5,
    include_weekly: bool = True,
    include_monthly: bool = True,
) -> Dict[str, object]:
    """
    Fit a HAR-RV model using OLS.

    The HAR-RV model (Corsi, 2009) regresses today's RV on:
        - Yesterday's RV (daily component)
        - Average RV over the past week (weekly component)
        - Average RV over the past month (monthly component)

    Args:
        prices: Price series.
        lags: Number of daily RV lags. Default 5.
        include_weekly: Include weekly RV component. Default True.
        include_monthly: Include monthly RV component. Default True.

    Returns:
        Dict with keys:
            - 'model': fitted OLS result
            - 'features': DataFrame of features used
            - 'target': Series of target values
            - 'predictions': in-sample predictions
            - 'params': coefficient summary
    """
    rv_d, rv_w, rv_m = _compute_rv_components(prices)

    features = pd.DataFrame(index=rv_d.index)
    features["RV_daily"] = rv_d.shift(1)
    if include_weekly:
        features["RV_weekly"] = rv_w.shift(1)
    if include_monthly:
        features["RV_monthly"] = rv_m.shift(1)

    for lag in range(1, lags + 1):
        features[f"RV_lag_{lag}"] = rv_d.shift(lag + 1)

    target = rv_d.copy()
    data = pd.concat([target, features], axis=1).dropna()
    data.columns = ["target"] + list(features.columns)

    X = add_constant(data[features.columns])
    y = data["target"]

    model = OLS(y, X).fit()

    return {
        "model": model,
        "features": data[features.columns],
        "target": y,
        "predictions": model.predict(X),
        "params": model.params.to_dict(),
        "rsquared": model.rsquared,
        "aic": model.aic,
        "bic": model.bic,
    }


def forecast_har_rv(
    model_result: Dict[str, object],
    horizon: int = 5,
) -> pd.Series:
    """
    Generate multi-step HAR-RV forecasts.

    For h > 1, forecasts are iterated: predicted RV from step t becomes
    the lagged daily RV for step t+1.

    Args:
        model_result: Result dict from ``fit_har_rv``.
        horizon: Forecast horizon in days. Default 5.

    Returns:
        Series of forecasted RV values.
    """
    model = model_result["model"]
    features = model_result["features"]
    last_row = features.iloc[-1:].copy()
    params = model.params

    forecasts = []
    for _ in range(horizon):
        X_pred = add_constant(last_row, has_constant="add")
        pred = float(model.predict(X_pred).iloc[0])
        forecasts.append(pred)
        last_row["RV_daily"] = pred
        for lag in range(2, 6):
            col = f"RV_lag_{lag}"
            if col in last_row.columns:
                last_row[col] = last_row[f"RV_lag_{lag - 1}"].values[0]

    return pd.Series(
        forecasts,
        index=[f"t+{i+1}" for i in range(horizon)],
        name="har_rv_forecast",
    )
