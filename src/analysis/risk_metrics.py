"""
Risk metrics calculations: VaR, CVaR, Sharpe ratio, Sortino ratio, and more.
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_var(returns, confidence=0.95, method="historical"):
    """
    Calculate Value at Risk (VaR).

    Args:
        returns (pd.Series or np.ndarray): Return series.
        confidence (float): Confidence level (e.g. 0.95). Default 0.95.
        method (str): 'historical' or 'parametric'. Default 'historical'.

    Returns:
        float: VaR as a positive number representing the loss threshold.

    Raises:
        ValueError: If an unknown method is specified.
    """
    returns = np.asarray(returns)
    alpha = 1.0 - confidence
    if method == "historical":
        var = -np.percentile(returns, alpha * 100)
    elif method == "parametric":
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        var = -(mu + sigma * stats.norm.ppf(alpha))
    else:
        raise ValueError(f"Unknown VaR method: '{method}'. Use 'historical' or 'parametric'.")
    return float(var)


def calculate_cvar(returns, confidence=0.95, method="historical"):
    """
    Calculate Conditional Value at Risk (CVaR), also known as Expected Shortfall.

    Args:
        returns (pd.Series or np.ndarray): Return series.
        confidence (float): Confidence level (e.g. 0.95). Default 0.95.
        method (str): 'historical' or 'parametric'. Default 'historical'.

    Returns:
        float: CVaR as a positive number.

    Raises:
        ValueError: If an unknown method is specified.
    """
    returns = np.asarray(returns)
    alpha = 1.0 - confidence
    if method == "historical":
        var_threshold = np.percentile(returns, alpha * 100)
        tail = returns[returns <= var_threshold]
        cvar = -tail.mean() if len(tail) > 0 else -var_threshold
    elif method == "parametric":
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        z = stats.norm.ppf(alpha)
        cvar = -(mu - sigma * stats.norm.pdf(z) / alpha)
    else:
        raise ValueError(f"Unknown CVaR method: '{method}'. Use 'historical' or 'parametric'.")
    return float(cvar)


def calculate_sharpe_ratio(returns, risk_free_rate=0.02, periods=252):
    """
    Calculate the annualized Sharpe ratio.

    Args:
        returns (pd.Series or np.ndarray): Daily return series.
        risk_free_rate (float): Annual risk-free rate. Default 0.02.
        periods (int): Number of trading periods per year. Default 252.

    Returns:
        float: Sharpe ratio. Returns 0.0 if volatility is zero.
    """
    returns = np.asarray(returns)
    daily_rf = risk_free_rate / periods
    excess = returns - daily_rf
    vol = returns.std(ddof=1)   # volatility of raw returns, not excess
    if vol == 0:
        return 0.0
    return float((excess.mean() / vol) * np.sqrt(periods))


def calculate_sortino_ratio(returns, risk_free_rate=0.02, periods=252):
    """
    Calculate the annualized Sortino ratio (uses downside deviation).

    Args:
        returns (pd.Series or np.ndarray): Daily return series.
        risk_free_rate (float): Annual risk-free rate. Default 0.02.
        periods (int): Trading periods per year. Default 252.

    Returns:
        float: Sortino ratio. Returns 0.0 if downside volatility is zero.
    """
    returns = np.asarray(returns)
    daily_rf = risk_free_rate / periods
    excess = returns - daily_rf
    downside = excess[excess < 0]
    downside_vol = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 0.0
    if downside_vol == 0:
        return 0.0
    return float((excess.mean() / downside_vol) * np.sqrt(periods))


def calculate_calmar_ratio(returns, periods=252):
    """
    Calculate the Calmar ratio (annualized return / max drawdown magnitude).

    Args:
        returns (pd.Series or np.ndarray): Daily return series.
        periods (int): Trading periods per year. Default 252.

    Returns:
        float: Calmar ratio. Returns 0.0 if max drawdown is zero.
    """
    returns = np.asarray(returns)
    ann_return = returns.mean() * periods
    cum = np.cumprod(1 + returns)
    rolling_max = np.maximum.accumulate(cum)
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())
    if max_dd == 0:
        return 0.0
    return float(ann_return / max_dd)


def rolling_var(returns, window=30, confidence=0.95):
    """
    Calculate rolling historical VaR over a sliding window.

    Args:
        returns (pd.Series): Daily return series with DatetimeIndex.
        window (int): Rolling window size in trading days. Default 30.
        confidence (float): VaR confidence level. Default 0.95.

    Returns:
        pd.Series: Rolling VaR series (positive values representing loss magnitude).
    """
    alpha = 1.0 - confidence
    return returns.rolling(window).quantile(alpha).abs()


def rolling_sharpe(returns, window=252, risk_free_rate=0.02, periods=252):
    """
    Calculate rolling annualized Sharpe ratio.

    Args:
        returns (pd.Series): Daily return series.
        window (int): Rolling window in trading days. Default 252.
        risk_free_rate (float): Annual risk-free rate. Default 0.02.
        periods (int): Trading periods per year. Default 252.

    Returns:
        pd.Series: Rolling Sharpe ratio series.
    """
    daily_rf = risk_free_rate / periods
    excess = returns - daily_rf
    roll_mean = excess.rolling(window).mean()
    roll_std = excess.rolling(window).std(ddof=1)
    return (roll_mean / roll_std) * np.sqrt(periods)


def risk_summary(returns, confidence=0.95, risk_free_rate=0.02, periods=252):
    """
    Compute a full risk metric summary for a return series.

    Args:
        returns (pd.Series): Daily return series.
        confidence (float): Confidence level for VaR/CVaR. Default 0.95.
        risk_free_rate (float): Annual risk-free rate. Default 0.02.
        periods (int): Trading periods per year. Default 252.

    Returns:
        dict: Keys: var, cvar, sharpe_ratio, sortino_ratio, calmar_ratio,
            annualized_return, annualized_volatility.
    """
    return {
        "var": calculate_var(returns, confidence=confidence),
        "cvar": calculate_cvar(returns, confidence=confidence),
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate=risk_free_rate, periods=periods),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate=risk_free_rate, periods=periods),
        "calmar_ratio": calculate_calmar_ratio(returns, periods=periods),
        "annualized_return": float(np.mean(returns) * periods),
        "annualized_volatility": float(np.std(returns, ddof=1) * np.sqrt(periods)),
    }
