"""
Portfolio-level calculations: returns, volatility, optimization, and statistics.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def calculate_portfolio_returns(returns, weights):
    """
    Calculate weighted portfolio returns.

    Args:
        returns (pd.DataFrame): Asset return DataFrame (dates x tickers).
        weights (array-like): Portfolio weights that sum to 1.0.

    Returns:
        pd.Series: Daily portfolio return series.
    """
    weights = np.array(weights)
    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.6f}")
    return returns.dot(weights)


def calculate_portfolio_volatility(returns, weights=None):
    """
    Calculate annualized portfolio volatility.

    If weights are provided, computes portfolio-level volatility from the
    covariance matrix. Otherwise treats `returns` as a single return series.

    Args:
        returns (pd.DataFrame or pd.Series): Returns data.
        weights (array-like, optional): Asset weights.

    Returns:
        float: Annualized portfolio volatility (std dev).
    """
    if weights is not None:
        weights = np.array(weights)
        cov = returns.cov() * 252
        variance = weights @ cov @ weights
        return float(np.sqrt(variance))
    else:
        series = returns if isinstance(returns, pd.Series) else returns.squeeze()
        return float(series.std(ddof=1) * np.sqrt(252))


def calculate_cumulative_returns(returns):
    """
    Calculate cumulative returns from a return series.

    Args:
        returns (pd.Series or pd.DataFrame): Return series or DataFrame.

    Returns:
        pd.Series or pd.DataFrame: Cumulative return series starting from 1.0.
    """
    return (1 + returns).cumprod()


def calculate_drawdown(cumulative_returns):
    """
    Calculate drawdown series from cumulative returns.

    Args:
        cumulative_returns (pd.Series): Cumulative return series.

    Returns:
        pd.Series: Drawdown at each time step (negative values).
    """
    rolling_max = cumulative_returns.cummax()
    return (cumulative_returns - rolling_max) / rolling_max


def calculate_max_drawdown(cumulative_returns):
    """
    Calculate maximum drawdown.

    Args:
        cumulative_returns (pd.Series): Cumulative return series.

    Returns:
        float: Maximum drawdown (negative value).
    """
    return float(calculate_drawdown(cumulative_returns).min())


def calculate_correlation_matrix(returns):
    """
    Compute the pairwise correlation matrix of asset returns.

    Args:
        returns (pd.DataFrame): Asset return DataFrame.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    return returns.corr()


def calculate_covariance_matrix(returns, annualize=True, periods=252):
    """
    Compute the covariance matrix of asset returns.

    Args:
        returns (pd.DataFrame): Asset return DataFrame.
        annualize (bool): Whether to annualize. Default True.
        periods (int): Trading periods per year. Default 252.

    Returns:
        pd.DataFrame: Covariance matrix.
    """
    cov = returns.cov()
    if annualize:
        cov = cov * periods
    return cov


def minimum_variance_weights(returns):
    """
    Compute portfolio weights for the global minimum variance portfolio.

    Args:
        returns (pd.DataFrame): Asset return DataFrame.

    Returns:
        np.ndarray: Optimal weights.
    """
    n = returns.shape[1]
    cov = returns.cov().values

    def portfolio_variance(w):
        return w @ cov @ w

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    result = minimize(
        portfolio_variance, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result.x


def maximum_sharpe_weights(returns, risk_free_rate=0.02, periods=252):
    """
    Compute portfolio weights that maximize the Sharpe ratio.

    Args:
        returns (pd.DataFrame): Asset return DataFrame.
        risk_free_rate (float): Annual risk-free rate. Default 0.02.
        periods (int): Trading periods per year. Default 252.

    Returns:
        np.ndarray: Optimal weights.
    """
    n = returns.shape[1]
    mean_returns = returns.mean() * periods
    cov = returns.cov() * periods
    rf_daily = risk_free_rate

    def neg_sharpe(w):
        port_return = w @ mean_returns
        port_vol = np.sqrt(w @ cov @ w)
        return -(port_return - rf_daily) / (port_vol + 1e-12)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    result = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints
    )
    return result.x


def portfolio_summary(returns, weights, risk_free_rate=0.02, periods=252):
    """
    Compute a summary of key portfolio statistics.

    Args:
        returns (pd.DataFrame): Asset return DataFrame.
        weights (array-like): Portfolio weights.
        risk_free_rate (float): Annual risk-free rate. Default 0.02.
        periods (int): Trading periods per year. Default 252.

    Returns:
        dict: Dictionary containing annualized_return, annualized_volatility,
            sharpe_ratio, max_drawdown, and cumulative_return.
    """
    from src.analysis.risk_metrics import calculate_sharpe_ratio

    port_returns = calculate_portfolio_returns(returns, weights)
    ann_ret = float(port_returns.mean() * periods)
    ann_vol = float(port_returns.std(ddof=1) * np.sqrt(periods))
    sharpe = calculate_sharpe_ratio(port_returns, risk_free_rate=risk_free_rate, periods=periods)
    cum_ret = calculate_cumulative_returns(port_returns)
    max_dd = calculate_max_drawdown(cum_ret)

    return {
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "cumulative_return": float(cum_ret.iloc[-1] - 1),
    }
