"""
Performance and risk metrics for backtest result evaluation.

Extends the risk metrics in ``src.analysis.risk_metrics`` with
backtest-specific calculations (drawdown, rolling performance,
benchmark comparison, alpha/beta).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def calculate_max_drawdown_from_series(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown from a return series.

    Args:
        returns: Portfolio or strategy return series.

    Returns:
        Maximum drawdown as a negative fraction (e.g. -0.25 for 25% loss).
    """
    cum = (1.0 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    return float(drawdown.min())


def backtest_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of backtest performance metrics.

    Args:
        returns: Strategy return series.
        risk_free_rate: Annual risk-free rate. Default 0.02.
        periods: Trading periods per year. Default 252.

    Returns:
        Dict with keys:
            - total_return, annualized_return, annualized_volatility
            - sharpe_ratio, sortino_ratio, calmar_ratio
            - max_drawdown, max_drawdown_duration
            - win_rate, profit_factor
            - avg_win, avg_loss
    """
    from src.analysis.risk_metrics import (
        calculate_calmar_ratio,
        calculate_sharpe_ratio,
        calculate_sortino_ratio,
    )

    cum = (1.0 + returns).cumprod()
    total_ret = float(cum.iloc[-1] - 1.0)
    ann_ret = float(returns.mean() * periods)
    ann_vol = float(returns.std(ddof=1) * np.sqrt(periods))
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=risk_free_rate, periods=periods)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=risk_free_rate, periods=periods)
    calmar = calculate_calmar_ratio(returns, periods=periods)
    max_dd = calculate_max_drawdown_from_series(returns)

    dd_series = _drawdown_series(returns)
    dd_duration = _max_drawdown_duration(dd_series)

    winning = returns[returns > 0]
    losing = returns[returns < 0]
    win_rate = len(winning) / (len(winning) + len(losing) + 1e-12)
    avg_win = float(winning.mean()) if len(winning) > 0 else 0.0
    avg_loss = float(losing.mean()) if len(losing) > 0 else 0.0
    profit_factor = abs(winning.sum() / (losing.sum() + 1e-12)) if len(losing) > 0 else float("inf")

    return {
        "total_return": total_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "max_drawdown_duration": dd_duration,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def _drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown series from returns."""
    cum = (1.0 + returns).cumprod()
    rolling_max = cum.cummax()
    return (cum - rolling_max) / rolling_max


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """Compute the longest drawdown duration in trading days."""
    is_dd = drawdown < 0
    if not is_dd.any():
        return 0
    dd_groups = (is_dd != is_dd.shift(1)).cumsum()
    dd_lengths = dd_groups[is_dd].value_counts()
    return int(dd_lengths.max()) if len(dd_lengths) > 0 else 0


def calculate_rolling_performance(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.02,
    periods: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling performance metrics over a sliding window.

    Args:
        returns: Strategy return series.
        window: Rolling window size in trading days. Default 252.
        risk_free_rate: Annual risk-free rate. Default 0.02.
        periods: Trading periods per year. Default 252.

    Returns:
        DataFrame with rolling return, volatility, Sharpe, and max drawdown.
    """
    roll_ret = returns.rolling(window).mean() * periods
    roll_vol = returns.rolling(window).std(ddof=1) * np.sqrt(periods)
    daily_rf = risk_free_rate / periods
    roll_sharpe = (returns.rolling(window).mean() - daily_rf) / returns.rolling(window).std(ddof=1)
    roll_sharpe = roll_sharpe * np.sqrt(periods)

    roll_max_dd = returns.rolling(window).apply(
        lambda x: calculate_max_drawdown_from_series(x), raw=False
    )

    return pd.DataFrame(
        {
            "rolling_return": roll_ret,
            "rolling_volatility": roll_vol,
            "rolling_sharpe": roll_sharpe,
            "rolling_max_drawdown": roll_max_dd,
        },
        index=returns.index,
    )


def calculate_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252,
) -> Dict[str, float]:
    """
    Calculate CAPM-style alpha and beta relative to a benchmark.

    Args:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series.
        risk_free_rate: Annual risk-free rate. Default 0.02.
        periods: Trading periods per year. Default 252.

    Returns:
        Dict with alpha (annualized), beta, r-squared, and tracking error.
    """
    daily_rf = risk_free_rate / periods
    strat_excess = strategy_returns - daily_rf
    bench_excess = benchmark_returns - daily_rf

    cov = np.cov(strat_excess, bench_excess)
    beta = cov[0, 1] / (cov[1, 1] + 1e-12)
    alpha_daily = strat_excess.mean() - beta * bench_excess.mean()
    alpha_annual = alpha_daily * periods

    residuals = strat_excess - beta * bench_excess
    tracking_error = float(residuals.std(ddof=1) * np.sqrt(periods))

    corr = np.corrcoef(strat_excess, bench_excess)[0, 1]
    r_squared = corr**2

    return {
        "alpha": alpha_annual,
        "beta": beta,
        "r_squared": r_squared,
        "tracking_error": tracking_error,
    }


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> pd.DataFrame:
    """
    Compare cumulative performance of strategy vs benchmark.

    Args:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series.

    Returns:
        DataFrame with aligned cumulative returns for strategy and benchmark.
    """
    combined = pd.DataFrame(
        {
            "strategy": (1.0 + strategy_returns).cumprod(),
            "benchmark": (1.0 + benchmark_returns).cumprod(),
        }
    ).dropna()
    return combined
