"""
Core backtesting engine for rebalancing-based portfolio strategies.

Simulates periodic portfolio rebalancing with configurable transaction
costs and provides detailed performance attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.backtesting.costs import CostModel, ProportionalCost


@dataclass
class BacktestResult:
    """Result of a backtest run."""

    dates: pd.DatetimeIndex
    portfolio_values: pd.Series
    weights_history: pd.DataFrame
    returns: pd.Series
    cumulative_returns: pd.Series
    turnover: pd.Series
    costs: pd.Series
    trades: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return float(self.cumulative_returns.iloc[-1] - 1.0)

    @property
    def total_cost(self) -> float:
        return float(self.costs.sum())

    @property
    def average_turnover(self) -> float:
        return float(self.turnover.mean())


def rebalance_strategy(
    returns: pd.DataFrame,
    weight_func: Callable[[pd.DataFrame], np.ndarray],
    rebalance_freq: str = "M",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = 1_000_000.0,
    cost_model: Optional[CostModel] = None,
    min_weight: float = 0.0,
) -> BacktestResult:
    """
    Run a rebalancing-based portfolio backtest.

    At each rebalance date, ``weight_func`` is called with the historical
    returns up to that date and must return a weight vector.

    Args:
        returns: DataFrame of asset returns (dates x assets).
        weight_func: Function that takes a returns DataFrame and returns
            weight array. E.g. ``lambda r: minimum_variance_weights(r)``.
        rebalance_freq: Pandas offset string for rebalancing frequency.
            'M' = monthly, 'W' = weekly, 'Q' = quarterly, 'D' = daily.
            Default 'M'.
        start_date: Optional start date (inclusive).
        end_date: Optional end date (inclusive).
        initial_capital: Starting portfolio value. Default 1,000,000.
        cost_model: Transaction cost model. If None, uses
            ``ProportionalCost(bps=10)`` (0.1% per trade).
        min_weight: Minimum weight threshold; below this the asset is
            treated as zero. Default 0.0.

    Returns:
        ``BacktestResult`` with portfolio value history, weights, returns,
        turnover, and costs.
    """
    if cost_model is None:
        cost_model = ProportionalCost(bps=10)

    data = returns.copy()
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]

    rebalance_dates = data.resample(rebalance_freq).apply(lambda x: x.index[0]).unique()
    rebalance_dates = pd.DatetimeIndex(sorted(set(rebalance_dates) & set(data.index)))

    n_assets = data.shape[1]
    current_weights = np.ones(n_assets) / n_assets
    portfolio_value = initial_capital
    prev_portfolio_value = initial_capital

    values = []
    weights_hist = []
    port_returns = []
    turnovers = []
    costs = []
    trades = []

    for i, date in enumerate(data.index):
        daily_rets = data.loc[date].values

        if date in rebalance_dates and i > 0:
            hist_returns = data.loc[:date].iloc[:-1]
            if len(hist_returns) >= 10:
                new_weights = weight_func(hist_returns)
                new_weights = np.asarray(new_weights, dtype=float)

                mask = new_weights < min_weight
                new_weights[mask] = 0.0
                new_weights = new_weights / (new_weights.sum() + 1e-12)

                turnover = np.abs(new_weights - current_weights).sum()
                trade_cost = cost_model.calculate(current_weights, new_weights, portfolio_value)
                portfolio_value -= trade_cost

                if turnover > 0:
                    trades.append(
                        {
                            "date": date,
                            "turnover": turnover,
                            "cost": trade_cost,
                            "old_weights": current_weights.copy(),
                            "new_weights": new_weights.copy(),
                        }
                    )

                current_weights = new_weights
            else:
                turnover = 0.0
                trade_cost = 0.0
        else:
            turnover = 0.0
            trade_cost = 0.0

        daily_return = current_weights @ daily_rets
        portfolio_value *= (1.0 + daily_return)

        values.append(portfolio_value)
        weights_hist.append(current_weights.copy())
        port_returns.append(daily_return)
        turnovers.append(turnover)
        costs.append(trade_cost)

        prev_portfolio_value = portfolio_value

    index = data.index
    portfolio_values = pd.Series(values, index=index, name="portfolio_value")
    weights_history = pd.DataFrame(weights_hist, index=index, columns=data.columns)
    returns_series = pd.Series(port_returns, index=index, name="strategy_return")
    cum_returns = (1.0 + returns_series).cumprod()
    turnover_series = pd.Series(turnovers, index=index, name="turnover")
    costs_series = pd.Series(costs, index=index, name="costs")

    return BacktestResult(
        dates=index,
        portfolio_values=portfolio_values,
        weights_history=weights_history,
        returns=returns_series,
        cumulative_returns=cum_returns,
        turnover=turnover_series,
        costs=costs_series,
        trades=trades,
        metadata={
            "initial_capital": initial_capital,
            "rebalance_freq": rebalance_freq,
            "n_assets": n_assets,
            "cost_model": str(cost_model),
        },
    )


class RebalanceBacktester:
    """
    Rebalancing-based portfolio backtester with configurable strategy.

    Wraps ``rebalance_strategy`` in a class interface for easier
    integration with optimization and parameter sweeps.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weight_func: Callable[[pd.DataFrame], np.ndarray],
        rebalance_freq: str = "M",
        cost_model: Optional[CostModel] = None,
        initial_capital: float = 1_000_000.0,
    ):
        self.returns = returns
        self.weight_func = weight_func
        self.rebalance_freq = rebalance_freq
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.result: Optional[BacktestResult] = None

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """Execute the backtest."""
        self.result = rebalance_strategy(
            returns=self.returns,
            weight_func=self.weight_func,
            rebalance_freq=self.rebalance_freq,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            cost_model=self.cost_model,
        )
        return self.result

    def summary(self) -> Dict[str, Any]:
        """Return key performance metrics."""
        if self.result is None:
            return {"error": "Run backtest first with .run()"}
        r = self.result
        return {
            "total_return": r.total_return,
            "total_cost": r.total_cost,
            "avg_turnover": r.average_turnover,
            "n_trades": len(r.trades),
            "start_date": r.dates[0],
            "end_date": r.dates[-1],
            "final_value": r.portfolio_values.iloc[-1],
            "peak_value": r.portfolio_values.max(),
            "min_value": r.portfolio_values.min(),
        }
