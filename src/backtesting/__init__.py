"""
Backtesting framework for portfolio strategies.

Provides:
- Rebalancing-based backtester with transaction costs
- Performance and risk metric calculators
- Transaction cost models (fixed, proportional, spread-based)
"""

from src.backtesting.backtester import (
    BacktestResult,
    RebalanceBacktester,
    rebalance_strategy,
)
from src.backtesting.costs import (
    CostModel,
    FixedCost,
    ProportionalCost,
    SpreadCost,
    calculate_total_costs,
)
from src.backtesting.metrics import (
    backtest_metrics,
    calculate_alpha_beta,
    calculate_max_drawdown_from_series,
    calculate_rolling_performance,
    compare_to_benchmark,
)

__all__ = [
    "RebalanceBacktester",
    "BacktestResult",
    "rebalance_strategy",
    "CostModel",
    "FixedCost",
    "ProportionalCost",
    "SpreadCost",
    "calculate_total_costs",
    "backtest_metrics",
    "calculate_max_drawdown_from_series",
    "calculate_rolling_performance",
    "calculate_alpha_beta",
    "compare_to_benchmark",
]
