"""
Transaction cost models for backtesting simulations.

Provides fixed, proportional, and spread-based cost models that can be
composed for realistic trading cost estimation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class CostModel(ABC):
    """Abstract base class for transaction cost models."""

    @abstractmethod
    def calculate(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        """
        Compute transaction cost for a rebalance.

        Args:
            old_weights: Current portfolio weights.
            new_weights: Target portfolio weights after rebalance.
            portfolio_value: Current portfolio value.

        Returns:
            Total cost in currency units.
        """
        ...

    def __str__(self) -> str:
        return self.__class__.__name__


class FixedCost(CostModel):
    """
    Fixed per-trade cost model.

    Charges a fixed amount per trade regardless of size.
    """

    def __init__(self, per_trade: float = 10.0):
        """
        Args:
            per_trade: Fixed cost per rebalance event (in currency).
                Default 10.0.
        """
        self.per_trade = per_trade

    def calculate(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        diff = np.abs(new_weights - old_weights).sum()
        if diff > 1e-8:
            return self.per_trade
        return 0.0

    def __str__(self) -> str:
        return f"FixedCost({self.per_trade:.2f})"


class ProportionalCost(CostModel):
    """
    Proportional transaction cost model.

    Costs are a fraction of the traded notional value.
    """

    def __init__(self, bps: float = 10.0):
        """
        Args:
            bps: Cost in basis points (1 bps = 0.01%). Default 10 (0.1%).
        """
        self.bps = bps
        self.fraction = bps / 10_000.0

    def calculate(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        traded = np.abs(new_weights - old_weights).sum() * portfolio_value
        return traded * self.fraction

    def __str__(self) -> str:
        return f"ProportionalCost({self.bps:.1f} bps)"


class SpreadCost(CostModel):
    """
    Bid-ask spread cost model.

    Models the cost of crossing the spread as a fixed fraction per trade.
    Useful for less liquid assets.
    """

    def __init__(self, spread_bps: float = 5.0):
        """
        Args:
            spread_bps: Bid-ask spread in basis points. Default 5 (0.05%).
        """
        self.spread_bps = spread_bps
        self.fraction = spread_bps / 10_000.0

    def calculate(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        portfolio_value: float,
    ) -> float:
        bought = np.maximum(new_weights - old_weights, 0.0).sum() * portfolio_value
        sold = np.maximum(old_weights - new_weights, 0.0).sum() * portfolio_value
        return (bought + sold) * self.fraction

    def __str__(self) -> str:
        return f"SpreadCost({self.spread_bps:.1f} bps)"


def calculate_total_costs(
    cost_models: list[CostModel],
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    portfolio_value: float,
) -> float:
    """
    Combine multiple cost models for a total cost estimate.

    Args:
        cost_models: List of cost model instances.
        old_weights: Current weights.
        new_weights: Target weights.
        portfolio_value: Current portfolio value.

    Returns:
        Total cost across all models.
    """
    total = 0.0
    for cm in cost_models:
        total += cm.calculate(old_weights, new_weights, portfolio_value)
    return total
