"""
CVaR (Conditional Value at Risk) portfolio optimization.

Uses linear programming to minimize CVaR (Expected Shortfall) directly
from scenario returns, producing a portfolio that is robust to tail risk.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize


def cvar_optimal_weights(
    returns: pd.DataFrame,
    confidence: float = 0.95,
    risk_aversion: float = 1.0,
    allow_short: bool = False,
) -> np.ndarray:
    """
    Compute CVaR-minimizing portfolio weights via scenario optimization.

    Solves:

        minimize_{w, z, alpha}  alpha + (1 / ((1 - conf) * T)) * sum(z_t)

    subject to:
        -r_t' * w - alpha <= z_t
        z_t >= 0
        sum(w) = 1
        w >= 0 (if no short selling)

    where r_t are the scenario returns, alpha is the VaR threshold,
    and z_t are the tail losses.

    Args:
        returns: DataFrame of asset returns (scenarios x assets).
        confidence: CVaR confidence level. Default 0.95.
        risk_aversion: CVaR contribution weight (not used directly in
            standard CVaR minimization; kept for API consistency).
        allow_short: Allow negative weights. Default False.

    Returns:
        Array of optimal portfolio weights.
    """
    T, n = returns.shape
    ret_matrix = returns.values
    alpha_conf = 1.0 / (1.0 - confidence) / T

    def objective(params: np.ndarray) -> float:
        w = params[:n]
        alpha = params[n]
        z = params[n + 1:]
        return float(alpha + alpha_conf * z.sum())

    w0 = np.ones(n) / n
    alpha0 = float(-np.percentile(ret_matrix @ w0, (1.0 - confidence) * 100))
    z0 = np.maximum(-ret_matrix @ w0 - alpha0, 0.0)
    x0 = np.concatenate([w0, [alpha0], z0])

    n_vars = n + 1 + T
    if allow_short:
        bounds = [(None, None)] * n + [(None, None)] + [(0.0, None)] * T
    else:
        bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * T

    constraints = [
        LinearConstraint(np.eye(n_vars)[:n].sum(axis=0).reshape(1, -1), 1.0, 1.0),
    ]

    for t in range(T):
        A_t = np.zeros(n_vars)
        A_t[:n] = -ret_matrix[t, :]
        A_t[n] = -1.0
        A_t[n + 1 + t] = -1.0
        constraints.append(LinearConstraint(A_t.reshape(1, -1), -np.inf, 0.0))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x[:n]
    weights = weights / weights.sum()
    return weights


def cvar_risk_contribution(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Decompose portfolio CVaR into per-asset risk contributions.

    Uses the Euler allocation principle:

        CVaR_i = w_i * d(CVaR) / d(w_i)

    approximated by averaging returns in tail scenarios.

    Args:
        returns: Asset return DataFrame.
        weights: Portfolio weights.
        confidence: CVaR confidence level.

    Returns:
        Dict mapping asset names to CVaR contributions.
    """
    port_returns = returns.values @ weights
    alpha = 1.0 - confidence
    var_threshold = np.percentile(port_returns, alpha * 100)
    tail_mask = port_returns <= var_threshold
    tail_returns = returns.values[tail_mask, :]

    if tail_returns.shape[0] == 0:
        return {col: 0.0 for col in returns.columns}

    total_cvar = -tail_returns.mean(axis=0) @ weights
    contributions = weights * (-tail_returns.mean(axis=0))
    contributions = contributions / (contributions.sum() + 1e-12) * total_cvar

    return dict(zip(returns.columns, contributions))


def efficient_frontier_cvar(
    returns: pd.DataFrame,
    target_returns: Optional[List[float]] = None,
    n_points: int = 20,
    allow_short: bool = False,
) -> pd.DataFrame:
    """
    Compute the CVaR-efficient frontier.

    For each target return level, minimizes CVaR.

    Args:
        returns: Asset return DataFrame.
        target_returns: List of target return levels. If None, generates
            evenly spaced from min to max asset return.
        n_points: Number of frontier points. Default 20.
        allow_short: Allow negative weights. Default False.

    Returns:
        DataFrame with columns: target_return, cvar, weights.
    """
    mean_rets = returns.mean()

    if target_returns is None:
        min_ret = mean_rets.min()
        max_ret = mean_rets.max()
        target_returns = np.linspace(min_ret, max_ret, n_points).tolist()

    frontier = []
    for t_ret in target_returns:
        w = _min_cvar_for_target(returns, t_ret, allow_short)
        port_ret = float(returns.values @ w)
        port_rets = returns.values @ w
        alpha = 0.05
        var_th = np.percentile(port_rets, alpha * 100)
        cvar = float(-port_rets[port_rets <= var_th].mean())
        frontier.append(
            {
                "target_return": t_ret,
                "expected_return": port_ret,
                "cvar": cvar,
                "weights": w,
            }
        )

    return pd.DataFrame(frontier)


def _min_cvar_for_target(
    returns: pd.DataFrame,
    target_return: float,
    allow_short: bool = False,
) -> np.ndarray:
    """Minimize CVaR subject to achieving a target return."""
    T, n = returns.shape
    ret_matrix = returns.values
    alpha_conf = 1.0 / (1.0 - 0.95) / T

    def objective(params: np.ndarray) -> float:
        w = params[:n]
        a = params[n]
        z = params[n + 1:]
        return float(a + alpha_conf * z.sum())

    w0 = np.ones(n) / n
    a0 = float(-np.percentile(ret_matrix @ w0, 5.0))
    z0 = np.maximum(-ret_matrix @ w0 - a0, 0.0)
    x0 = np.concatenate([w0, [a0], z0])

    n_vars = n + 1 + T
    if allow_short:
        bounds = [(None, None)] * n + [(None, None)] + [(0.0, None)] * T
    else:
        bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * T

    mean_ret = returns.mean().values

    constraints = [
        LinearConstraint(np.eye(n_vars)[:n].sum(axis=0).reshape(1, -1), 1.0, 1.0),
        LinearConstraint(np.concatenate([mean_ret, [0.0], np.zeros(T)]).reshape(1, -1), target_return, np.inf),
    ]

    for t in range(T):
        A_t = np.zeros(n_vars)
        A_t[:n] = -ret_matrix[t, :]
        A_t[n] = -1.0
        A_t[n + 1 + t] = -1.0
        constraints.append(LinearConstraint(A_t.reshape(1, -1), -np.inf, 0.0))

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    w = result.x[:n]
    return w / w.sum()
