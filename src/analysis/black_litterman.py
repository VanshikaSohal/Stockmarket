"""
Black-Litterman portfolio optimization model.

Blends prior market equilibrium returns with investor views to produce
posterior expected returns used in mean-variance optimization.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def market_implied_returns(
    cov_matrix: pd.DataFrame,
    market_caps: pd.Series,
    risk_aversion: float = 2.5,
) -> pd.Series:
    """
    Compute market-capitalization weighted implied returns (prior).

    Uses the reverse-optimization formula:

        pi = delta * Sigma * w_mkt

    where delta is the risk aversion coefficient, Sigma is the covariance
    matrix, and w_mkt are the market-cap weights.

    Args:
        cov_matrix: Annualized covariance matrix (assets x assets).
        market_caps: Market capitalization series for each asset.
        risk_aversion: Risk aversion coefficient (lambda). Default 2.5.

    Returns:
        Series of implied equilibrium returns (prior).
    """
    mkt_weights = market_caps / market_caps.sum()
    implied_returns = risk_aversion * cov_matrix @ mkt_weights
    return pd.Series(implied_returns, index=cov_matrix.index, name="implied_return")


def black_litterman_estimate(
    cov_matrix: pd.DataFrame,
    prior_returns: pd.Series,
    view_matrix: np.ndarray,
    view_returns: np.ndarray,
    view_confidences: np.ndarray,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
) -> Dict[str, object]:
    """
    Compute Black-Litterman posterior expected returns and covariance.

    The model blends a prior (market equilibrium) with absolute or relative
    views using a Bayesian approach:

        mu_post = [(tau*Sigma)^{-1} + P' * Omega^{-1} * P]^{-1}
                * [(tau*Sigma)^{-1} * pi + P' * Omega^{-1} * Q]

    where:
        pi   = prior expected returns
        tau  = uncertainty scaling (0.01-0.10)
        Sigma = prior covariance
        P    = view matrix (k x n, each row encodes a view)
        Q    = view return vector (k x 1)
        Omega = diagonal uncertainty matrix from view_confidences

    Args:
        cov_matrix: Prior covariance matrix (n x n).
        prior_returns: Prior expected returns (n x 1).
        view_matrix: View matrix P (k x n). Each row is a view.
            Absolute views: single 1 at asset position.
            Relative views: +1 for long, -1 for short.
        view_returns: View return vector Q (k x 1).
        view_confidences: Confidence level for each view (0-1).
            Used to compute Omega = (1/c - 1) * tau * diag(P * Sigma * P').
        tau: Prior uncertainty scaling. Default 0.05.
        risk_aversion: Risk aversion coefficient. Default 2.5.

    Returns:
        Dict with keys:
            - 'posterior_returns': posterior expected returns (Series)
            - 'posterior_covariance': posterior covariance (DataFrame)
            - 'prior_returns': prior returns (Series)
            - 'weights': optimal weights from posterior (Series)
            - 'view_contribution': contribution of each view (dict)
    """
    n = cov_matrix.shape[0]
    k = len(view_returns)

    prior = prior_returns.values.reshape(-1, 1)
    Sigma = cov_matrix.values
    P = view_matrix.astype(float)
    Q = view_returns.reshape(-1, 1)
    conf = np.asarray(view_confidences).reshape(-1, 1)

    Omega = np.diag(((1.0 / (conf.flatten() + 1e-12)) - 1.0) * tau * np.diag(P @ Sigma @ P.T))

    tau_Sigma_inv = np.linalg.inv(tau * Sigma)
    PO_inv = P.T @ np.linalg.inv(Omega) @ P
    PO_inv_Q = P.T @ np.linalg.inv(Omega) @ Q

    mu_post = np.linalg.inv(tau_Sigma_inv + PO_inv) @ (tau_Sigma_inv @ prior + PO_inv_Q)
    Sigma_post = Sigma + np.linalg.inv(tau_Sigma_inv + PO_inv)

    mu_post_series = pd.Series(mu_post.flatten(), index=cov_matrix.index, name="posterior_return")
    Sigma_post_df = pd.DataFrame(Sigma_post, index=cov_matrix.index, columns=cov_matrix.columns)

    weights = _optimal_weights_from_posterior(mu_post_series, Sigma_post_df, risk_aversion)
    weight_series = pd.Series(weights, index=cov_matrix.index, name="bl_weights")

    view_contrib = {}
    for i in range(k):
        view_contrib[f"view_{i+1}"] = {
            "view_vector": P[i, :].tolist(),
            "view_return": float(Q[i, 0]),
            "confidence": float(conf[i, 0]),
        }

    return {
        "posterior_returns": mu_post_series,
        "posterior_covariance": Sigma_post_df,
        "prior_returns": prior_returns,
        "weights": weight_series,
        "view_contribution": view_contrib,
    }


def _optimal_weights_from_posterior(
    mu: pd.Series,
    Sigma: pd.DataFrame,
    risk_aversion: float = 2.5,
) -> np.ndarray:
    """
    Compute mean-variance optimal weights from posterior estimates.

    w* = (1 / delta) * Sigma^{-1} * mu

    Args:
        mu: Posterior expected returns.
        Sigma: Posterior covariance matrix.
        risk_aversion: Risk aversion coefficient.

    Returns:
        Optimal weight vector (may include short positions).
    """
    Sigma_inv = np.linalg.inv(Sigma.values)
    raw_weights = (1.0 / risk_aversion) * Sigma_inv @ mu.values
    raw_weights = raw_weights / raw_weights.sum()
    return raw_weights


def absolute_view(
    asset_name: str,
    expected_return: float,
    asset_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create an absolute view: "Asset X will return Y%".

    Args:
        asset_name: Name of the asset.
        expected_return: Expected return value.
        asset_list: Full list of asset names.

    Returns:
        Tuple of (view_matrix_row, view_return, view_confidence).
    """
    idx = asset_list.index(asset_name)
    P = np.zeros(len(asset_list))
    P[idx] = 1.0
    return P.reshape(1, -1), np.array([expected_return]), np.array([1.0])


def relative_view(
    outperforming_asset: str,
    underperforming_asset: str,
    expected_spread: float,
    asset_list: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a relative view: "Asset A will outperform Asset B by X%".

    Args:
        outperforming_asset: Asset expected to outperform.
        underperforming_asset: Asset expected to underperform.
        expected_spread: Expected return difference.
        asset_list: Full list of asset names.

    Returns:
        Tuple of (view_matrix_row, view_return, view_confidence).
    """
    P = np.zeros(len(asset_list))
    idx_out = asset_list.index(outperforming_asset)
    idx_under = asset_list.index(underperforming_asset)
    P[idx_out] = 1.0
    P[idx_under] = -1.0
    return P.reshape(1, -1), np.array([expected_spread]), np.array([1.0])
