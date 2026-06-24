"""
Hierarchical Risk Parity (HRP) portfolio optimization.

Implements the López de Prado (2016) algorithm:

1. Tree Clustering: Build a correlation-based dendrogram.
2. Quasi-Diagonalization: Reorder the covariance matrix.
3. Recursive Bisection: Split weights inversely to variance.

This approach is robust to ill-conditioned covariance matrices
and produces diversified portfolios without quadratic programming.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _cov_to_corr(cov: pd.DataFrame) -> pd.DataFrame:
    """Convert covariance matrix to correlation matrix."""
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


def _corr_to_dist(corr: pd.DataFrame) -> np.ndarray:
    """Convert correlation matrix to distance matrix."""
    dist = np.sqrt(2.0 * (1.0 - corr))
    return squareform(dist, checks=False)


def _get_quasi_diag(link: np.ndarray, items: List[str]) -> List[str]:
    """
    Quasi-diagonalize the covariance matrix using the linkage tree (Lopez de Prado, 2016).

    Recursively traverses the dendrogram from the root, concatenating left
    and right child orderings so that similar assets are adjacent.

    Args:
        link: Linkage matrix from scipy clustering.
        items: List of asset names.

    Returns:
        Ordered list of asset names.
    """
    link = link.astype(int)
    n = len(items)

    def _order(node: int) -> List[int]:
        if node < n:
            return [node]
        left = int(link[node - n][0])
        right = int(link[node - n][1])
        return _order(left) + _order(right)

    sorted_idx = _order(2 * n - 2)
    return [items[i] for i in sorted_idx]


def _get_cluster_var(cov: pd.DataFrame, items: List[str]) -> float:
    """Compute the variance of a cluster (diagonal approach)."""
    sub_cov = cov.loc[items, items]
    w = np.ones(len(items)) / len(items)
    return float(w @ sub_cov @ w)


def hrp_weights(cov: pd.DataFrame) -> pd.Series:
    """
    Compute Hierarchical Risk Parity portfolio weights.

    Args:
        cov: Covariance matrix (assets x assets).

    Returns:
        Series of portfolio weights that sum to 1.0.

    Example:
        >>> cov = pd.DataFrame({
        ...     "A": [0.1, 0.02], "B": [0.02, 0.12]
        ... }, index=["A", "B"])
        >>> w = hrp_weights(cov)
        >>> w.sum()
        1.0
    """
    corr = _cov_to_corr(cov)
    dist = _corr_to_dist(corr)

    link = linkage(dist, method="ward", optimal_ordering=True)
    sorted_items = _get_quasi_diag(link, list(cov.columns))

    weights = pd.Series(1.0, index=sorted_items)

    clusters = [sorted_items]
    while len(clusters) > 0:
        cluster = clusters.pop(0)
        if len(cluster) == 1:
            continue
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]

        var_left = _get_cluster_var(cov, left)
        var_right = _get_cluster_var(cov, right)
        alpha = 1.0 - var_left / (var_left + var_right)

        for item in left:
            weights[item] *= alpha
        for item in right:
            weights[item] *= (1.0 - alpha)

        clusters.append(left)
        clusters.append(right)

    weights = weights / weights.sum()
    return weights.loc[list(cov.columns)]


def hrp_portfolio_summary(
    returns: pd.DataFrame,
    weights: pd.Series,
    risk_free_rate: float = 0.02,
    periods: int = 252,
) -> Dict[str, float]:
    """
    Compute portfolio summary statistics for an HRP-weighted portfolio.

    Args:
        returns: Asset return DataFrame.
        weights: HRP weights (from ``hrp_weights``).
        risk_free_rate: Annual risk-free rate. Default 0.02.
        periods: Trading periods per year. Default 252.

    Returns:
        Dict of portfolio metrics.
    """
    from src.analysis.portfolio import portfolio_summary

    return portfolio_summary(returns, weights, risk_free_rate=risk_free_rate, periods=periods)
