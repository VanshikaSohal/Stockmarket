"""
Shared utility functions used across the project.
"""

import numpy as np
import pandas as pd


def annualize_returns(returns, periods=252):
    """
    Annualize mean daily returns.

    Args:
        returns (pd.Series or np.ndarray): Daily return series.
        periods (int): Trading days per year. Default 252.

    Returns:
        float: Annualized return.
    """
    return float(np.mean(returns) * periods)


def annualize_volatility(returns, periods=252):
    """
    Annualize volatility (standard deviation) of daily returns.

    Args:
        returns (pd.Series or np.ndarray): Daily return series.
        periods (int): Trading days per year. Default 252.

    Returns:
        float: Annualized volatility.
    """
    return float(np.std(returns, ddof=1) * np.sqrt(periods))


def create_lagged_features(series, lags):
    """
    Create lagged feature columns from a time series.

    Args:
        series (pd.Series): Input time series with a DatetimeIndex.
        lags (int or list of int): Number of lags, or explicit list of lag values.

    Returns:
        pd.DataFrame: DataFrame containing lag columns named 'lag_1', 'lag_2', etc.,
            with rows that contain NaN (from lag creation) dropped.
    """
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    df = pd.DataFrame({"target": series})
    for lag in lags:
        df[f"lag_{lag}"] = series.shift(lag)
    return df.dropna()


def rolling_window_split(n_samples, train_size, step=1):
    """
    Generate (train_end, test_end) index pairs for a rolling window walk-forward split.

    Args:
        n_samples (int): Total number of observations.
        train_size (int): Fixed training window size.
        step (int): Step between successive test positions.

    Yields:
        tuple: (train_indices, test_index) where train_indices is a slice and
            test_index is a single integer.
    """
    for start in range(train_size, n_samples, step):
        train = list(range(start - train_size, start))
        test = start
        if test < n_samples:
            yield train, test
