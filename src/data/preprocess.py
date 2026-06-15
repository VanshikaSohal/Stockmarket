"""
Data preprocessing and feature engineering functions.
"""

import numpy as np
import pandas as pd


def clean_data(df):
    """
    Clean a price or returns DataFrame by forward-filling, back-filling, and
    dropping rows where all values are NaN.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    df = df.ffill().bfill()
    df = df.dropna(how="all")
    return df


def calculate_returns(prices):
    """
    Calculate simple (percentage) daily returns from price data.

    Args:
        prices (pd.DataFrame): Price DataFrame (tickers as columns, dates as index).

    Returns:
        pd.DataFrame: Daily simple returns, with the first row (which is NaN)
            dropped.
    """
    return prices.pct_change().dropna()


def calculate_log_returns(prices):
    """
    Calculate log daily returns from price data.

    Args:
        prices (pd.DataFrame): Price DataFrame (tickers as columns, dates as index).

    Returns:
        pd.DataFrame: Daily log returns, with the first row (NaN) dropped.
    """
    return np.log(prices / prices.shift(1)).dropna()


def add_technical_features(prices, windows=(5, 10, 20)):
    """
    Compute a set of technical features for each ticker column in a price DataFrame.

    Features computed per ticker:
        - Simple moving averages for each window size
        - Rolling volatility (std dev of log returns) for each window size
        - Momentum (return over each window)

    Args:
        prices (pd.DataFrame): Price DataFrame (tickers as columns, dates as index).
        windows (tuple of int): Rolling window sizes. Default (5, 10, 20).

    Returns:
        pd.DataFrame: Feature DataFrame. Rows with leading NaNs are dropped.
    """
    features = {}
    log_rets = np.log(prices / prices.shift(1))

    for col in prices.columns:
        for w in windows:
            features[f"{col}_sma_{w}"] = prices[col].rolling(w).mean()
            features[f"{col}_vol_{w}"] = log_rets[col].rolling(w).std()
            features[f"{col}_mom_{w}"] = prices[col].pct_change(w)

    feat_df = pd.DataFrame(features, index=prices.index)
    return feat_df.dropna()


def normalize_returns(returns):
    """
    Z-score normalize each column in a returns DataFrame.

    Args:
        returns (pd.DataFrame): Returns DataFrame.

    Returns:
        pd.DataFrame: Normalized returns with zero mean and unit variance per column.
    """
    return (returns - returns.mean()) / returns.std(ddof=1)
