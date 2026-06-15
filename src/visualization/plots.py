"""
Visualization functions for portfolio analysis, risk metrics, and ML results.
All functions return the matplotlib Figure object so callers can save or display it.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


def plot_portfolio_performance(returns, title="Cumulative Portfolio Returns", figsize=(12, 5)):
    """
    Plot cumulative portfolio returns over time.

    Args:
        returns (pd.Series): Daily portfolio return series with DatetimeIndex.
        title (str): Plot title.
        figsize (tuple): Figure size. Default (12, 5).

    Returns:
        matplotlib.figure.Figure
    """
    cumulative = (1 + returns).cumprod()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(cumulative.index, cumulative.values, linewidth=1.5, color="steelblue")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (base 1.0)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_risk_return_scatter(mean_returns, volatilities, labels=None, figsize=(8, 6)):
    """
    Plot a risk-return scatter for individual assets.

    Args:
        mean_returns (array-like): Annualized mean returns per asset.
        volatilities (array-like): Annualized volatilities per asset.
        labels (list of str, optional): Asset names.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(volatilities, mean_returns, s=80, color="steelblue", zorder=3)
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (volatilities[i], mean_returns[i]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Risk-Return Scatter")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(corr_matrix, title="Correlation Matrix", figsize=(10, 8)):
    """
    Plot a heatmap of a correlation matrix.

    Args:
        corr_matrix (pd.DataFrame): Correlation matrix.
        title (str): Plot title.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_return_distributions(returns, figsize=(14, 8)):
    """
    Plot return distribution histograms (with KDE) for each asset.

    Args:
        returns (pd.DataFrame): Return DataFrame with tickers as columns.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    n_cols = 4
    n_rows = int(np.ceil(len(returns.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(returns.columns):
        axes[i].hist(returns[col].dropna(), bins=50, density=True, alpha=0.6, color="steelblue")
        returns[col].dropna().plot(kind="kde", ax=axes[i], color="darkblue", linewidth=1.5)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel("Return")
        axes[i].set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Return Distributions", fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_rolling_var(var_series, title="Rolling VaR (95%)", figsize=(12, 4)):
    """
    Plot a rolling VaR series over time.

    Args:
        var_series (pd.Series): Rolling VaR values with DatetimeIndex.
        title (str): Plot title.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(var_series.index, var_series.values, color="crimson", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("VaR")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_drawdown(cumulative_returns, title="Portfolio Drawdown", figsize=(12, 4)):
    """
    Plot the drawdown series.

    Args:
        cumulative_returns (pd.Series): Cumulative return series (base 1.0).
        title (str): Plot title.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(drawdown.index, drawdown.values, 0, color="crimson", alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig


def plot_feature_importances(importances, top_n=20, title="Feature Importances", figsize=(10, 6)):
    """
    Plot a horizontal bar chart of feature importances.

    Args:
        importances (pd.Series): Feature importances indexed by feature name,
            sorted descending.
        top_n (int): Number of top features to display. Default 20.
        title (str): Plot title.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    top = importances.head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top.index, top.values, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


def plot_prediction_vs_actual(y_test, y_pred, title="Predicted vs Actual", figsize=(12, 5)):
    """
    Plot actual vs predicted values from a time series or regression model.

    Args:
        y_test (array-like): Actual values.
        y_pred (array-like): Predicted values.
        title (str): Plot title.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(y_test, label="Actual", linewidth=1.2, color="steelblue")
    ax.plot(y_pred, label="Predicted", linewidth=1.2, linestyle="--", color="crimson")
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_efficient_frontier(mean_returns, cov_matrix, n_portfolios=3000, risk_free_rate=0.02, figsize=(10, 7)):
    """
    Simulate random portfolios and plot the efficient frontier.

    Args:
        mean_returns (pd.Series): Annualized mean returns per asset.
        cov_matrix (pd.DataFrame): Annualized covariance matrix.
        n_portfolios (int): Number of random portfolios to simulate. Default 3000.
        risk_free_rate (float): Annual risk-free rate for Sharpe coloring. Default 0.02.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure
    """
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    rng = np.random.default_rng(42)

    for i in range(n_portfolios):
        w = rng.dirichlet(np.ones(n_assets))
        port_ret = float(w @ mean_returns)
        port_vol = float(np.sqrt(w @ cov_matrix.values @ w))
        sharpe = (port_ret - risk_free_rate) / (port_vol + 1e-12)
        results[0, i] = port_vol
        results[1, i] = port_ret
        results[2, i] = sharpe

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        results[0], results[1], c=results[2], cmap="viridis", alpha=0.5, s=5
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Efficient Frontier (Simulated)")
    plt.tight_layout()
    return fig
