"""
Module for fetching stock data from Yahoo Finance via yfinance.
"""

import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def fetch_stock_data(tickers, start_date, end_date, auto_adjust=True):
    """
    Fetch OHLCV stock data for multiple tickers from Yahoo Finance.

    Args:
        tickers (list of str): Stock ticker symbols, e.g. ['AAPL', 'MSFT'].
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        auto_adjust (bool): Whether to use auto-adjusted prices. Default True.

    Returns:
        pd.DataFrame: Multi-level column DataFrame with levels (field, ticker).
            Fields: Open, High, Low, Close, Volume.

    Raises:
        ValueError: If no data could be downloaded for any ticker.
    """
    logger.info("Downloading data for %d tickers from %s to %s", len(tickers), start_date, end_date)
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")
    # Flatten column MultiIndex when only one ticker is requested
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = pd.MultiIndex.from_tuples(
            [(col[0], col[1]) for col in raw.columns], names=["field", "ticker"]
        )
    logger.info("Downloaded %d rows", len(raw))
    return raw


def extract_close_prices(raw_data):
    """
    Extract closing prices from a raw OHLCV DataFrame.

    Args:
        raw_data (pd.DataFrame): Multi-level column DataFrame as returned by
            fetch_stock_data.

    Returns:
        pd.DataFrame: DataFrame of closing prices with ticker names as columns.
    """
    if isinstance(raw_data.columns, pd.MultiIndex):
        close = raw_data["Close"]
    else:
        close = raw_data[["Close"]]
    return close


def clean_price_data(prices):
    """
    Forward-fill then back-fill missing price values, then drop any remaining NaNs.

    Args:
        prices (pd.DataFrame): Price DataFrame (tickers as columns, dates as index).

    Returns:
        pd.DataFrame: Cleaned price DataFrame.
    """
    return prices.ffill().bfill().dropna()
