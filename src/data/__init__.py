# Lazy imports: fetch_data requires yfinance (network-dependent).
# Import directly from submodules when needed, e.g.:
#   from src.data.fetch_data import fetch_stock_data
#   from src.data.preprocess import calculate_returns

from src.data.preprocess import (
    add_technical_features,
    calculate_log_returns,
    calculate_returns,
    clean_data,
    normalize_returns,
)

__all__ = [
    "add_technical_features",
    "calculate_log_returns",
    "calculate_returns",
    "clean_data",
    "normalize_returns",
    # fetch_data exports available via: from src.data.fetch_data import ...
    "fetch_stock_data",
    "extract_close_prices",
    "clean_price_data",
]
