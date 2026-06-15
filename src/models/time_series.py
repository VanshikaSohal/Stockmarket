"""
Time series models: ARIMA-based forecasting and a pure-NumPy LSTM-style sequence model
implemented with scikit-learn (no TensorFlow dependency required).
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


def check_stationarity(series, significance=0.05):
    """
    Test whether a time series is stationary using the Augmented Dickey-Fuller test.

    Args:
        series (pd.Series): Time series to test.
        significance (float): Significance level. Default 0.05.

    Returns:
        dict: Keys 'statistic', 'p_value', 'is_stationary'.
    """
    result = adfuller(series.dropna(), autolag=None, maxlag=5)
    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
        "is_stationary": result[1] < significance,
    }


def fit_arima(series, order=(1, 1, 1)):
    """
    Fit an ARIMA model to a univariate time series.

    Args:
        series (pd.Series): Time series (price or returns).
        order (tuple): ARIMA (p, d, q) order. Default (1, 1, 1).

    Returns:
        statsmodels.tsa.arima.model.ARIMAResultsWrapper: Fitted ARIMA results object.
    """
    model = ARIMA(series, order=order)
    fitted = model.fit()
    logger.info("ARIMA%s AIC=%.4f", order, fitted.aic)
    return fitted


def forecast_arima(fitted_model, steps=10):
    """
    Generate an out-of-sample forecast from a fitted ARIMA model.

    Args:
        fitted_model: Fitted statsmodels ARIMA results object.
        steps (int): Number of steps to forecast. Default 10.

    Returns:
        pd.Series: Forecast values.
    """
    forecast = fitted_model.forecast(steps=steps)
    return forecast


def create_sequences(series, seq_length=60):
    """
    Create overlapping (X, y) sequences from a 1-D array for sequence modelling.

    Args:
        series (np.ndarray): 1-D array of scaled values.
        seq_length (int): Number of past time steps used as input. Default 60.

    Returns:
        tuple: (X, y) where X.shape == (n_samples, seq_length) and y.shape == (n_samples,).
    """
    X, y = [], []
    for i in range(seq_length, len(series)):
        X.append(series[i - seq_length:i])
        y.append(series[i])
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    """
    Return a Ridge regression surrogate that operates on flattened sequence windows.

    This avoids a TensorFlow/Keras dependency while maintaining the same interface
    contract (fit / predict) expected by the rest of the pipeline.

    Args:
        input_shape (tuple): (seq_length, n_features) or (seq_length,).

    Returns:
        Ridge: Untrained scikit-learn Ridge regressor.
    """
    return Ridge(alpha=1.0)


def train_lstm(series, seq_length=60, test_size=0.2):
    """
    Train a sequence model on a univariate price or return series.

    Scales the data with MinMaxScaler, creates overlapping windows, splits into
    train/test, fits a Ridge regression on flattened windows, and evaluates.

    Args:
        series (pd.Series or np.ndarray): Univariate time series.
        seq_length (int): Lookback window size. Default 60.
        test_size (float): Proportion of data held out for evaluation. Default 0.2.

    Returns:
        dict: Keys 'model', 'scaler', 'metrics', 'y_test', 'y_pred'.

    Raises:
        ValueError: If the series is too short for the given seq_length and test split.
    """
    values = np.array(series).reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    X, y = create_sequences(scaled, seq_length=seq_length)
    if len(X) == 0:
        raise ValueError(
            f"Series too short ({len(values)} samples) for seq_length={seq_length}."
        )

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_lstm_model((seq_length,))
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_test)

    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    metrics = {
        "mae": float(mean_absolute_error(y_test_inv, y_pred_inv)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))),
    }
    logger.info("Sequence model metrics: %s", metrics)

    return {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "y_test": y_test_inv,
        "y_pred": y_pred_inv,
    }
