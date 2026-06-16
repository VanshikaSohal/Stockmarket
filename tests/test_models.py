"""
Unit tests for supervised_ml and time_series model modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.supervised_ml import (
    build_features_and_target,
    get_feature_importances,
    train_random_forest,
    train_xgboost,
)
from src.models.time_series import (
    check_stationarity,
    create_sequences,
    fit_arima,
    forecast_arima,
    train_ridge_sequence,
)


def make_returns(n=300, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0005, 0.01, size=(n, 4))
    return pd.DataFrame(data, index=idx, columns=["AAPL", "MSFT", "GOOG", "JPM"])


class TestBuildFeaturesAndTarget:
    def test_direction_binary(self):
        rets = make_returns()
        X, y, names = build_features_and_target(rets, "AAPL", lags=3, target="direction")
        assert set(y).issubset({0, 1})

    def test_return_target_continuous(self):
        rets = make_returns()
        X, y, names = build_features_and_target(rets, "AAPL", lags=3, target="return")
        assert y.dtype in (np.float32, np.float64)

    def test_feature_count(self):
        rets = make_returns()
        X, y, names = build_features_and_target(rets, "AAPL", lags=5, target="direction")
        expected_features = len(rets.columns) * 5
        assert X.shape[1] == expected_features
        assert len(names) == expected_features

    def test_no_nans(self):
        rets = make_returns()
        X, y, _ = build_features_and_target(rets, "AAPL", lags=5)
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()


class TestTrainRandomForest:
    def test_classification_accuracy_range(self):
        rets = make_returns()
        X, y, _ = build_features_and_target(rets, "AAPL", lags=5, target="direction")
        result = train_random_forest(X, y, task="classification")
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    def test_regression_metrics_present(self):
        rets = make_returns()
        X, y, _ = build_features_and_target(rets, "AAPL", lags=5, target="return")
        result = train_random_forest(X, y, task="regression")
        for key in ("mae", "rmse", "r2"):
            assert key in result["metrics"]

    def test_result_keys(self):
        rets = make_returns()
        X, y, _ = build_features_and_target(rets, "AAPL", lags=3)
        result = train_random_forest(X, y)
        for key in ("model", "scaler", "metrics", "X_test", "y_test", "y_pred"):
            assert key in result


class TestTrainXGBoost:
    def test_classification_accuracy_range(self):
        rets = make_returns()
        X, y, _ = build_features_and_target(rets, "AAPL", lags=5, target="direction")
        result = train_xgboost(X, y, task="classification")
        assert 0.0 <= result["metrics"]["accuracy"] <= 1.0

    def test_regression_metrics_present(self):
        rets = make_returns()
        X, y, _ = build_features_and_target(rets, "AAPL", lags=5, target="return")
        result = train_xgboost(X, y, task="regression")
        for key in ("mae", "rmse", "r2"):
            assert key in result["metrics"]


class TestGetFeatureImportances:
    def test_returns_series(self):
        rets = make_returns()
        X, y, names = build_features_and_target(rets, "AAPL", lags=5)
        result = train_random_forest(X, y)
        imp = get_feature_importances(result, names)
        assert hasattr(imp, "index")
        assert len(imp) == len(names)

    def test_sorted_descending(self):
        rets = make_returns()
        X, y, names = build_features_and_target(rets, "AAPL", lags=5)
        result = train_xgboost(X, y)
        imp = get_feature_importances(result, names)
        assert (imp.values == np.sort(imp.values)[::-1]).all()


class TestCheckStationarity:
    def test_stationary_white_noise(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 1, 300))
        result = check_stationarity(s)
        assert result["is_stationary"] is True

    def test_nonstationary_random_walk(self):
        rng = np.random.default_rng(0)
        s = pd.Series(np.cumsum(rng.normal(0, 1, 300)))
        result = check_stationarity(s)
        # Not guaranteed but very likely with a long random walk
        assert isinstance(result["p_value"], float)

    def test_keys(self):
        s = pd.Series(np.random.randn(100))
        result = check_stationarity(s)
        for k in ("statistic", "p_value", "is_stationary"):
            assert k in result


class TestCreateSequences:
    def test_shapes(self):
        data = np.arange(100, dtype=float)
        X, y = create_sequences(data, seq_length=10)
        assert X.shape == (90, 10)
        assert y.shape == (90,)

    def test_alignment(self):
        data = np.arange(20, dtype=float)
        X, y = create_sequences(data, seq_length=5)
        # First window: [0,1,2,3,4], target: 5
        np.testing.assert_array_equal(X[0], [0, 1, 2, 3, 4])
        assert y[0] == 5.0


class TestFitAndForecastARIMA:
    def test_forecast_length(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 0.01, 150))
        fitted = fit_arima(s, order=(1, 0, 1))
        fc = forecast_arima(fitted, steps=10)
        assert len(fc) == 10

    def test_forecast_returns_series(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.normal(0, 0.01, 100))
        fitted = fit_arima(s, order=(0, 0, 1))
        fc = forecast_arima(fitted, steps=5)
        assert isinstance(fc, pd.Series)


class TestTrainRidgeSequence:
    def test_metrics_present(self):
        rng = np.random.default_rng(5)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
        result = train_ridge_sequence(prices, seq_length=20, test_size=0.2)
        for key in ("mae", "rmse"):
            assert key in result["metrics"]

    def test_prediction_length_matches_test(self):
        rng = np.random.default_rng(5)
        prices = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
        result = train_ridge_sequence(prices, seq_length=20, test_size=0.2)
        assert len(result["y_pred"]) == len(result["y_test"])

    def test_too_short_raises(self):
        prices = pd.Series([100.0] * 10)
        with pytest.raises(ValueError):
            train_ridge_sequence(prices, seq_length=60)
