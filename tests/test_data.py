"""
Unit tests for the data pipeline: fetch_data and preprocess modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    add_technical_features,
    calculate_log_returns,
    calculate_returns,
    clean_data,
    normalize_returns,
)


def make_price_df(n=100, tickers=("A", "B", "C"), seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0, 0.01, size=(n, len(tickers))), axis=0),
        index=idx,
        columns=list(tickers),
    )
    return prices


class TestCleanData:
    def test_forward_fill(self):
        prices = make_price_df(n=10)
        prices.iloc[3, 0] = np.nan
        cleaned = clean_data(prices)
        assert not cleaned.isnull().any().any()

    def test_drops_all_nan_rows(self):
        prices = make_price_df(n=10)
        prices.iloc[5] = np.nan
        cleaned = clean_data(prices)
        assert not cleaned.isnull().all(axis=1).any()

    def test_does_not_mutate_input(self):
        prices = make_price_df(n=10)
        prices.iloc[2, 1] = np.nan
        original = prices.copy()
        clean_data(prices)
        pd.testing.assert_frame_equal(prices, original)


class TestCalculateReturns:
    def test_shape(self):
        prices = make_price_df(n=50)
        rets = calculate_returns(prices)
        assert rets.shape == (49, 3)

    def test_no_nans(self):
        prices = make_price_df(n=50)
        rets = calculate_returns(prices)
        assert not rets.isnull().any().any()

    def test_first_row_dropped(self):
        prices = make_price_df(n=50)
        rets = calculate_returns(prices)
        assert rets.index[0] == prices.index[1]


class TestCalculateLogReturns:
    def test_shape(self):
        prices = make_price_df(n=50)
        log_rets = calculate_log_returns(prices)
        assert log_rets.shape == (49, 3)

    def test_values_approximately_equal_to_simple_for_small_returns(self):
        prices = make_price_df(n=200, seed=0)
        simple = calculate_returns(prices)
        log = calculate_log_returns(prices)
        # log returns ≈ simple returns for small moves
        np.testing.assert_allclose(log.values, simple.values, atol=0.005)

    def test_no_nans(self):
        prices = make_price_df(n=50)
        assert not calculate_log_returns(prices).isnull().any().any()


class TestAddTechnicalFeatures:
    def test_columns_created(self):
        prices = make_price_df(n=100)
        feats = add_technical_features(prices, windows=(5, 10))
        tickers = list(prices.columns)
        for t in tickers:
            for w in (5, 10):
                assert f"{t}_sma_{w}" in feats.columns
                assert f"{t}_vol_{w}" in feats.columns
                assert f"{t}_mom_{w}" in feats.columns

    def test_no_nans_after_dropna(self):
        prices = make_price_df(n=100)
        feats = add_technical_features(prices, windows=(5, 20))
        assert not feats.isnull().any().any()

    def test_row_count_reduced_by_max_window(self):
        prices = make_price_df(n=100)
        feats = add_technical_features(prices, windows=(5, 20))
        assert len(feats) <= len(prices) - 20


class TestNormalizeReturns:
    def test_zero_mean(self):
        prices = make_price_df(n=100)
        rets = calculate_returns(prices)
        normed = normalize_returns(rets)
        np.testing.assert_allclose(normed.mean().values, np.zeros(3), atol=1e-10)

    def test_unit_std(self):
        prices = make_price_df(n=100)
        rets = calculate_returns(prices)
        normed = normalize_returns(rets)
        np.testing.assert_allclose(normed.std(ddof=1).values, np.ones(3), atol=1e-10)
