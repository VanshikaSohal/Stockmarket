"""
Unit tests for risk_metrics and portfolio modules.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.risk_metrics import (
    calculate_calmar_ratio,
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    risk_summary,
    rolling_sharpe,
    rolling_var,
)
from src.analysis.portfolio import (
    calculate_cumulative_returns,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_portfolio_returns,
    calculate_portfolio_volatility,
    portfolio_summary,
)


def make_returns(n=500, mu=0.0005, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(mu, sigma, size=(n, 3))
    return pd.DataFrame(data, index=idx, columns=["A", "B", "C"])


class TestVaR:
    def test_historical_positive(self):
        rets = make_returns()
        var = calculate_var(rets["A"], confidence=0.95)
        assert var > 0

    def test_parametric_positive(self):
        rets = make_returns()
        var = calculate_var(rets["A"], confidence=0.95, method="parametric")
        assert var > 0

    def test_higher_confidence_higher_var(self):
        rets = make_returns()
        var95 = calculate_var(rets["A"], confidence=0.95)
        var99 = calculate_var(rets["A"], confidence=0.99)
        assert var99 >= var95

    def test_unknown_method_raises(self):
        rets = make_returns()
        with pytest.raises(ValueError):
            calculate_var(rets["A"], method="unknown")


class TestCVaR:
    def test_cvar_geq_var(self):
        rets = make_returns()
        var = calculate_var(rets["A"], confidence=0.95)
        cvar = calculate_cvar(rets["A"], confidence=0.95)
        assert cvar >= var - 1e-9

    def test_parametric_positive(self):
        rets = make_returns()
        cvar = calculate_cvar(rets["A"], confidence=0.95, method="parametric")
        assert cvar > 0


class TestSharpeRatio:
    def test_positive_for_positive_excess_returns(self):
        rng = np.random.default_rng(1)
        rets = pd.Series(rng.normal(0.001, 0.01, 500))
        assert calculate_sharpe_ratio(rets, risk_free_rate=0.0) > 0

    def test_zero_vol_returns_zero(self):
        rets = pd.Series([0.0] * 100)
        assert calculate_sharpe_ratio(rets) == 0.0


class TestSortinoRatio:
    def test_sortino_geq_sharpe_for_positive_returns(self):
        rng = np.random.default_rng(2)
        rets = pd.Series(rng.normal(0.001, 0.01, 500))
        sharpe = calculate_sharpe_ratio(rets, risk_free_rate=0.0)
        sortino = calculate_sortino_ratio(rets, risk_free_rate=0.0)
        # Sortino should be >= Sharpe when returns are mostly positive
        assert sortino >= sharpe - 1e-6


class TestCalmarRatio:
    def test_positive_for_trending_up(self):
        # Strictly increasing series has zero drawdown, Calmar undefined; use a noisy series
        rng = np.random.default_rng(3)
        rets = pd.Series(rng.normal(0.001, 0.008, 500))
        calmar = calculate_calmar_ratio(rets)
        assert calmar > 0


class TestRollingVaR:
    def test_output_length(self):
        rets = make_returns()["A"]
        rv = rolling_var(rets, window=20)
        assert len(rv) == len(rets)

    def test_all_nonnegative_after_window(self):
        rets = make_returns()["A"]
        rv = rolling_var(rets, window=20).dropna()
        assert (rv >= 0).all()


class TestRollingSharpe:
    def test_output_length(self):
        rets = make_returns()["A"]
        rs = rolling_sharpe(rets, window=60)
        assert len(rs) == len(rets)


class TestRiskSummary:
    def test_keys_present(self):
        rets = make_returns()["A"]
        summary = risk_summary(rets)
        for key in ("var", "cvar", "sharpe_ratio", "sortino_ratio", "calmar_ratio",
                    "annualized_return", "annualized_volatility"):
            assert key in summary

    def test_values_are_floats(self):
        rets = make_returns()["A"]
        summary = risk_summary(rets)
        for k, v in summary.items():
            assert isinstance(v, float), f"{k} is not a float"


class TestPortfolioReturns:
    def test_shape(self):
        rets = make_returns()
        w = np.array([0.5, 0.3, 0.2])
        port = calculate_portfolio_returns(rets, w)
        assert len(port) == len(rets)

    def test_weights_must_sum_to_one(self):
        rets = make_returns()
        with pytest.raises(ValueError):
            calculate_portfolio_returns(rets, [0.5, 0.5, 0.5])

    def test_equal_weight_between_bounds(self):
        rets = make_returns()
        w = [1 / 3, 1 / 3, 1 / 3]
        port = calculate_portfolio_returns(rets, w)
        # portfolio return should be within min/max of individual asset returns
        assert (port >= rets.min(axis=1) - 1e-12).all()
        assert (port <= rets.max(axis=1) + 1e-12).all()


class TestPortfolioVolatility:
    def test_positive(self):
        rets = make_returns()
        w = [1 / 3, 1 / 3, 1 / 3]
        vol = calculate_portfolio_volatility(rets, weights=w)
        assert vol > 0


class TestCumulativeReturns:
    def test_starts_at_one_plus_first_return(self):
        rets = make_returns()["A"]
        cum = calculate_cumulative_returns(rets)
        assert abs(cum.iloc[0] - (1 + rets.iloc[0])) < 1e-12

    def test_monotone_for_positive_returns(self):
        rets = pd.Series([0.01] * 50)
        cum = calculate_cumulative_returns(rets)
        assert (cum.diff().dropna() > 0).all()


class TestDrawdown:
    def test_always_nonpositive(self):
        rets = make_returns()["A"]
        cum = calculate_cumulative_returns(rets)
        dd = calculate_drawdown(cum)
        assert (dd <= 1e-12).all()

    def test_max_drawdown_nonpositive(self):
        rets = make_returns()["A"]
        cum = calculate_cumulative_returns(rets)
        assert calculate_max_drawdown(cum) <= 0


class TestPortfolioSummary:
    def test_keys(self):
        rets = make_returns()
        w = [1 / 3, 1 / 3, 1 / 3]
        s = portfolio_summary(rets, w)
        for k in ("annualized_return", "annualized_volatility", "sharpe_ratio",
                  "max_drawdown", "cumulative_return"):
            assert k in s

    def test_max_drawdown_nonpositive(self):
        rets = make_returns()
        w = [1 / 3, 1 / 3, 1 / 3]
        s = portfolio_summary(rets, w)
        assert s["max_drawdown"] <= 0
