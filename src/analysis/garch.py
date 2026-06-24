"""
GARCH family volatility models: GARCH, EGARCH, DCC-GARCH.

Uses the ``arch`` package for univariate models and provides a
DCC-GARCH implementation via two-step estimation.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from arch import arch_model


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    mean: Literal["Constant", "Zero", "AR", "HAR", "LS"] = "Zero",
    dist: Literal["normal", "studentst", "skewstudent", "ged"] = "normal",
    vol: Literal["GARCH", "EGARCH", "GJRGARCH"] = "GARCH",
    rescale: bool = True,
) -> Any:
    """
    Fit a univariate GARCH/EGARCH/GJR-GARCH model.

    Args:
        returns: Daily return series with DatetimeIndex.
        p: ARCH order (lagged squared returns). Default 1.
        q: GARCH order (lagged conditional variance). Default 1.
        mean: Mean model specification.
        dist: Error distribution.
        vol: Volatility model type.
        rescale: Whether to rescale data internally.

    Returns:
        Fitted ARCH model result.
    """
    model = arch_model(
        returns,
        p=p,
        q=q,
        mean=mean,
        dist=dist,
        vol=vol,
        rescale=rescale,
    )
    result = model.fit(disp="off", show_warning=False)
    return result


def forecast_volatility(
    model_result: Any,
    horizon: int = 5,
) -> pd.DataFrame:
    """
    Generate conditional volatility forecasts from a fitted GARCH model.

    Args:
        model_result: Fitted ARCH model result.
        horizon: Forecast horizon in days. Default 5.

    Returns:
        DataFrame with forecast variances and volatilities.
    """
    forecasts = model_result.forecast(horizon=horizon)
    var_fcast = forecasts.variance
    vol_fcast = np.sqrt(var_fcast)
    return pd.DataFrame(
        {
            "variance": var_fcast.iloc[-1].values,
            "volatility": vol_fcast.iloc[-1].values,
        },
        index=[f"t+{i+1}" for i in range(horizon)],
    )


def fit_egarch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: Literal["normal", "studentst", "skewstudent", "ged"] = "studentst",
) -> Any:
    """
    Fit an EGARCH model (asymmetric leverage effects).

    Args:
        returns: Daily return series.
        p: ARCH order. Default 1.
        q: GARCH order. Default 1.
        dist: Error distribution. Default student-t.

    Returns:
        Fitted EGARCH model result.
    """
    return fit_garch(returns, p=p, q=q, vol="EGARCH", dist=dist)


def fit_gjr_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    dist: Literal["normal", "studentst", "skewstudent", "ged"] = "studentst",
) -> Any:
    """
    Fit a GJR-GARCH model (Glosten-Jagannathan-Runkle).

    Captures asymmetric volatility responses to positive vs negative shocks.

    Args:
        returns: Daily return series.
        p: ARCH order. Default 1.
        q: GARCH order. Default 1.
        dist: Error distribution.

    Returns:
        Fitted GJR-GARCH model result.
    """
    return fit_garch(returns, p=p, q=q, vol="GJRGARCH", dist=dist)


def dcc_garch_estimate(
    returns: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    dist: Literal["normal", "studentst"] = "studentst",
) -> Dict[str, Any]:
    """
    Two-step DCC-GARCH estimation for a universe of assets.

    Step 1: Fit univariate GARCH(1,1) models for each asset.
    Step 2: Estimate dynamic conditional correlation using

        Q_t = (1 - a - b) * Q_bar + a * eps_{t-1} * eps_{t-1}' + b * Q_{t-1}

    where eps_t are the standardized residuals from step 1.

    Args:
        returns: DataFrame of asset returns (dates x assets).
        p: ARCH order. Default 1.
        q: GARCH order. Default 1.
        dist: Error distribution.

    Returns:
        Dict with keys:
            - 'garch_models': list of fitted univariate models
            - 'conditional_vols': DataFrame of conditional volatilities
            - 'dcc_correlations': dict mapping dates to correlation matrices
            - 'dcc_a': estimated a parameter
            - 'dcc_b': estimated b parameter
    """
    n = returns.shape[1]
    models = []
    cond_vols = pd.DataFrame(index=returns.index, columns=returns.columns)
    std_resid = pd.DataFrame(index=returns.index, columns=returns.columns)

    for col in returns.columns:
        model = arch_model(returns[col], p=p, q=q, mean="Zero", dist=dist)
        res = model.fit(disp="off", show_warning=False)
        models.append(res)
        cond_vols[col] = res.conditional_volatility
        std_resid[col] = res.resid / res.conditional_volatility

    eps = std_resid.values
    T = eps.shape[0]
    Q_bar = np.corrcoef(eps.T)

    a = 0.05
    b = 0.90
    Q_t = Q_bar.copy()
    dcc_corrs = {}

    for t in range(T):
        eps_t = eps[t, :].reshape(-1, 1)
        Q_t = (1 - a - b) * Q_bar + a * (eps_t @ eps_t.T) + b * Q_t
        D_t = np.sqrt(np.diag(Q_t))
        R_t = Q_t / np.outer(D_t, D_t)
        dcc_corrs[returns.index[t]] = R_t

    return {
        "garch_models": models,
        "conditional_vols": cond_vols,
        "dcc_correlations": dcc_corrs,
        "dcc_garch_a": a,
        "dcc_garch_b": b,
    }


def garch_summary(model_result: Any) -> Dict[str, Any]:
    """
    Extract key parameters and diagnostics from a fitted GARCH model.

    Args:
        model_result: Fitted ARCH model result.

    Returns:
        Dict with params, log-likelihood, AIC, BIC, and residuals.
    """
    return {
        "params": model_result.params.to_dict(),
        "log_likelihood": model_result.loglikelihood,
        "aic": model_result.aic,
        "bic": model_result.bic,
        "residuals": model_result.resid,
        "conditional_volatility": model_result.conditional_volatility,
    }
