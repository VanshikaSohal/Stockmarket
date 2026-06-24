"""
Bayesian hierarchical models for financial time series.

Provides:
- Hierarchical linear regression with partial pooling (PyMC)
- Horseshoe prior for sparse regression
- LOO-CV (Pareto-smoothed importance sampling) via ArviZ
- NumPyro fallback for GPU-accelerated inference
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import arviz as az
    import pymc as pm

    _PYMC_AVAILABLE = True
except ImportError:
    _PYMC_AVAILABLE = False

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS

    _NUMPYRO_AVAILABLE = True
except ImportError:
    _NUMPYRO_AVAILABLE = False


def hierarchical_regression(
    X: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    n_samples: int = 2000,
    n_chains: int = 4,
    nuts_kwargs: Optional[Dict[str, Any]] = None,
    backend: str = "pymc",
) -> Dict[str, Any]:
    """
    Hierarchical Bayesian linear regression with partial pooling.

    Model:
        y_i ~ N(beta[group_i]' * X_i, sigma^2)
        beta[g] ~ N(mu_beta, sigma_beta^2)
        mu_beta ~ N(0, 10)
        sigma_beta ~ HalfCauchy(5)
        sigma ~ HalfCauchy(2)

    Args:
        X: Feature matrix (n x p).
        y: Target vector (n x 1).
        group_idx: Group index for each observation (0 to n_groups-1).
        n_groups: Number of distinct groups.
        n_samples: Number of MCMC samples per chain. Default 2000.
        n_chains: Number of MCMC chains. Default 4.
        nuts_kwargs: Optional kwargs for NUTS sampler.
        backend: 'pymc' or 'numpyro'. Default 'pymc'.

    Returns:
        Dict with keys:
            - 'trace': inference trace object
            - 'summary': ArviZ summary DataFrame
            - 'loo': LOO-CV results
            - 'group_effects': posterior mean betas per group
            - 'pooled_effect': posterior mean of mu_beta
            - 'r2': Bayesian R-squared
    """
    if backend == "pymc":
        return _hierarchical_regression_pymc(
            X, y, group_idx, n_groups, n_samples, n_chains, nuts_kwargs or {}
        )
    elif backend == "numpyro":
        return _hierarchical_regression_numpyro(
            X, y, group_idx, n_groups, n_samples, n_chains, nuts_kwargs or {}
        )
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Use 'pymc' or 'numpyro'.")


def _hierarchical_regression_pymc(
    X: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    n_samples: int,
    n_chains: int,
    nuts_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if not _PYMC_AVAILABLE:
        raise ImportError("PyMC is required for 'pymc' backend. Install with: pip install pymc arviz")

    n, p = X.shape

    with pm.Model() as model:
        mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=10.0, shape=p)
        sigma_beta = pm.HalfCauchy("sigma_beta", beta=5.0, shape=p)
        beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=(n_groups, p))
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_offset)

        sigma = pm.HalfCauchy("sigma", beta=2.0)
        mu = pm.math.dot(X, beta[group_idx].T)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=n_samples,
            chains=n_chains,
            nuts_sampler="nutpie",
            random_seed=42,
            progressbar=False,
            **nuts_kwargs,
        )

    summary = az.summary(trace, var_names=["mu_beta", "sigma_beta", "beta", "sigma"])
    loo = az.loo(trace, var_names=["obs"])
    r2 = _bayesian_r2(trace, y, model)

    group_effects = {}
    for g in range(n_groups):
        group_effects[f"group_{g}"] = trace.posterior["beta"].sel(
            beta_dim_0=g
        ).mean(dim=["chain", "draw"]).values

    return {
        "trace": trace,
        "summary": summary,
        "loo": loo,
        "group_effects": group_effects,
        "pooled_effect": trace.posterior["mu_beta"].mean(dim=["chain", "draw"]).values,
        "r2": r2,
    }


def _hierarchical_regression_numpyro(
    X: np.ndarray,
    y: np.ndarray,
    group_idx: np.ndarray,
    n_groups: int,
    n_samples: int,
    n_chains: int,
    nuts_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    if not _NUMPYRO_AVAILABLE:
        raise ImportError("NumPyro is required for 'numpyro' backend. Install with: pip install numpyro jax jaxlib")

    n, p = X.shape

    def model(X, y, group_idx, n_groups):
        mu_beta = numpyro.sample("mu_beta", dist.Normal(jnp.zeros(p), 10.0 * jnp.ones(p)))
        sigma_beta = numpyro.sample("sigma_beta", dist.HalfCauchy(5.0 * jnp.ones(p)))
        beta_offset = numpyro.sample(
            "beta_offset", dist.Normal(jnp.zeros((n_groups, p)), jnp.ones((n_groups, p)))
        )
        beta = mu_beta + sigma_beta * beta_offset
        sigma = numpyro.sample("sigma", dist.HalfCauchy(2.0))
        mu = jnp.sum(X * beta[group_idx], axis=-1)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    import jax.numpy as jnp

    kernel = NUTS(model, **nuts_kwargs)
    mcmc = MCMC(kernel, num_samples=n_samples, num_warmup=n_samples // 2, num_chains=n_chains)
    mcmc.run(X=X, y=y, group_idx=group_idx, n_groups=n_groups)
    mcmc.print_summary()

    trace = mcmc.get_samples(group_by_chain=True)
    summary = _numpyro_to_arviz(mcmc)
    loo = None

    return {
        "trace": trace,
        "summary": summary,
        "loo": loo,
        "mcmc": mcmc,
    }


def _bayesian_r2(trace, y, model) -> float:
    """Compute Bayesian R-squared (Gelman et al 2019)."""
    posterior = trace.posterior
    y_pred = posterior["obs"].mean(dim=["chain", "draw"]).values
    var_resid = ((y - y_pred) ** 2).mean()
    var_pred = y_pred.var()
    return float(var_pred / (var_pred + var_resid + 1e-12))


def _numpyro_to_arviz(mcmc):
    """Placeholder: convert NumPyro MCMC to ArviZ summary."""
    return {"status": "arviz_conversion_requires_pymc"}


def horseshoe_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 2000,
    n_chains: int = 4,
    nuts_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Bayesian regression with horseshoe prior for sparse variable selection.

    Model:
        y_i ~ N(X_i * beta, sigma^2)
        beta_j ~ N(0, tau^2 * lambda_j^2)
        lambda_j ~ HalfCauchy(0, 1)  (local shrinkage)
        tau ~ HalfCauchy(0, 1)       (global shrinkage)
        sigma ~ HalfCauchy(0, 2)

    The horseshoe prior strongly shrinks irrelevant coefficients to zero
    while leaving large signals unshrunk.

    Args:
        X: Feature matrix (n x p).
        y: Target vector (n x 1).
        n_samples: MCMC samples per chain. Default 2000.
        n_chains: Number of chains. Default 4.
        nuts_kwargs: Optional kwargs for NUTS.

    Returns:
        Dict with trace, summary, loo, beta_estimates, and inclusion_prob.
    """
    if not _PYMC_AVAILABLE:
        raise ImportError("PyMC is required. Install with: pip install pymc arviz")

    n, p = X.shape

    with pm.Model() as model:
        sigma = pm.HalfCauchy("sigma", beta=2.0)
        tau = pm.HalfCauchy("tau", beta=1.0)

        lambda_raw = pm.HalfCauchy("lambda_raw", beta=1.0, shape=p)
        lambda_shrink = pm.Deterministic("lambda", lambda_raw * tau)

        c2 = 5.0
        lambda_tilde = pm.Deterministic(
            "lambda_tilde",
            pm.math.sqrt(c2 / (c2 + tau**2 * lambda_raw**2)) * lambda_shrink,
        )

        beta = pm.Normal("beta", mu=0.0, sigma=lambda_tilde, shape=p)
        mu = pm.math.dot(X, beta)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=y)

        trace = pm.sample(
            draws=n_samples,
            chains=n_chains,
            nuts_sampler="nutpie",
            random_seed=42,
            progressbar=False,
            **(nuts_kwargs or {}),
        )

    summary = az.summary(trace, var_names=["beta", "sigma", "tau"])
    loo = az.loo(trace, var_names=["obs"])
    r2 = _bayesian_r2(trace, y, model)

    beta_post = trace.posterior["beta"]
    beta_mean = beta_post.mean(dim=["chain", "draw"]).values
    beta_ci = az.hdi(beta_post, hdi_prob=0.94).values

    inclusion_prob = (np.abs(beta_post.values).mean(axis=(0, 1)) > 0.01).astype(float)

    return {
        "trace": trace,
        "summary": summary,
        "loo": loo,
        "beta_estimates": beta_mean,
        "beta_hdi": beta_ci,
        "inclusion_prob": inclusion_prob,
        "tau_estimate": float(trace.posterior["tau"].mean(dim=["chain", "draw"]).values),
        "r2": r2,
    }


def loo_cv_comparison(
    models: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compare multiple Bayesian models using LOO-CV.

    Computes ELPD (expected log pointwise predictive density) and
    standard errors for model comparison.

    Args:
        models: Dict mapping model names to PyMC trace objects.

    Returns:
        DataFrame with model comparison: elpd_loo, p_loo, elpd_diff, se.
    """
    if not _PYMC_AVAILABLE:
        raise ImportError("PyMC/ArviZ is required.")

    traces = {name: m["trace"] for name, m in models.items()}
    loo_dict = {name: az.loo(trace) for name, trace in traces.items()}

    comparison = az.compare(loo_dict)
    return comparison


def hierarchical_pooling_factor(loo_result) -> float:
    """
    Compute the hierarchical pooling factor from LOO results.

    Derived from the effective number of parameters (p_loo) relative
    to the total number of observations. A value close to 1 indicates
    strong pooling (shrinkage), close to 0 indicates no pooling.

    Args:
        loo_result: ArviZ LOO result object.

    Returns:
        Pooling factor between 0 and 1.
    """
    p_loo = loo_result.p_loo
    n = loo_result.n
    return float(p_loo / n) if n > 0 else 0.0
