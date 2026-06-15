# Methodology

## Data Collection

- **Source**: Yahoo Finance via yfinance
- **Universe**: 11 large-cap US equities across 6 sectors
- **Period**: 2020-01-01 to 2024-12-30 (approximately 1,256 trading days)
- **Frequency**: Daily OHLCV
- **Cleaning**: Forward-fill then back-fill missing values; drop columns still fully NaN

## Feature Engineering

- **Simple returns**: `(P_t - P_{t-1}) / P_{t-1}`
- **Log returns**: `ln(P_t / P_{t-1})`
- **Technical features**: rolling SMA, rolling volatility, momentum (windows: 5, 10, 20 days)
- **Lag features**: past N days of returns for each asset (used as ML inputs)

## Risk Metrics

| Metric | Formula | Notes |
|---|---|---|
| VaR (historical) | `-percentile(r, 1-confidence)` | 95% and 99% confidence levels |
| VaR (parametric) | `-(mu + sigma * z_alpha)` | Assumes normality |
| CVaR / ES | `E[r | r <= VaR]` | Average loss beyond VaR |
| Sharpe ratio | `(mu_excess / sigma) * sqrt(252)` | Risk-free rate 2% p.a. |
| Sortino ratio | `(mu_excess / sigma_downside) * sqrt(252)` | Uses downside deviation only |
| Calmar ratio | `(mu_annual) / |max_drawdown|` | — |
| Max drawdown | `min((cum_t - cum_max_t) / cum_max_t)` | — |

## Portfolio Optimisation

Weights are chosen by minimising portfolio variance (global minimum variance) or
maximising the Sharpe ratio, both subject to `sum(w) = 1` and `w_i >= 0`.
Scipy `minimize` with SLSQP is used.

## ML Forecasting

### Direction Classification
- **Target**: binary (1 = positive next-day return, 0 = negative)
- **Features**: lagged returns of all assets (5 lags × 11 assets = 55 features)
- **Models**: Random Forest (100 trees, max_depth 5) and XGBoost (100 rounds, lr 0.05)
- **Split**: chronological 80/20, no shuffling

### Return Regression
- Same feature set; target is the raw next-day return value

### Sequence Model
- MinMaxScaler normalisation → overlapping 60-step windows → Ridge regression
- Acts as an LSTM surrogate with no deep-learning dependency

### ARIMA Baseline
- Statsmodels ARIMA(1,0,1) on return series
- ADF test used to confirm approximate stationarity before fitting

## Bayesian Modelling

Uses the Normal-Inverse-Gamma conjugate model:

- **Prior**: `(mu_0, kappa_0, alpha_0, beta_0)` with weakly informative hyperparameters
- **Posterior**: closed-form update; marginal posterior of `mu` is Student-t
- **Shrinkage**: James-Stein style blending of sample mean toward the grand cross-asset mean
