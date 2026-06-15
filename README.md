# Portfolio Risk Analyzer

A production-grade portfolio risk analysis system that combines classical financial
theory with machine learning to deliver data-driven risk insights.

## Project Structure

```
Stockmarket/
├── config.yaml                  # Central configuration (tickers, weights, params)
├── requirements.txt
├── conftest.py                  # pytest root config
├── data/
│   ├── raw/                     # CSV files produced by notebook 01
│   └── processed/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_portfolio_analysis.ipynb
│   ├── 04_ml_forecasting.ipynb
│   └── 05_bayesian_modeling.ipynb
├── src/
│   ├── data/
│   │   ├── fetch_data.py        # yfinance download helpers
│   │   └── preprocess.py        # returns, log returns, feature engineering
│   ├── analysis/
│   │   ├── portfolio.py         # portfolio returns, vol, drawdown, optimisation
│   │   └── risk_metrics.py      # VaR, CVaR, Sharpe, Sortino, Calmar, rolling
│   ├── models/
│   │   ├── supervised_ml.py     # Random Forest and XGBoost (classification / regression)
│   │   └── time_series.py       # ARIMA, sequence model (Ridge on sliding windows)
│   ├── utils/
│   │   ├── config.py            # YAML config loader
│   │   └── helpers.py           # annualization, lag features, rolling split
│   └── visualization/
│       └── plots.py             # matplotlib/seaborn chart functions
├── tests/
│   ├── test_data.py
│   ├── test_risk_metrics.py
│   └── test_models.py
├── reports/
│   ├── figures/
│   └── results/
└── docs/
    ├── METHODOLOGY.md
    └── PORTFOLIO_SELECTION.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Notebooks

Run notebooks **in order** from the `notebooks/` directory (or set the kernel CWD to
the project root):

| Notebook | Purpose |
|---|---|
| 01_data_collection | Download OHLCV data and save raw CSVs |
| 02_eda | Return distributions, correlation, rolling volatility |
| 03_portfolio_analysis | VaR, CVaR, Sharpe, drawdown, efficient frontier |
| 04_ml_forecasting | RF / XGBoost direction + return models; ARIMA baseline |
| 05_bayesian_modeling | Conjugate NIG posterior, shrinkage estimates |

## Running the Tests

```bash
python -m pytest tests/ -v
```

## Key Features

- Full OHLCV data pipeline via yfinance with forward/back-fill cleaning
- Annualised risk metrics: VaR (historical + parametric), CVaR, Sharpe, Sortino, Calmar
- Portfolio optimisation: minimum variance and maximum Sharpe weights (scipy)
- Efficient frontier simulation
- ML forecasting: Random Forest and XGBoost (direction classification + return regression)
- Time-series modelling: ARIMA baseline and sliding-window Ridge sequence model
- Bayesian return estimation: Normal-Inverse-Gamma conjugate posterior
- Modular `src/` library — all notebook logic backed by importable, tested functions

## Tech Stack

Python 3.12+, pandas, numpy, scipy, statsmodels, scikit-learn, xgboost, yfinance,
matplotlib, seaborn, pyyaml, jupyter
