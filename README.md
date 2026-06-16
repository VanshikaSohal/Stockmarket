# Portfolio Risk Analyzer

A production-grade, end-to-end **stock portfolio risk analysis system** that combines classical financial theory with machine learning and Bayesian inference. Built entirely in Python with a clean modular architecture.

---

## What This Project Does

Given a universe of 11 large-cap US stocks, this system:

1. Downloads 5 years of historical OHLCV data from Yahoo Finance
2. Runs full exploratory data analysis — return distributions, correlations, rolling volatility
3. Computes institutional-grade risk metrics — VaR, CVaR, Sharpe, Sortino, Calmar, Max Drawdown
4. Optimises portfolio weights — Minimum Variance and Maximum Sharpe portfolios via SLSQP
5. Simulates the Efficient Frontier across 5,000 random portfolios
6. Trains ML models (Random Forest + XGBoost) to predict next-day direction and returns
7. Fits ARIMA and `train_ridge_sequence` (sliding-window Ridge regression) for price forecasting
8. Estimates Bayesian posterior return distributions using a conjugate Normal-Inverse-Gamma model

---

## Portfolio Universe

11 stocks across 6 sectors — 2020-01-01 to 2024-12-27 (1,255 trading days)

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple | Technology |
| MSFT | Microsoft | Technology |
| GOOGL | Alphabet | Communication Services |
| AMZN | Amazon | Consumer Discretionary |
| JNJ | Johnson & Johnson | Health Care |
| UNH | UnitedHealth | Health Care |
| PFE | Pfizer | Health Care |
| JPM | JPMorgan Chase | Financials |
| V | Visa | Financials |
| WMT | Walmart | Consumer Staples |
| XOM | Exxon Mobil | Energy |

---

## Key Results

### Risk Metrics (Equal-Weight Portfolio)

| Metric | Value |
|---|---|
| Annualised Return | 18.38% |
| Annualised Volatility | 19.96% |
| Sharpe Ratio | 0.82 |
| Sortino Ratio | 0.78 |
| VaR (95%) | 1.65% daily |
| CVaR (95%) | 3.00% daily |

### ML Forecasting (AAPL Direction Classification)

| Model | Accuracy |
|---|---|
| Random Forest | 52.4% |
| XGBoost | 53.2% |

> Note: >50% on stock direction prediction with only lag features is expected — markets are near-efficient.

### Bayesian Estimates (Annualised, shrinkage-adjusted)

| Ticker | Return Estimate |
|---|---|
| AAPL | 24.31% |
| AMZN | 21.04% |
| GOOGL | 22.27% |
| JNJ | 11.48% |
| JPM | 18.66% |

---

## Visualizations Produced

### Notebook 02 — EDA

**Return Distributions** — histogram + KDE for all 11 stocks  
Shows which stocks have fat tails (PFE, XOM) vs near-normal (WMT, JNJ)

**Risk-Return Scatter** — each stock plotted as Annualised Return vs Annualised Volatility  
Instantly shows which stocks give the best return per unit of risk

**Correlation Heatmap** — pairwise correlation matrix  
Tech stocks (AAPL, MSFT, GOOGL, AMZN) are highly correlated (~0.7+); XOM is least correlated with others

**30-Day Rolling Volatility** — all 11 stocks over time  
COVID crash (March 2020) clearly visible as a volatility spike across all names

---

### Notebook 03 — Portfolio Analysis

**Cumulative Returns Chart** — 3 portfolios compared over 5 years  
Equal-Weight vs Minimum Variance vs Maximum Sharpe — shows how optimised portfolios diverge over time

**Drawdown Chart** — how far portfolio fell from its peak at each point  
Red shaded area below zero — COVID drawdown (March 2020) clearly visible

**Rolling 30-Day VaR** — how daily risk changed over time  
Spikes during high-volatility periods (COVID, 2022 rate hike selloff)

**Efficient Frontier** — 5,000 simulated random portfolios  
Each dot is a portfolio; color = Sharpe ratio; the upper-left curve is the frontier  
Shows the risk-return tradeoff and where the optimal portfolios lie

**Correlation Heatmap** — portfolio-level asset correlation

---

### Notebook 04 — ML Forecasting

**XGBoost Feature Importances** — which lag features matter most for AAPL direction  
Recent lags (lag_1, lag_2) dominate; cross-stock features also contribute

**Sequence Model: Predicted vs Actual** — Ridge regression on 60-day windows  
Predicted AAPL price trajectory closely tracks actual prices in the test period

**ARIMA Forecast** — 30-step out-of-sample forecast vs actual log returns  
ARIMA correctly identifies the near-zero mean-reverting nature of returns

---

### Notebook 05 — Bayesian Modeling

**Posterior Predictive Distributions** — 4 tickers (AAPL, AMZN, GOOGL, JNJ)  
Each chart overlays the empirical return histogram with the Bayesian posterior predictive distribution (Student-t)  
95% credible intervals shown as dashed lines — wider for volatile stocks

**MLE vs Bayesian Shrinkage Bar Chart** — all 11 tickers  
Shrinkage pulls extreme estimates toward the grand mean — more conservative and robust estimates for portfolio construction

---

## Project Structure

```
Stockmarket/
├── config.yaml                   ← Central config: tickers, weights, parameters
├── requirements.txt
├── conftest.py                   ← pytest path config
│
├── data/
│   └── raw/
│       ├── stock_data.csv        ← OHLCV for all 11 stocks (1,256 rows)
│       ├── stock_returns.csv     ← Daily simple returns (1,255 rows × 11 cols)
│       └── stock_log_returns.csv ← Daily log returns
│
├── notebooks/
│   ├── 01_data_collection.ipynb  ← Download + save raw data
│   ├── 02_eda.ipynb              ← 4 charts: distributions, scatter, heatmap, rolling vol
│   ├── 03_portfolio_analysis.ipynb ← 5 charts + full risk metrics table
│   ├── 04_ml_forecasting.ipynb   ← 3 charts + RF/XGB/ARIMA results
│   └── 05_bayesian_modeling.ipynb ← 5 charts + posterior tables
│
├── src/
│   ├── utils/
│   │   ├── config.py             ← load_config() — reads config.yaml
│   │   └── helpers.py            ← annualize, lag features, rolling splits
│   ├── data/
│   │   ├── fetch_data.py         ← yfinance download helpers
│   │   └── preprocess.py         ← returns, log returns, technical features, normalization
│   ├── analysis/
│   │   ├── risk_metrics.py       ← VaR, CVaR, Sharpe, Sortino, Calmar, rolling metrics
│   │   └── portfolio.py          ← portfolio returns, drawdown, min-var/max-Sharpe optimization
│   ├── models/
│   │   ├── supervised_ml.py      ← Random Forest + XGBoost (classification + regression)
│   │   └── time_series.py        ← ARIMA, ADF test, train_ridge_sequence (sliding window Ridge)
│   └── visualization/
│       └── plots.py              ← 9 chart functions, all return matplotlib Figure
│
├── tests/
│   ├── test_data.py              ← preprocessing unit tests
│   ├── test_risk_metrics.py      ← risk metric + portfolio unit tests (39 passing)
│   └── test_models.py            ← ML model unit tests
│
└── docs/
    ├── METHODOLOGY.md            ← All formulas: VaR, Sharpe, ARIMA, Bayesian math
    └── PORTFOLIO_SELECTION.md    ← Why these 11 stocks — sector rationale
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run notebooks in order
Open Jupyter with the `stock` kernel, then run:

```
01_data_collection  →  02_eda  →  03_portfolio_analysis  →  04_ml_forecasting  →  05_bayesian_modeling
```

Each notebook reads from `data/raw/` — so after running 01 once, notebooks 02-05 can be run in any order.

### 3. Run tests
```bash
python -m pytest tests/ -v
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | pandas, numpy, yfinance |
| Visualization | matplotlib, seaborn |
| Risk & Finance | scipy (optimization + stats) |
| Time Series | statsmodels (ARIMA, ADF) |
| Machine Learning | scikit-learn (Random Forest, Ridge, StandardScaler) |
| Boosting | xgboost |
| Config | pyyaml |
| Testing | pytest |

---

