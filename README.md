# Portfolio Risk Analyzer

[![CI](https://github.com/athtetsu/Stockmarket/actions/workflows/ci.yml/badge.svg)](https://github.com/athtetsu/Stockmarket/actions/workflows/ci.yml)
[![Docker](https://github.com/athtetsu/Stockmarket/actions/workflows/docker.yml/badge.svg)](https://github.com/athtetsu/Stockmarket/actions/workflows/docker.yml)
[![codecov](https://codecov.io/gh/athtetsu/Stockmarket/branch/main/graph/badge.svg)](https://codecov.io/gh/athtetsu/Stockmarket)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/container-Docker-2496ED.svg)](https://docker.com)

> An institutional-grade, end-to-end quantitative finance pipeline that combines classical portfolio theory, machine learning forecasting, and Bayesian inference to deliver actionable risk insights across an 11-stock universe.

---

## Why This Project Matters

Most retail finance tools stop at price charts. This system goes further — it simulates the kind of pipeline used by quantitative analysts and risk desks at hedge funds and asset managers:

- **Classical finance**: VaR, CVaR, Sharpe, Sortino, Calmar, drawdown, efficient frontier
- **Portfolio optimisation**: SLSQP-based minimum variance and maximum Sharpe weight allocation
- **Machine learning**: Random Forest and XGBoost for direction classification and return regression
- **Time series**: ARIMA baseline and sliding-window Ridge regression for price forecasting
- **Bayesian inference**: Normal-Inverse-Gamma conjugate posteriors with James-Stein shrinkage

This is not a tutorial project. Every module is production-structured, tested, and backed by real market data.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        config.yaml                                  │
│           (tickers, weights, dates, confidence levels)              │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  01  DATA COLLECTION                                                │
│      yfinance → 11 stocks × 1,255 days → OHLCV CSVs               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  02  PREPROCESSING & EDA                                            │
│      Simple returns · Log returns · Technical features             │
│      Distributions · Correlation matrix · Rolling volatility       │
└──────────┬────────────────────────────────────────┬────────────────┘
           │                                        │
           ▼                                        ▼
┌─────────────────────────┐            ┌────────────────────────────┐
│  03  PORTFOLIO ANALYSIS │            │  04  ML FORECASTING        │
│                         │            │                            │
│  · Min Variance weights │            │  · Random Forest (52.4%)   │
│  · Max Sharpe weights   │            │  · XGBoost (53.2%)         │
│  · Efficient Frontier   │            │  · ARIMA(1,0,1) baseline   │
│  · VaR / CVaR / Sharpe  │            │  · Ridge sequence model    │
│  · Sortino / Calmar     │            │    (MAE = 3.62)            │
│  · Drawdown analysis    │            │                            │
└─────────────────────────┘            └────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│  05  BAYESIAN MODELING                                              │
│      Normal-Inverse-Gamma posteriors · James-Stein shrinkage       │
│      Posterior predictive distributions · Credible intervals       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Portfolio Universe

11 large-cap US equities across 6 GICS sectors — 2020-01-03 to 2024-12-27

| Ticker | Company | Sector | Ann. Return | Sharpe |
|---|---|---|---|---|
| AAPL | Apple | Technology | 30.2% | 0.89 |
| MSFT | Microsoft | Technology | 25.4% | 0.77 |
| GOOGL | Alphabet | Communication | 26.2% | 0.74 |
| AMZN | Amazon | Consumer Disc. | 23.7% | 0.60 |
| WMT | Walmart | Consumer Staples | 20.9% | 0.84 |
| JPM | JPMorgan | Financials | 18.9% | 0.52 |
| XOM | Exxon Mobil | Energy | 19.0% | 0.49 |
| UNH | UnitedHealth | Health Care | 17.1% | 0.51 |
| V | Visa | Financials | 14.9% | 0.46 |
| JNJ | J&J | Health Care | 4.6% | 0.13 |
| PFE | Pfizer | Health Care | 1.4% | −0.02 |

---

## Key Results

### Portfolio Optimisation

| Portfolio | Ann. Return | Sharpe | Max Drawdown | 5yr Cumulative |
|---|---|---|---|---|
| Equal Weight | 18.4% | 0.82 | −29.0% | +126.1% |
| Minimum Variance | 18.4% | 0.82 | −29.0% | +126.1% |
| **Maximum Sharpe** | **24.1%** | **1.08** | **−20.9%** | **+199.0%** |

### Risk Metrics (Equal-Weight Portfolio)

| Metric | Value | What it means |
|---|---|---|
| VaR (95%) | 1.65% daily | On 95% of days, loss will not exceed 1.65% |
| CVaR (95%) | 3.00% daily | Average loss on the worst 5% of days |
| Sharpe Ratio | 0.82 | 0.82 units of return per unit of risk |
| Sortino Ratio | 0.78 | Penalises only downside volatility |
| Calmar Ratio | 0.63 | Annual return relative to worst drawdown |
| Max Drawdown | −29.0% | Worst peak-to-trough decline (COVID crash) |

### ML Forecasting (AAPL)

| Model | Task | Result |
|---|---|---|
| Random Forest | Direction classification | 52.4% accuracy |
| XGBoost | Direction classification | 53.2% accuracy |
| Random Forest | Return regression | MAE 0.0105 |
| XGBoost | Return regression | MAE 0.0108 |
| Ridge Sequence | Price forecasting | MAE 3.62 |
| ARIMA(1,0,1) | Returns baseline | ADF p < 0.0001 |

---

## Visual Insights

### Return Distributions — Understanding tail risk
![Return Distributions](./reports/figures/return_distributions.png)

---

### Risk–Return Tradeoff — Which stocks are efficient?
![Risk Return Scatter](./reports/figures/risk_return_scatter.png)

---

### Correlation Structure — Diversification check
![Correlation Heatmap](./reports/figures/correlation_heatmap.png)

---

### Efficient Frontier — Portfolio optimization landscape
![Efficient Frontier](./reports/figures/efficient_frontier.png)

---

### Portfolio Performance — Does optimization work?
![Cumulative Returns](./reports/figures/cumulative_returns.png)

---

### Risk Control — Drawdown analysis
![Drawdown](./reports/figures/drawdown.png)

---

### Market Stress — Rolling volatility
![Rolling Volatility](./reports/figures/rolling_volatility.png)


## Project Structure

```
Stockmarket/
├── config.yaml                        ← Tickers, weights, risk parameters
├── requirements.txt
├── pyproject.toml                     ← Modern packaging (uv)
├── Dockerfile                         ← Multi-stage production build
├── docker-compose.yml                 ← Orchestrated services
├── Makefile                           ← Common dev tasks
├── conftest.py
├── .env.example                       ← Environment variable template
├── .pre-commit-config.yaml            ← Pre-commit hooks (ruff, mypy, bandit)
├── .editorconfig                      ← Editor settings
├── .github/workflows/
│   ├── ci.yml                         ← Lint → typecheck → test → build
│   └── docker.yml                     ← Build & push Docker image
├── data/
│   └── raw/
│       ├── stock_data.csv             ← OHLCV — 11 stocks × 1,256 rows
│       ├── stock_returns.csv          ← Daily simple returns
│       └── stock_log_returns.csv      ← Daily log returns
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_portfolio_analysis.ipynb
│   ├── 04_ml_forecasting.ipynb
│   └── 05_bayesian_modeling.ipynb
├── src/
│   ├── api/                           ← FastAPI REST endpoints
│   │   ├── __init__.py
│   │   ├── main.py                    ← App factory with lifespan
│   │   └── routes.py                  ← /health, /risk, /optimize, /config
│   ├── utils/         config.py · settings.py · helpers.py
│   ├── data/          fetch_data.py · preprocess.py
│   ├── analysis/      risk_metrics.py · portfolio.py
│   ├── models/        supervised_ml.py · time_series.py
│   └── visualization/ plots.py
├── tests/
│   ├── test_data.py
│   ├── test_risk_metrics.py
│   └── test_models.py
├── reports/figures/                   ← All 17 exported charts
└── docs/
    ├── METHODOLOGY.md
    └── PORTFOLIO_SELECTION.md
```

### New Files (Phase 0 — Foundation)

| File | Purpose |
|------|---------|
| `pyproject.toml` | Modern packaging with optional dependency groups (dev, api, dl, bayes, ml) |
| `Dockerfile` | Multi-stage Docker build (builder → runtime) |
| `docker-compose.yml` | API + dev + Jupyter services |
| `Makefile` | Common dev tasks (`make test`, `make lint`, `make docker-up`) |
| `.github/workflows/ci.yml` | CI: lint → typecheck → test (matrix) → build |
| `.github/workflows/docker.yml` | Docker build & push to GHCR |
| `.pre-commit-config.yaml` | Pre-commit hooks: ruff, black, isort, mypy, bandit |
| `.editorconfig` | Cross-editor consistency |
| `.env.example` | Environment variable template |
| `src/utils/settings.py` | Pydantic-settings with YAML + env var + validation |
| `src/api/main.py` | FastAPI app factory with Prometheus metrics, CORS, structured logging |
| `src/api/routes.py` | REST endpoints for risk metrics, optimization, config |

---

## How to Run

```bash
# ── Option 1: uv (recommended) ────────────────────────────────────────

# Install uv (one-time):  https://docs.astral.sh/uv/#installation
# Bootstrap project
make init

# Alternatively, manually:
uv sync --group dev --all-extras

# Run notebooks
uv run jupyter notebook

# Run tests
uv run pytest tests/ -v

# ── Option 2: pip ─────────────────────────────────────────────────────

pip install -r requirements.txt
jupyter notebook
python -m pytest tests/ -v

# ── Option 3: Docker ──────────────────────────────────────────────────

docker compose up -d api          # Start API on port 8000
docker compose up notebook        # Start Jupyter on port 8888
docker compose up dev             # Start dev server with hot-reload

# ── Option 4: API server ──────────────────────────────────────────────

make api-dev                      # Dev server with hot-reload
# OR
uv run uvicorn src.api.main:app --reload --port 8000
```

> All notebooks use the `stock` kernel (Python 3.10). After running notebook 01 once, notebooks 02–05 can be run independently.

---

## Tech Stack

| Layer | Libraries |
|---|---|
| Data | pandas, numpy, yfinance |
| Visualisation | matplotlib, seaborn |
| Optimisation | scipy (SLSQP) |
| Statistics | scipy.stats, statsmodels (ARIMA, ADF) |
| Machine Learning | scikit-learn (Random Forest, Ridge, StandardScaler) |
| Boosting | xgboost |
| Config | pyyaml, **pydantic + pydantic-settings** |
| API | **FastAPI**, **uvicorn** |
| Metrics | **prometheus-client** |
| Logging | **structlog** |
| Packaging | **uv**, **pyproject.toml** |
| Code Quality | **ruff**, **black**, **isort**, **mypy**, **pre-commit** |
| CI/CD | **GitHub Actions** (lint → typecheck → test → build) |
| Deployment | **Docker** (multi-stage), **docker-compose** |
| Testing | pytest (39+ tests) |

---


