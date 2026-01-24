# Portfolio Risk Analyzer

## Overview
A comprehensive ML-powered portfolio risk analysis tool.

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for analysis
- `src/`: Reusable source code modules
- `models/`: Saved trained models
- `reports/`: Generated figures and results
- `docs/`: Project documentation

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Run notebooks in order (01 to 05)
2. Or use modules from src/

## Features
- Multi-stock data collection
- Portfolio risk metrics (VaR, CVaR, Sharpe)
- ML forecasting (Random Forest, LSTM)
- Interactive visualizations

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, TensorFlow, Plotly

Start
  │
  ▼
Data Collection
  ├─ Choose tickers (5–10 companies)
  ├─ Download historical stock data (Open, High, Low, Close, Volume)
  └─ Save raw data in folder
  │
  ▼
Data Preprocessing
  ├─ Handle missing values (forward/backward fill)
  ├─ Align dates across all stocks
  ├─ Calculate daily returns: (Today Close - Yesterday Close)/Yesterday Close
  └─ Calculate log returns (optional, for ML)
  │
  ▼
Exploratory Data Analysis (EDA)
  ├─ Visualize stock price trends (line plots)
  ├─ Plot daily returns distribution (histograms)
  ├─ Compute risk metrics: standard deviation, volatility
  ├─ Correlation analysis (if multiple stocks)
  └─ Volume analysis
  │
  ▼
Portfolio Analysis
  ├─ Assign weights to stocks in portfolio
  ├─ Compute portfolio daily returns (weighted sum)
  ├─ Compute portfolio risk (standard deviation of returns)
  ├─ Compare individual stock vs portfolio risk/return
  └─ Optional: Monte Carlo simulation for portfolio outcomes
  │
  ▼
ML Forecasting Layer (Optional)
  ├─ Prepare past 30–60 days returns as input
  ├─ Train ML models: Linear Regression / Random Forest / XGBoost
  ├─ Train Deep Learning models: LSTM / RNN (time-series)
  └─ Predict next-day / next-week returns
  │
  ▼
Bayesian Modeling (Optional)
  ├─ Model returns as probabilistic distribution
  ├─ Compute posterior estimates of expected returns
  └─ Predict probability of gain/loss
  │
  ▼
Dashboard / Full Stack Integration
  ├─ Frontend: Interactive plots / charts
  ├─ Backend: API endpoints for returns, risk, predictions
  └─ Database: Store historical & calculated data
  │
  ▼
Documentation & Summary
  ├─ Describe data cleaning, EDA, portfolio calculation
  ├─ Summarize ML / Bayesian predictions (if applied)
  └─ Highlight skills learned: Python, Pandas, NumPy, ML, Full Stack
  │
  ▼
End
