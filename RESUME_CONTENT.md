# Resume Content — Stock Portfolio Risk Analyzer

---

## Project Title Line

**Stock Portfolio Risk Analyzer | Python, Scikit-learn, XGBoost, Statsmodels, SciPy**

---

## Resume Bullets (use these directly)

- Built an end-to-end ML pipeline analyzing risk and return for an 11-stock portfolio (1,255 trading days, 2020–2024); equal-weight portfolio achieved 18.4% annualised return, Sharpe ratio 0.82, and max drawdown −29%; computed VaR(95%) = 1.65%, CVaR = 3.0%, Sortino = 0.78, Calmar = 0.63.

- Optimised portfolio weights using SciPy SLSQP; Maximum Sharpe portfolio achieved Sharpe 1.08, annualised return 24.1%, and cumulative return +199% vs +126% for equal-weight, with max drawdown reduced from −29% to −20.9%; implemented efficient frontier via 5,000 Monte Carlo simulations.

- Trained Random Forest and XGBoost on 55 lag-based features across 1,250 samples; XGBoost achieved 53.2% direction classification accuracy; implemented ARIMA baseline (ADF p-value < 0.0001) and sliding-window Ridge regression for time series forecasting (MAE = 3.62).

- Applied Bayesian inference with Normal-Inverse-Gamma conjugate model and James-Stein shrinkage estimator; AAPL posterior annualised return estimated at 24.3% with 95% credible interval [−3.8%, +4.0%] daily; shrinkage applied across all 11 tickers for robust cross-asset return estimation.

---

## All Key Numbers (for interviews / talking points)

### Dataset
| Metric | Value |
|---|---|
| Stocks | 11 large-cap US equities across 6 sectors |
| Date range | 2020-01-03 to 2024-12-27 |
| Trading days | 1,255 |
| Total observations | 13,805 (1,255 × 11) |

### Individual Stock Performance (Annualised, 2020–2024)
| Ticker | Return | Volatility | Sharpe |
|---|---|---|---|
| AAPL | 30.2% | 31.7% | 0.89 |
| MSFT | 25.4% | 30.5% | 0.77 |
| GOOGL | 26.2% | 32.5% | 0.74 |
| AMZN | 23.7% | 36.0% | 0.60 |
| WMT | 20.9% | 22.6% | 0.84 |
| JPM | 18.9% | 32.6% | 0.52 |
| XOM | 19.0% | 34.4% | 0.49 |
| UNH | 17.1% | 29.9% | 0.51 |
| V | 14.9% | 27.9% | 0.46 |
| JNJ | 4.6% | 19.7% | 0.13 |
| PFE | 1.4% | 27.5% | −0.02 |

### Portfolio Risk Metrics (Equal-Weight)
| Metric | Value |
|---|---|
| Annualised Return | 18.38% |
| Annualised Volatility | 19.96% |
| Sharpe Ratio | 0.82 |
| Sortino Ratio | 0.78 |
| Calmar Ratio | 0.63 |
| VaR (95%, daily) | 1.65% |
| CVaR (95%, daily) | 3.00% |
| Max Drawdown | −29.02% |
| Cumulative Return (5yr) | +126.1% |

### Portfolio Comparison
| Portfolio | Ann. Return | Sharpe | Max Drawdown | 5yr Cumulative |
|---|---|---|---|---|
| Equal Weight | 18.38% | 0.82 | −29.0% | +126.1% |
| Minimum Variance | 18.38% | 0.82 | −29.0% | +126.1% |
| Maximum Sharpe | 24.11% | 1.08 | −20.9% | +199.0% |

### ML Models (AAPL Direction Classification)
| Model | Accuracy | Task |
|---|---|---|
| Random Forest | 52.4% | Direction (up/down) |
| XGBoost | 53.2% | Direction (up/down) |
| Random Forest | MAE 0.0105, RMSE 0.0144 | Return regression |
| XGBoost | MAE 0.0108, RMSE 0.0145 | Return regression |
| Ridge Sequence Model | MAE 3.62, RMSE 4.54 | Price forecasting |
| ARIMA(1,0,1) | ADF p < 0.0001 (stationary) | Returns baseline |

> ML features: 55 features (5 lags × 11 stocks), 1,250 training samples

### Bayesian Estimates (Annualised, James-Stein Shrinkage)
| Ticker | MLE Return | Shrinkage Estimate |
|---|---|---|
| AAPL | 30.2% | 24.3% |
| AMZN | 23.7% | 21.0% |
| GOOGL | 26.2% | 22.3% |
| MSFT | 25.4% | 21.9% |
| WMT | 20.9% | 19.6% |
| JPM | 18.9% | 18.7% |
| XOM | 19.0% | 18.7% |
| UNH | 17.1% | 17.8% |
| V | 14.9% | 16.6% |
| JNJ | 4.6% | 11.5% |
| PFE | 1.4% | 9.9% |

### Bayesian Credible Intervals (95%, Daily Returns)
| Ticker | Lower | Posterior Mean | Upper |
|---|---|---|---|
| AAPL | −3.79% | +0.12% | +4.03% |
| AMZN | −4.35% | +0.09% | +4.54% |
| GOOGL | −3.91% | +0.10% | +4.12% |
| JNJ | −2.41% | +0.02% | +2.45% |
| WMT | −2.71% | +0.08% | +2.87% |

---

## 3 Best Numbers to Say in an Interview

1. **+199% cumulative return** — Maximum Sharpe optimised portfolio over 5 years (2020–2024)
2. **53.2% direction accuracy** — XGBoost on near-efficient market data with only lag features
3. **13,805 daily observations** — 1,255 trading days × 11 stocks processed end-to-end
