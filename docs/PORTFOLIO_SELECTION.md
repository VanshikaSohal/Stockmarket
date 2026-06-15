# Portfolio Selection Rationale

## Objective

Construct a diversified 11-stock universe for risk analysis across multiple sectors,
providing enough breadth to demonstrate correlation structure, efficient frontier
optimisation, and cross-asset ML feature engineering.

## Selection Criteria

1. Sector diversification across at least 5 GICS sectors
2. Market capitalisation > $50B (large-cap, high liquidity)
3. Continuous daily price history from 2020-01-01 onward
4. Low incidence of missing data or trading halts

## Selected Companies

| Ticker | Company | Sector | Config Weight |
|---|---|---|---|
| AAPL | Apple | Technology | 12% |
| MSFT | Microsoft | Technology | 12% |
| GOOGL | Alphabet | Communication Services | 12% |
| AMZN | Amazon | Consumer Discretionary | 12% |
| JNJ | Johnson & Johnson | Health Care | 10% |
| UNH | UnitedHealth Group | Health Care | 8% |
| PFE | Pfizer | Health Care | 8% |
| JPM | JPMorgan Chase | Financials | 10% |
| V | Visa | Financials | 10% |
| WMT | Walmart | Consumer Staples | 8% |
| XOM | Exxon Mobil | Energy | 8% |

## Rationale by Sector

**Technology (AAPL, MSFT, GOOGL, AMZN)** — Highest growth and liquidity; forms the
core of the portfolio. Correlation within this group is high, which is an interesting
stress-test for diversification metrics.

**Health Care (JNJ, UNH, PFE)** — Defensive names that tend to be uncorrelated with
technology during risk-off periods. PFE provides COVID-era volatility that enriches
the dataset.

**Financials (JPM, V)** — Interest-rate sensitive; adds macro factor exposure.

**Consumer Staples (WMT)** — Low-beta, inflation-resilient anchor.

**Energy (XOM)** — Commodity-linked; acts as an inflation and geopolitical risk hedge.
