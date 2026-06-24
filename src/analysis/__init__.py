from src.analysis.portfolio import (
    calculate_correlation_matrix,
    calculate_covariance_matrix,
    calculate_cumulative_returns,
    calculate_drawdown,
    calculate_max_drawdown,
    calculate_portfolio_returns,
    calculate_portfolio_volatility,
    maximum_sharpe_weights,
    minimum_variance_weights,
    portfolio_summary,
)
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

# ── Phase 2A: Volatility ───────────────────────────────────────────────────
from src.analysis.garch import (
    dcc_garch_estimate,
    fit_egarch,
    fit_garch,
    fit_gjr_garch,
    forecast_volatility,
    garch_summary,
)
from src.analysis.realized_vol import (
    bipower_variation,
    fit_har_rv,
    forecast_har_rv,
    jump_component,
    prepare_har_features,
    realized_quarticity,
    realized_variance,
    realized_volatility,
)

# ── Phase 2B: Advanced Optimization ──────────────────────────────────────
from src.analysis.black_litterman import (
    absolute_view,
    black_litterman_estimate,
    market_implied_returns,
    relative_view,
)
from src.analysis.hrp import (
    hrp_portfolio_summary,
    hrp_weights,
)
from src.analysis.cvar_opt import (
    cvar_optimal_weights,
    cvar_risk_contribution,
    efficient_frontier_cvar,
)

__all__ = [
    "calculate_portfolio_returns",
    "calculate_portfolio_volatility",
    "calculate_cumulative_returns",
    "calculate_drawdown",
    "calculate_max_drawdown",
    "calculate_correlation_matrix",
    "calculate_covariance_matrix",
    "minimum_variance_weights",
    "maximum_sharpe_weights",
    "portfolio_summary",
    "calculate_var",
    "calculate_cvar",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_calmar_ratio",
    "rolling_var",
    "rolling_sharpe",
    "risk_summary",
    # Phase 2A
    "fit_garch",
    "fit_egarch",
    "fit_gjr_garch",
    "forecast_volatility",
    "garch_summary",
    "dcc_garch_estimate",
    "realized_volatility",
    "realized_variance",
    "bipower_variation",
    "realized_quarticity",
    "jump_component",
    "prepare_har_features",
    "fit_har_rv",
    "forecast_har_rv",
    # Phase 2B
    "market_implied_returns",
    "black_litterman_estimate",
    "absolute_view",
    "relative_view",
    "hrp_weights",
    "hrp_portfolio_summary",
    "cvar_optimal_weights",
    "cvar_risk_contribution",
    "efficient_frontier_cvar",
]
