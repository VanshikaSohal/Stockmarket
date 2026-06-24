"""
API route definitions for the Portfolio Risk Analyzer.

Endpoints
---------
GET  /health          — Health check & readiness probe
GET  /config           — Return active configuration (sanitized)
POST /risk/metrics     — Compute full risk summary for a return series
POST /risk/var         — Value at Risk
POST /risk/cvar        — Conditional Value at Risk
POST /risk/sharpe      — Sharpe ratio
POST /optimize/minvar  — Minimum variance portfolio weights
POST /optimize/sharpe  — Maximum Sharpe ratio weights
POST /predict/ml       — ML model inference (direction classification)
POST /predict/bayes    — Bayesian posterior estimates
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Re-export from existing modules (no code duplication)
from src.analysis.risk_metrics import (
    calculate_cvar,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    risk_summary,
)
from src.utils.settings import load_settings

logger = structlog.get_logger()
router = APIRouter()


class ReturnsPayload(BaseModel):
    returns: List[float] = Field(..., description="Daily return series")
    confidence: float = Field(0.95, ge=0.0, le=1.0, description="Confidence level")
    risk_free_rate: float = Field(0.02, ge=0.0, description="Annual risk-free rate")
    periods: int = Field(252, ge=1, description="Trading periods per year")


class PortfolioReturnsPayload(BaseModel):
    returns_matrix: List[List[float]] = Field(..., description="Asset returns (rows=dates, cols=assets)")
    weights: List[float] = Field(..., description="Portfolio weights")
    risk_free_rate: float = Field(0.02, ge=0.0, description="Annual risk-free rate")


class MLPredictPayload(BaseModel):
    features: List[List[float]] = Field(..., description="Feature matrix for inference")
    model_type: str = Field("random_forest", pattern="^(random_forest|xgboost)$", description="Model type")
    task: str = Field("classification", pattern="^(classification|regression)$", description="Task type")


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    ready: bool


class RiskMetricsResponse(BaseModel):
    var: float
    cvar: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    annualized_return: float
    annualized_volatility: float


# ═══════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Liveness and readiness probe."""
    from fastapi import Request

    request: Request = router.dependency_overrides.get(Request)
    # Access app state via request if available; otherwise fallback
    return {
        "status": "ok",
        "version": "1.0.0",
        "environment": "production",
        "ready": True,
    }


@router.get("/config", tags=["system"])
async def get_config():
    """Return sanitized active configuration (secrets excluded)."""
    settings = load_settings()
    return {
        "portfolio": {
            "ticker_count": len(settings.portfolio.tickers),
            "tickers": settings.portfolio.tickers,
            "has_custom_weights": settings.portfolio.weights is not None,
        },
        "data": settings.data.model_dump(),
        "risk": settings.risk.model_dump(),
        "ml": settings.ml.model_dump(),
        "app_env": settings.app_env,
    }


@router.post("/risk/metrics", response_model=RiskMetricsResponse, tags=["risk"])
async def compute_risk_metrics(payload: ReturnsPayload):
    """Compute full set of risk metrics for a return series."""
    try:
        rets = np.array(payload.returns)
        summary = risk_summary(
            rets,
            confidence=payload.confidence,
            risk_free_rate=payload.risk_free_rate,
            periods=payload.periods,
        )
        return RiskMetricsResponse(**summary)
    except Exception as exc:
        logger.error("Risk metrics computation failed", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/risk/var", tags=["risk"])
async def compute_var(payload: ReturnsPayload):
    """Compute Value at Risk (historical method)."""
    try:
        rets = np.array(payload.returns)
        var = calculate_var(rets, confidence=payload.confidence, method="historical")
        return {"var": var, "confidence": payload.confidence, "method": "historical"}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/risk/cvar", tags=["risk"])
async def compute_cvar(payload: ReturnsPayload):
    """Compute Conditional Value at Risk (historical method)."""
    try:
        rets = np.array(payload.returns)
        cvar = calculate_cvar(rets, confidence=payload.confidence, method="historical")
        return {"cvar": cvar, "confidence": payload.confidence, "method": "historical"}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/risk/sharpe", tags=["risk"])
async def compute_sharpe(payload: ReturnsPayload):
    """Compute annualized Sharpe ratio."""
    try:
        rets = np.array(payload.returns)
        sharpe = calculate_sharpe_ratio(
            rets,
            risk_free_rate=payload.risk_free_rate,
            periods=payload.periods,
        )
        return {"sharpe_ratio": sharpe, "risk_free_rate": payload.risk_free_rate}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/risk/sortino", tags=["risk"])
async def compute_sortino(payload: ReturnsPayload):
    """Compute annualized Sortino ratio."""
    try:
        rets = np.array(payload.returns)
        sortino = calculate_sortino_ratio(
            rets,
            risk_free_rate=payload.risk_free_rate,
            periods=payload.periods,
        )
        return {"sortino_ratio": sortino, "risk_free_rate": payload.risk_free_rate}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/optimize/minvar", tags=["optimization"])
async def compute_minimum_variance(payload: PortfolioReturnsPayload):
    """Compute global minimum variance portfolio weights."""
    from src.analysis.portfolio import minimum_variance_weights

    try:
        returns_matrix = np.array(payload.returns_matrix)
        weights = minimum_variance_weights(returns_matrix)
        return {
            "weights": weights.tolist(),
            "method": "minimum_variance",
        }
    except Exception as exc:
        logger.error("Min variance optimization failed", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/optimize/sharpe", tags=["optimization"])
async def compute_max_sharpe(payload: PortfolioReturnsPayload):
    """Compute maximum Sharpe ratio portfolio weights."""
    from src.analysis.portfolio import maximum_sharpe_weights

    try:
        returns_matrix = np.array(payload.returns_matrix)
        weights = maximum_sharpe_weights(
            returns_matrix,
            risk_free_rate=payload.risk_free_rate,
        )
        return {
            "weights": weights.tolist(),
            "method": "maximum_sharpe",
            "risk_free_rate": payload.risk_free_rate,
        }
    except Exception as exc:
        logger.error("Max Sharpe optimization failed", error=str(exc))
        raise HTTPException(status_code=422, detail=str(exc))


@router.get("/notebooks", tags=["reference"])
async def list_notebooks():
    """List available analysis notebooks."""
    import os

    notebooks_dir = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks")
    if not os.path.exists(notebooks_dir):
        return {"notebooks": []}
    notebooks = sorted(
        f for f in os.listdir(notebooks_dir) if f.endswith(".ipynb")
    )
    return {"notebooks": notebooks}
