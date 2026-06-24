"""
FastAPI application entry point.

Provides REST endpoints for:
- Portfolio risk metrics
- Portfolio optimization
- ML model inference
- Bayesian return estimates
- Health checks & monitoring

Start with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from src.api.auth import get_current_user
from src.api.middleware import configure_middleware, configure_rate_limiting
from src.utils.settings import AppSettings, load_settings

# ── Structured logging ─────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.dev.ConsoleRenderer() if __debug__ else structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


# ── Application context ────────────────────────────────────────────────────


class AppContext:
    """Shared application state accessible from route handlers."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    def mark_ready(self) -> None:
        self._ready = True


# ═══════════════════════════════════════════════════════════════════════════
# Lifespan handler (replaces on_event for FastAPI 0.115+)
# ═══════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    ctx: AppContext = app.state.ctx
    log = logger.bind(env=ctx.settings.app_env)

    log.info("Starting Portfolio Risk Analyzer API", port=ctx.settings.api_port)

    # Warm up: validate settings load
    _ = load_settings()
    log.info("Configuration validated")

    ctx.mark_ready()
    log.info("Application ready")

    yield

    log.info("Shutting down")


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    if settings is None:
        settings = load_settings()

    ctx = AppContext(settings)

    app = FastAPI(
        title="Portfolio Risk Analyzer API",
        description=(
            "Institutional-grade quantitative finance pipeline. "
            "Compute risk metrics, optimize portfolios, run ML forecasts, "
            "and perform Bayesian inference on multi-asset portfolios."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Aditya Thetsu",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
    )

    app.state.ctx = ctx

    # ── CORS ────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Middleware ───────────────────────────────────────────────────────
    configure_middleware(app)

    # ── Rate limiting ───────────────────────────────────────────────────
    configure_rate_limiting(app)

    # ── Prometheus metrics ──────────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Register routers ────────────────────────────────────────────────
    from src.api.routes import router as api_router

    app.include_router(api_router, prefix="/api/v1")

    # ── Auth dependency override ─────────────────────────────────────────
    if not settings.api_auth_enabled:
        app.dependency_overrides[get_current_user] = lambda: {"sub": "anonymous", "role": "admin"}

    return app


# ═══════════════════════════════════════════════════════════════════════════
# Module-level app instance for ASGI servers
# ═══════════════════════════════════════════════════════════════════════════

app = create_app()
