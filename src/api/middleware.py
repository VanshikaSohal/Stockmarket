"""
API middleware: rate limiting, request logging, and error handling.

Uses slowapi for rate limiting with an in-memory backend and provides
custom middleware for structured request logging.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI, Request, Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

import structlog

logger = structlog.get_logger()

# ── Rate limiter ───────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


def configure_rate_limiting(app: FastAPI) -> None:
    """Attach the rate limiter to a FastAPI application."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Request logging middleware ──────────────────────────────────────────────


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        log = logger.bind(
            method=request.method,
            path=request.url.path,
            query=str(request.url.query),
            client_host=request.client.host if request.client else "unknown",
        )
        log.info("Request started")

        import time

        start = time.time()
        try:
            response = await call_next(request)
            duration_ms = (time.time() - start) * 1000
            log.info(
                "Request completed",
                status_code=response.status_code,
                duration_ms=f"{duration_ms:.1f}",
            )
            return response
        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            log.error(
                "Request failed",
                error=str(exc),
                duration_ms=f"{duration_ms:.1f}",
            )
            raise


def configure_middleware(app: FastAPI) -> None:
    """Add all custom middleware to the FastAPI application."""
    app.add_middleware(RequestLoggingMiddleware)
