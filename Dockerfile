# =============================================================================
# Stage 1: Builder — install all dependencies
# =============================================================================
FROM python:3.10-slim-bookworm AS builder

ENV \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=1 \
  UV_COMPILE_BYTECODE=1

WORKDIR /build

# Install system build deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
  gcc \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /uvx /bin/

# Copy dependency specs
COPY pyproject.toml ./

# Install core + api dependencies (not dev/dl/bayes — keep prod image slim)
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-deps \
      'pydantic>=2.9.0' \
      'pydantic-settings>=2.5.0' \
      'pydantic-yaml>=1.3.0' \
      'fastapi>=0.115.0' \
      'uvicorn[standard]>=0.31.0' \
      'python-multipart>=0.0.9' \
      'httpx>=0.27.0' \
      'structlog>=24.4.0' \
      'prometheus-client>=0.21.0' \
    && uv pip install \
      'pandas==2.2.2' \
      'numpy==1.26.4' \
      'yfinance==0.2.40' \
      'scikit-learn==1.5.0' \
      'xgboost==2.0.3' \
      'pyyaml==6.0.1' \
      'scipy==1.13.0' \
      'statsmodels==0.14.2' \
      'matplotlib==3.9.0' \
      'seaborn==0.13.2' \
      'plotly==5.22.0' \
      'jupyter==1.0.0' \
      'nbformat==5.10.4' \
      'ipykernel==6.29.4'

# =============================================================================
# Stage 2: Runtime — minimal production image
# =============================================================================
FROM python:3.10-slim-bookworm AS runtime

ENV \
  PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  PORT=8000 \
  APP_ENV=production

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY config.yaml conftest.py pyproject.toml ./

# Create non-root user
RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --gid 1001 app && \
    chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import http.client; http.client.HTTPConnection('localhost', ${PORT}).request('GET', '/health'); assert http.client.HTTPConnection('localhost', ${PORT}).getresponse().status == 200" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
