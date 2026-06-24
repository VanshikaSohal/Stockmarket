# ──────────────────────────────────────────────────────────────────────────────
# Portfolio Risk Analyzer — Makefile
# ──────────────────────────────────────────────────────────────────────────────
# Common development tasks. Requires Python ≥ 3.10 and uv.
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help install lint format typecheck test coverage build docker-dev docker-build docker-up clean

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Dependencies ──────────────────────────────────────────────────────────────

install: ## Install core + dev dependencies
	uv sync --group dev --all-extras

install-prod: ## Install only core dependencies
	uv sync --no-dev

install-api: ## Install core + API dependencies
	uv sync --group api

install-dl: ## Install core + deep learning dependencies
	uv sync --group dl

install-bayes: ## Install core + Bayesian dependencies
	uv sync --group bayes

# ── Code Quality ──────────────────────────────────────────────────────────────

format: ## Format code with ruff
	uv run ruff format src/ tests/ conftest.py

lint: ## Lint with ruff
	uv run ruff check src/ tests/ conftest.py

lint-fix: ## Lint and auto-fix
	uv run ruff check --fix src/ tests/ conftest.py

isort: ## Sort imports with isort
	uv run isort src/ tests/ conftest.py

typecheck: ## Type check with mypy
	uv run mypy src/ conftest.py

check: lint isort typecheck ## Run all code quality checks

# ── Testing ────────────────────────────────────────────────────────────────────

test: ## Run all tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage report
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

test-xdist: ## Run tests in parallel
	uv run pytest tests/ -v -n auto --dist loadscope

test-quick: ## Run tests, fail fast on first error
	uv run pytest tests/ -v -x

coverage: test-cov ## Alias for test coverage

coverage-html: ## Generate HTML coverage report
	uv run pytest tests/ --cov=src --cov-report=html
	@echo "Open htmlcov/index.html in your browser."

# ── Pre-commit ────────────────────────────────────────────────────────────────

precommit-install: ## Install pre-commit hooks
	uv run pre-commit install

precommit-run: ## Run pre-commit on all files
	uv run pre-commit run --all-files

precommit-update: ## Update pre-commit hooks to latest versions
	uv run pre-commit autoupdate

# ── Pre-commit ────────────────────────────────────────────────────────────────

build: ## Build the Python package
	uv build

publish: ## Publish to PyPI (requires API token)
	uv publish

# ── Docker ────────────────────────────────────────────────────────────────────

docker-build: ## Build Docker image
	docker compose build api

docker-up: ## Start API service in background
	docker compose up -d api

docker-dev: ## Start development server with hot-reload
	docker compose up dev

docker-down: ## Stop all containers
	docker compose down

docker-logs: ## Tail API logs
	docker compose logs -f api

docker-clean: ## Remove all containers, networks, images
	docker compose down --rmi all --volumes

# ── API Server ────────────────────────────────────────────────────────────────

api-dev: ## Start FastAPI dev server with hot-reload
	uv run uvicorn src.api.main:app --reload --port 8000

api-prod: ## Start FastAPI production server
	uv run gunicorn src.api.main:app \
		--worker-class uvicorn.workers.UvicornWorker \
		--workers 4 \
		--bind 0.0.0.0:8000 \
		--access-logfile -

# ── Notebooks ─────────────────────────────────────────────────────────────────

jupyter: ## Start Jupyter notebook server
	uv run jupyter notebook --notebook-dir=notebooks

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean: ## Clean build artifacts and caches
	rm -rf dist/ build/ *.egg-info/ .mypy_cache/ .ruff_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

# ── Housekeeping ──────────────────────────────────────────────────────────────

deps-outdated: ## Show outdated dependencies
	uv pip list --outdated

deps-audit: ## Audit dependencies for known vulnerabilities
	uv pip audit

init: install precommit-install ## Bootstrap project from scratch
