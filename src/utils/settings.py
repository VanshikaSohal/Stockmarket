"""
Type-safe application settings powered by Pydantic.

Provides validated, documented configuration that can be loaded from
YAML files, environment variables, or .env files with schema validation
and autocomplete in modern IDEs.

Usage:
    from src.utils.settings import AppSettings, load_settings

    settings = load_settings()
    settings.portfolio.tickers         # validated list of tickers
    settings.risk.var_confidence       # 0.95
    settings.ml.sequence_length        # 60

The existing ``load_config`` from ``config.py`` continues to work as before;
this module is an additive layer with type safety.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, PositiveInt, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from yaml import safe_load as yaml_safe_load


# =============================================================================
# Nested configuration models
# =============================================================================


class DataConfig(BaseModel):
    start_date: str = Field("2020-01-01", description="Start date for data download (YYYY-MM-DD)")
    end_date: str = Field("2024-12-30", description="End date for data download (YYYY-MM-DD)")


class RiskMetricsConfig(BaseModel):
    var_confidence: float = Field(0.95, ge=0.0, le=1.0, description="VaR confidence level")
    cvar_confidence: float = Field(0.95, ge=0.0, le=1.0, description="CVaR confidence level")
    rolling_window: PositiveInt = Field(30, description="Rolling window size (trading days)")
    annualization_factor: PositiveInt = Field(252, description="Trading days per year for annualization")


class MLParamsConfig(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=1.0, description="Fraction of data held out for testing")
    random_state: PositiveInt = Field(42, description="Random seed for reproducibility")
    sequence_length: PositiveInt = Field(60, description="Lookback window for sequence models")
    n_lags: PositiveInt = Field(5, description="Number of lag features for ML models")


class PortfolioConfig(BaseModel):
    tickers: List[str] = Field(
        default=[
            "AAPL", "MSFT", "GOOGL", "JNJ", "JPM",
            "XOM", "AMZN", "V", "UNH", "WMT", "PFE",
        ],
        min_length=1,
        description="Stock universe ticker symbols",
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Portfolio weights (sum to 1.0). None = equal weight.",
    )

    @field_validator("weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is not None and abs(sum(v) - 1.0) > 1e-6:
            msg = f"Weights must sum to 1.0, got {sum(v):.6f}"
            raise ValueError(msg)
        return v

    @field_validator("tickers")
    @classmethod
    def tickers_must_be_uppercase(cls, v: List[str]) -> List[str]:
        return [t.upper() for t in v]


# =============================================================================
# Top-level application settings
# =============================================================================


class AppSettings(BaseSettings):
    """
    Complete application configuration with validation.

    Loads from:
      1. Environment variables (prefixed with ``PRA_``)
      2. ``config.yaml`` in the project root (via :func:`load_settings`)
      3. ``.env`` file (optional)
    """

    model_config = SettingsConfigDict(
        env_prefix="PRA_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=False,
        validate_default=True,
    )

    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    risk: RiskMetricsConfig = Field(default_factory=RiskMetricsConfig)
    ml: MLParamsConfig = Field(default_factory=MLParamsConfig)
    app_env: str = Field("development", description="Runtime environment")
    log_level: str = Field("INFO", description="Logging level")
    debug: bool = Field(False, description="Enable debug mode")
    api_host: str = Field("0.0.0.0", description="API server host")
    api_port: PositiveInt = Field(8000, description="API server port")
    api_workers: PositiveInt = Field(4, description="Number of API worker processes")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    api_secret_key: str = Field(
        "change-me-in-production-use-a-strong-random-key",
        description="JWT signing secret key (set via PRA_API_SECRET_KEY env var)",
    )
    api_auth_enabled: bool = Field(
        False,
        description="Enable JWT authentication on API endpoints",
    )
    data_dir: Path = Field(Path("data"), description="Data directory")
    reports_dir: Path = Field(Path("reports"), description="Reports output directory")
    models_dir: Path = Field(Path("models"), description="Trained model artifacts directory")

    def model_post_init(self, __context) -> None:
        for d in (self.data_dir, self.reports_dir, self.models_dir):
            d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Factory
# =============================================================================


def load_settings(
    config_path: Optional[str] = None,
    env_file: Optional[str] = None,
    **overrides,
) -> AppSettings:
    """
    Load and validate application settings.

    Priority (highest wins):
      1. Explicit ``**overrides`` keyword arguments
      2. Environment variables (``PRA_*``)
      3. YAML file
      4. ``.env`` file
      5. Default values

    Args:
        config_path: Path to a YAML config file. Defaults to ``config.yaml``
            in the project root.
        env_file: Path to a ``.env`` file. Defaults to ``.env`` in the project root.
        **overrides: Any setting overrides as keyword arguments.

    Returns:
        AppSettings: Validated settings object.

    Example:
        >>> settings = load_settings(api_port=8080, log_level="DEBUG")
        >>> settings.portfolio.tickers
        ['AAPL', 'MSFT', ...]
        >>> settings.risk.var_confidence
        0.95
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")

    init_kwargs = {}

    # 1. Load from YAML
    if os.path.exists(config_path):
        with open(config_path) as f:
            yaml_data = yaml_safe_load(f) or {}
        mapping = {
            "stocks": ("portfolio", "tickers"),
            "weights": ("portfolio", "weights"),
            "data": ("data", None),
            "risk_metrics": ("risk", None),
            "ml_params": ("ml", None),
        }
        for yaml_key, (settings_key, sub_key) in mapping.items():
            if yaml_key in yaml_data:
                val = yaml_data[yaml_key]
                if sub_key is not None:
                    nested = init_kwargs.setdefault(settings_key, {})
                    nested[sub_key] = val
                else:
                    init_kwargs[settings_key] = val

    # 2. Load from .env
    if env_file is None:
        env_file = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_file):
        from dotenv import load_dotenv

        load_dotenv(env_file)

    # 3. Apply overrides
    init_kwargs.update(overrides)

    return AppSettings(**init_kwargs)
