"""Models package: classical ML, ensemble, deep learning, and interpretability."""

# ── Existing (unchanged) ──────────────────────────────────────────────────
from src.models.supervised_ml import (
    build_features_and_target,
    get_feature_importances,
    train_catboost,
    train_lightgbm,
    train_random_forest,
    train_random_forest_with_mlflow,
    train_xgboost,
    train_xgboost_with_mlflow,
)
from src.models.time_series import (
    build_lstm_model,
    check_stationarity,
    create_sequences,
    fit_arima,
    forecast_arima,
    train_ridge_sequence,
)

# ── Sprint 1A: MLflow + Optuna + Walk-forward CV ─────────────────────────
from src.models.mlflow_tracking import (
    auto_log,
    configure_mlflow,
    log_dataframe,
    log_figure,
    log_metrics_from_dict,
    log_params_from_dict,
    log_sklearn_model,
    mlflow_run,
)
from src.models.optuna_tuning import (
    get_best_model,
    optimize_hyperparameters,
)
from src.models.walk_forward import (
    PurgedWalkForward,
    expanding_window_split,
    walk_forward_evaluate,
)

# ── Sprint 1B: Ensemble + Interpretability ────────────────────────────────
from src.models.ensemble import (
    build_stacking_ensemble,
    train_stacking_ensemble,
)
from src.models.interpret import (
    create_explainer,
    explain_model,
    plot_shap_dependence,
    plot_shap_force,
    plot_shap_summary,
    plot_shap_waterfall,
)

# ── Sprint 1C: Deep Learning ──────────────────────────────────────────────
from src.models.deep_learning import (
    ProbabilisticSequenceRegressor,
    SequenceRegressor,
    train_nbeats,
    train_sequence_rnn,
    train_temporal_fusion_transformer,
)

# ── Phase 3: Bayesian Models ─────────────────────────────────────────────
from src.models.bayesian_hierarchical import (
    hierarchical_regression,
    hierarchical_pooling_factor,
    horseshoe_regression,
    loo_cv_comparison,
)

__all__ = [
    # Existing
    "build_features_and_target",
    "train_random_forest",
    "train_random_forest_with_mlflow",
    "train_xgboost",
    "train_xgboost_with_mlflow",
    "train_lightgbm",
    "train_catboost",
    "get_feature_importances",
    "check_stationarity",
    "fit_arima",
    "forecast_arima",
    "create_sequences",
    "build_lstm_model",
    "train_ridge_sequence",
    # Sprint 1A
    "configure_mlflow",
    "mlflow_run",
    "auto_log",
    "log_sklearn_model",
    "log_metrics_from_dict",
    "log_params_from_dict",
    "log_dataframe",
    "log_figure",
    "optimize_hyperparameters",
    "get_best_model",
    "PurgedWalkForward",
    "walk_forward_evaluate",
    "expanding_window_split",
    # Sprint 1B
    "build_stacking_ensemble",
    "train_stacking_ensemble",
    "create_explainer",
    "explain_model",
    "plot_shap_summary",
    "plot_shap_dependence",
    "plot_shap_force",
    "plot_shap_waterfall",
    # Sprint 1C
    "SequenceRegressor",
    "ProbabilisticSequenceRegressor",
    "train_sequence_rnn",
    "train_temporal_fusion_transformer",
    "train_nbeats",
    # Phase 3
    "hierarchical_regression",
    "horseshoe_regression",
    "loo_cv_comparison",
    "hierarchical_pooling_factor",
]
