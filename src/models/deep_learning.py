"""
Deep learning models for time series forecasting.

Provides PyTorch-based implementations of:
  - LSTM / GRU recurrent networks
  - Temporal Fusion Transformer (via pytorch-forecasting)
  - N-BEATS (via pytorch-forecasting)

All models follow a consistent train/eval API compatible with the
existing pipeline (fit / predict / evaluate).
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

logger = logging.getLogger(__name__)

# =============================================================================
# PyTorch availability
# =============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None  # type: ignore
    torch = None  # type: ignore

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    PL_AVAILABLE = True
except ImportError:
    PL_AVAILABLE = False
    pl = None  # type: ignore

try:
    from pytorch_forecasting import NBeats, TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss

    PTF_AVAILABLE = True
except ImportError:
    PTF_AVAILABLE = False


# =============================================================================
# Shared utilities
# =============================================================================


def _validate_torch():
    """Raise if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for deep learning models. "
            "Install with: pip install torch"
        )


def _to_tensor(
    data: np.ndarray, dtype=torch.float32, device: Optional[str] = None
) -> torch.Tensor:
    """Convert a numpy array to a torch tensor."""
    _validate_torch()
    tensor = torch.tensor(data, dtype=dtype)
    if device:
        tensor = tensor.to(device)
    return tensor


def _inverse_scale(predictions: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform predictions using a fitted sklearn scaler."""
    if scaler is None:
        return predictions
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()


# =============================================================================
# LSTM / GRU model
# =============================================================================


class SequenceRegressor(nn.Module):
    """
    PyTorch LSTM or GRU for univariate time series regression.

    Architecture:
        [LinearEncoder] -> [LSTM / GRU] -> [Dropout] -> [LinearDecoder]

    Args:
        input_size: Number of input features (1 for univariate).
        hidden_size: Number of hidden units per layer.
        num_layers: Number of recurrent layers.
        dropout: Dropout rate between layers.
        bidirectional: If True, use bidirectional RNN.
        cell_type: ``"lstm"`` or ``"gru"``.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        cell_type: str = "lstm",
    ):
        _validate_torch()
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size * self.num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden * num_dir)
        last_out = rnn_out[:, -1, :]  # (batch, hidden * num_dir)
        last_out = self.dropout(last_out)
        return self.regressor(last_out)  # (batch, 1)


class ProbabilisticSequenceRegressor(nn.Module):
    """
    LSTM/GRU with a probabilistic output head (Gaussian or Quantile).

    Architecture:
        [LSTM/GRU] -> [Dropout] -> [Gaussian/Quantile head]

    Gaussian head outputs (mu, sigma).
    Quantile head outputs predictions at specified quantiles.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: str = "lstm",
        probabilistic_head: str = "gaussian",
        quantiles: Optional[List[float]] = None,
    ):
        _validate_torch()
        super().__init__()

        self.probabilistic_head = probabilistic_head
        self.quantiles = quantiles or [0.05, 0.5, 0.95]
        self.num_quantiles = len(self.quantiles)

        rnn_cls = nn.LSTM if cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)

        if probabilistic_head == "gaussian":
            self.head = nn.Linear(hidden_size, 2)  # mu, log_sigma
        elif probabilistic_head == "quantile":
            self.head = nn.Linear(hidden_size, self.num_quantiles)
        else:
            raise ValueError(f"Unknown head: {probabilistic_head}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        last_out = rnn_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.head(last_out)

    def predict_with_uncertainty(
        self, x: torch.Tensor
    ) -> Dict[str, np.ndarray]:
        """Return point prediction and uncertainty intervals."""
        with torch.no_grad():
            output = self.forward(x).cpu().numpy()

        if self.probabilistic_head == "gaussian":
            mu = output[:, 0]
            log_sigma = output[:, 1]
            sigma = np.exp(log_sigma)
            return {
                "prediction": mu,
                "std": sigma,
                "lower_95": mu - 1.96 * sigma,
                "upper_95": mu + 1.96 * sigma,
            }
        else:
            return {
                "prediction": output[:, self.quantiles.index(0.5)],
                "quantiles": {
                    str(q): output[:, i]
                    for i, q in enumerate(self.quantiles)
                },
            }


# =============================================================================
# Training function
# =============================================================================


def train_sequence_rnn(
    series: np.ndarray,
    seq_length: int = 60,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    cell_type: str = "lstm",
    probabilistic_head: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train an LSTM or GRU for sequence price/return forecasting.

    Args:
        series: 1-D numpy array of values.
        seq_length: Lookback window.
        hidden_size: RNN hidden size.
        num_layers: Number of RNN layers.
        dropout: Dropout rate.
        cell_type: ``"lstm"`` or ``"gru"``.
        probabilistic_head: If ``"gaussian"`` or ``"quantile"``, uses a
            probabilistic output head. ``None`` for point predictions.
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Adam learning rate.
        test_size: Fraction held out.
        random_state: Random seed.
        device: ``"cuda"`` or ``"cpu"``. Auto-detects if None.
        verbose: Print training progress.

    Returns:
        Dict with ``model``, ``metrics``, ``y_test``, ``y_pred``,
        ``prediction_intervals`` (if probabilistic), ``train_losses``.
    """
    _validate_torch()

    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import MinMaxScaler

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length : i])
        y.append(scaled[i])
    X = np.array(X).reshape(-1, seq_length, 1)
    y = np.array(y)

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Datasets
    train_dataset = TensorDataset(
        _to_tensor(X_train), _to_tensor(y_train).view(-1, 1)
    )
    test_dataset = TensorDataset(
        _to_tensor(X_test), _to_tensor(y_test).view(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    if probabilistic_head:
        model = ProbabilisticSequenceRegressor(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
            probabilistic_head=probabilistic_head,
        )
    else:
        model = SequenceRegressor(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(Xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * Xb.size(0)
        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)
        if verbose and epoch % 10 == 0:
            logger.info("Epoch %3d/%d — loss: %.6f", epoch, epochs, epoch_loss)

    # Evaluate
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            preds = model(Xb)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    y_pred_scaled = np.concatenate(all_preds, axis=0).flatten()
    y_test_actual = np.concatenate(all_targets, axis=0).flatten()

    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test_actual.reshape(-1, 1)).flatten()

    metrics = {
        "mae": float(mean_absolute_error(y_test_inv, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test_inv, y_pred))),
        "train_loss_min": float(min(train_losses)),
        "train_loss_final": float(train_losses[-1]),
    }

    result = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "y_test": y_test_inv,
        "y_pred": y_pred,
        "train_losses": train_losses,
    }

    if probabilistic_head:
        with torch.no_grad():
            Xb = _to_tensor(X_test).to(device)
            result["prediction_intervals"] = model.predict_with_uncertainty(Xb)
            # Inverse scale intervals if available
            if "std" in result["prediction_intervals"]:
                for k in ["prediction", "lower_95", "upper_95"]:
                    result["prediction_intervals"][k] = _inverse_scale(
                        result["prediction_intervals"][k], scaler
                    )

    logger.info("Sequence RNN (%s) metrics: %s", cell_type, metrics)
    return result


# =============================================================================
# TFT & N-BEATS wrappers (via pytorch-forecasting)
# =============================================================================


def _ptf_available():
    if not PTF_AVAILABLE:
        raise ImportError(
            "pytorch-forecasting and pytorch-lightning are required. "
            "Install with: pip install pytorch-forecasting pytorch-lightning"
        )


def train_temporal_fusion_transformer(
    data: pd.DataFrame,
    time_idx_col: str = "time_idx",
    group_id_col: str = "ticker",
    target_col: str = "value",
    seq_length: int = 60,
    hidden_size: int = 64,
    attention_head_size: int = 4,
    dropout: float = 0.1,
    max_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train a Temporal Fusion Transformer for multi-horizon forecasting.

    Requires ``pytorch-forecasting`` and ``pytorch-lightning``.

    Args:
        data: DataFrame with columns ``[time_idx_col, group_id_col, target_col]``
            plus any known/relevant exogenous features.
        time_idx_col: Column name for time index (integer, 0..N).
        group_id_col: Column name for time series group (e.g. ticker).
        target_col: Column name for target variable.
        seq_length: Encoder/decoder sequence length.
        hidden_size: Size of LSTM layers.
        attention_head_size: Number of attention heads.
        dropout: Dropout rate.
        max_epochs: Training epochs.
        batch_size: Batch size.
        learning_rate: Adam learning rate.
        test_size: Fraction held out.
        random_state: Random seed.

    Returns:
        Dict with ``model``, ``metrics``, ``predictions``, ``trainer``.
    """
    _ptf_available()

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import TensorBoardLogger

    pl.seed_everything(random_state)

    # Build TimeSeriesDataSet
    max_encoder_length = seq_length
    max_prediction_length = 1  # single-step for now

    training_cutoff = int(len(data) * (1 - test_size))

    training = TimeSeriesDataSet(
        data[:training_cutoff],
        time_idx=time_idx_col,
        target=target_col,
        group_ids=[group_id_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=[group_id_col]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data[training_cutoff:])

    train_loader = DataLoader(training, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    # Model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        hidden_size=hidden_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=hidden_size // 2,
        loss=QuantileLoss([0.025, 0.25, 0.5, 0.75, 0.975]),
        learning_rate=learning_rate,
        reduce_on_plateau_patience=4,
        **kwargs,
    )

    # Trainer
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, mode="min"
    )
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(tft, train_loader, val_loader)

    # Evaluate
    best_model = TemporalFusionTransformer.load_from_checkpoint(
        checkpoint.best_model_path
    )
    predictions = best_model.predict(val_loader)
    raw_predictions = best_model.predict(val_loader, mode="raw", return_index=True)

    metrics = {
        "val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
    }

    return {
        "model": best_model,
        "trainer": trainer,
        "metrics": metrics,
        "predictions": predictions.numpy().flatten() if predictions is not None else None,
        "raw_predictions": raw_predictions,
    }


def train_nbeats(
    data: pd.DataFrame,
    time_idx_col: str = "time_idx",
    group_id_col: str = "ticker",
    target_col: str = "value",
    seq_length: int = 60,
    n_blocks: int = 3,
    n_layers: int = 4,
    n_hidden: int = 64,
    max_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """
    Train an N-BEATS model for interpretable time series forecasting.

    Args:
        Similar to ``train_temporal_fusion_transformer`` but with N-BEATS-specific
        architecture parameters (``n_blocks``, ``n_layers``, ``n_hidden``).

    Returns:
        Dict with ``model``, ``metrics``, ``predictions``, ``trainer``.
    """
    _ptf_available()

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    pl.seed_everything(random_state)

    max_encoder_length = seq_length
    max_prediction_length = 1
    training_cutoff = int(len(data) * (1 - test_size))

    training = TimeSeriesDataSet(
        data[:training_cutoff],
        time_idx=time_idx_col,
        target=target_col,
        group_ids=[group_id_col],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[target_col],
        target_normalizer=GroupNormalizer(groups=[group_id_col]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data[training_cutoff:])
    train_loader = DataLoader(training, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validation, batch_size=batch_size, shuffle=False)

    nbeats = NBeats.from_dataset(
        training,
        n_blocks=n_blocks,
        n_layers=n_layers,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        loss=MAE(),
        **kwargs,
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=False,
        logger=False,
    )

    trainer.fit(nbeats, train_loader, val_loader)

    best_model = NBeats.load_from_checkpoint(checkpoint.best_model_path)
    predictions = best_model.predict(val_loader)

    return {
        "model": best_model,
        "trainer": trainer,
        "metrics": {"val_loss": float(trainer.callback_metrics.get("val_loss", 0))},
        "predictions": predictions.numpy().flatten() if predictions is not None else None,
    }
