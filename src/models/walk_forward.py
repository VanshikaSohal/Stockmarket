"""
Walk-forward cross-validation with purging and embargoing.

Implements the purged/embargoed walk-forward validation methodology
described in López de Prado's "Advances in Financial Machine Learning".

Key concepts:
  - **Expanding window**: training set grows as we move forward in time.
  - **Purging**: remove from the training set any observations whose
    labels overlap with the test set in time.
  - **Embargoing**: after each test set, skip a buffer period before
    the next training set to prevent data leakage.
"""

from __future__ import annotations

import logging
from typing import Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


# =============================================================================
# Walk-forward split generator
# =============================================================================


class PurgedWalkForward(BaseCrossValidator):
    """
    Purged walk-forward cross-validation splitter.

    For each split:
      - Training set: ``[0, test_start - embargo)``
      - Test set: ``[test_start, test_end)``

    Parameters
    ----------
    n_test : int
        Number of samples in each test fold.
    n_start : int
        Minimum number of training samples before the first test.
    purge : int
        Number of samples to purge from the end of each training set
        to avoid label leakage into the test set.
    embargo : int
        Number of samples to skip after each test set before the next
        training set begins.
    step : int
        Step between successive test folds. Default equals ``n_test``.
    """

    def __init__(
        self,
        n_test: int = 252,
        n_start: int = 756,
        purge: int = 0,
        embargo: int = 5,
        step: Optional[int] = None,
    ):
        self.n_test = n_test
        self.n_start = n_start
        self.purge = purge
        self.embargo = embargo
        self.step = step or n_test

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        if X is not None:
            n = len(X)
        elif y is not None:
            n = len(y)
        else:
            return 0
        return max(0, (n - self.n_start) // self.step)

    def split(
        self, X, y=None, groups=None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n = len(X)
        test_start = self.n_start

        while test_start + self.n_test <= n:
            test_end = test_start + self.n_test
            train_end = test_start - self.purge
            train_start = 0

            # Apply embargo from previous test set
            if test_start > self.n_start:
                train_start = max(train_start, test_start - self.step + self.embargo)

            train_idx = np.arange(train_start, max(train_end, train_start))
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

            test_start += self.step


# =============================================================================
# Evaluation
# =============================================================================


def walk_forward_evaluate(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    cv: PurgedWalkForward,
    task: str = "classification",
    return_predictions: bool = False,
) -> Dict:
    """
    Evaluate a model using purged walk-forward cross-validation.

    Args:
        model_fn: A callable ``model_fn(X_train, y_train) -> fitted_model``.
        X: Feature matrix.
        y: Target vector.
        cv: ``PurgedWalkForward`` splitter instance.
        task: ``"classification"`` or ``"regression"``.
        return_predictions: If True, return out-of-sample predictions.

    Returns:
        Dict with keys:
          - ``metrics``: aggregated performance metrics.
          - ``per_fold_metrics``: list of per-fold metric dicts.
          - ``predictions``: (optional) array of OOS predictions.
    """
    from sklearn.preprocessing import StandardScaler

    oos_preds = np.full(len(y), np.nan)
    per_fold = []
    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_fn(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        if task == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_test, y_pred)),
            }
        else:
            metrics = {
                "mae": float(mean_absolute_error(y_test, y_pred)),
                "r2": float(r2_score(y_test, y_pred)),
            }

        metrics["fold"] = fold
        metrics["train_size"] = len(train_idx)
        metrics["test_size"] = len(test_idx)
        per_fold.append(metrics)
        oos_preds[test_idx] = y_pred

        logger.debug("Fold %d/%d: %s", fold + 1, cv.get_n_splits(X), metrics)

    # Aggregate
    oos_valid = oos_preds[~np.isnan(oos_preds)]
    y_valid = y[~np.isnan(oos_preds)]

    if task == "classification":
        aggregated = {
            "accuracy": float(accuracy_score(y_valid, oos_valid)),
            "accuracy_std": float(np.std([f["accuracy"] for f in per_fold])),
        }
    else:
        aggregated = {
            "mae": float(mean_absolute_error(y_valid, oos_valid)),
            "r2": float(r2_score(y_valid, oos_valid)),
        }

    result = {"metrics": aggregated, "per_fold_metrics": per_fold}
    if return_predictions:
        result["predictions"] = oos_preds

    logger.info("Walk-forward %s: %s", task, aggregated)
    return result


# =============================================================================
# Expanding window (no purging, simpler variant)
# =============================================================================


def expanding_window_split(
    n_samples: int,
    initial_train: int,
    test_size: int = 1,
    step: int = 1,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Simple expanding window splitter (no purge/embargo).

    Yields (train_idx, test_idx) pairs where the training set grows
    with each split.
    """
    for test_end in range(initial_train + test_size, n_samples + 1, step):
        test_start = test_end - test_size
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        yield train_idx, test_idx
