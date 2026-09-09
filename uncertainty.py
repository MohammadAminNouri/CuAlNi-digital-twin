"""Distribution-free and resampling uncertainty utilities.

Intervals are reported with their method and empirical calibration diagnostics;
they are never relabelled as physical measurement uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone


@dataclass(frozen=True)
class IntervalCalibration:
    method: str
    alpha: float
    nominal_coverage: float
    residual_quantile: float
    n_calibration: int
    empirical_oof_coverage: float | None
    limitations: tuple[str, ...]


def conformal_residual_quantile(
    observed: np.ndarray,
    predicted: np.ndarray,
    *,
    alpha: float = 0.10,
) -> IntervalCalibration:
    """Finite-sample corrected symmetric split/CV-conformal residual radius."""

    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    finite = np.isfinite(observed) & np.isfinite(predicted)
    residuals = np.abs(observed[finite] - predicted[finite])
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    if residuals.size < 3:
        raise ValueError("At least three calibration residuals are required")
    quantile_level = min(1.0, np.ceil((residuals.size + 1) * (1.0 - alpha)) / residuals.size)
    try:
        radius = float(np.quantile(residuals, quantile_level, method="higher"))
    except TypeError:  # NumPy < 1.22
        radius = float(np.quantile(residuals, quantile_level, interpolation="higher"))
    coverage = float(np.mean(residuals <= radius))
    return IntervalCalibration(
        method="symmetric CV-conformal absolute-residual interval",
        alpha=float(alpha),
        nominal_coverage=float(1.0 - alpha),
        residual_quantile=radius,
        n_calibration=int(residuals.size),
        empirical_oof_coverage=coverage,
        limitations=(
            "Coverage requires future observations to be exchangeable with the training data.",
            "A symmetric global interval does not model composition-dependent noise.",
            "Cross-validation reuse makes this an approximate CV-conformal interval.",
        ),
    )


@dataclass(frozen=True)
class BootstrapPrediction:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    standard_deviation: np.ndarray
    n_bootstrap: int
    alpha: float
    method: str = "nonparametric pairs bootstrap"


def bootstrap_prediction_interval(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    X_new: pd.DataFrame,
    *,
    n_bootstrap: int = 300,
    alpha: float = 0.10,
    random_state: int = 42,
) -> BootstrapPrediction:
    """Estimate model-fit variability by resampling training pairs.

    This interval excludes irreducible measurement noise and is intentionally kept
    separate from conformal predictive intervals.
    """

    if n_bootstrap < 30:
        raise ValueError("At least 30 bootstrap replicates are required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    y_values = np.asarray(y, dtype=float)
    rng = np.random.default_rng(random_state)
    predictions: list[np.ndarray] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(X), size=len(X))
        fitted = clone(estimator).fit(X.iloc[indices], y_values[indices])
        predictions.append(np.asarray(fitted.predict(X_new), dtype=float))
    matrix = np.vstack(predictions)
    return BootstrapPrediction(
        mean=np.mean(matrix, axis=0),
        lower=np.quantile(matrix, alpha / 2.0, axis=0),
        upper=np.quantile(matrix, 1.0 - alpha / 2.0, axis=0),
        standard_deviation=np.std(matrix, axis=0, ddof=1),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
    )
