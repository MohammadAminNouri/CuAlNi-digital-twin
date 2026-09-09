"""Cross-validation with fold-level audit trails and no training/test leakage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    train_size: int
    test_size: int
    mae: float
    rmse: float
    r2: float | None


@dataclass
class ValidationResult:
    model_name: str
    folds: list[FoldMetrics]
    oof_prediction: np.ndarray
    oof_observed: np.ndarray
    split_strategy: str

    @property
    def summary(self) -> dict[str, float | int | str | None]:
        mae_values = np.array([fold.mae for fold in self.folds], dtype=float)
        rmse_values = np.array([fold.rmse for fold in self.folds], dtype=float)
        r2_values = np.array(
            [fold.r2 for fold in self.folds if fold.r2 is not None], dtype=float
        )
        return {
            "model": self.model_name,
            "n_folds": len(self.folds),
            "MAE": float(np.mean(mae_values)),
            "MAE_fold_sd": float(np.std(mae_values, ddof=1)) if len(mae_values) > 1 else None,
            "RMSE": float(np.sqrt(mean_squared_error(self.oof_observed, self.oof_prediction))),
            "RMSE_fold_mean": float(np.mean(rmse_values)),
            "R2": float(r2_score(self.oof_observed, self.oof_prediction))
            if len(self.oof_observed) > 1
            else None,
            "R2_fold_mean": float(np.mean(r2_values)) if r2_values.size else None,
            "split_strategy": self.split_strategy,
        }

    def fold_table(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(fold) for fold in self.folds])


def _splitter(
    n_rows: int,
    n_splits: int,
    groups: Iterable[object] | None,
    random_state: int,
):
    if groups is not None:
        group_values = np.asarray(list(groups))
        unique = np.unique(group_values)
        if unique.size >= 2:
            folds = min(n_splits, unique.size)
            if folds < 2:
                raise ValueError("Grouped validation needs at least two independent sources")
            return GroupKFold(n_splits=folds), group_values, "GroupKFold(source_id)"
    folds = min(n_splits, n_rows)
    if folds < 2:
        raise ValueError("Cross-validation needs at least two rows")
    return KFold(n_splits=folds, shuffle=True, random_state=random_state), None, "KFold(shuffled)"


def cross_validate_regressor(
    model_name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_splits: int = 5,
    groups: Iterable[object] | None = None,
    random_state: int = 42,
) -> ValidationResult:
    """Evaluate an estimator and preserve each out-of-fold prediction."""

    y_values = np.asarray(y, dtype=float)
    if len(X) != len(y_values):
        raise ValueError("X and y lengths differ")
    splitter, group_values, strategy = _splitter(len(X), n_splits, groups, random_state)
    oof = np.full(len(y_values), np.nan, dtype=float)
    folds: list[FoldMetrics] = []
    split_iter = splitter.split(X, y_values, group_values)
    for fold_number, (train_index, test_index) in enumerate(split_iter, start=1):
        fitted = clone(estimator).fit(X.iloc[train_index], y_values[train_index])
        predicted = np.asarray(fitted.predict(X.iloc[test_index]), dtype=float)
        oof[test_index] = predicted
        observed = y_values[test_index]
        folds.append(
            FoldMetrics(
                fold=fold_number,
                train_size=int(len(train_index)),
                test_size=int(len(test_index)),
                mae=float(mean_absolute_error(observed, predicted)),
                rmse=float(np.sqrt(mean_squared_error(observed, predicted))),
                r2=float(r2_score(observed, predicted)) if len(test_index) >= 2 else None,
            )
        )
    if np.isnan(oof).any():
        raise RuntimeError("Cross-validation did not produce a prediction for every row")
    return ValidationResult(model_name, folds, oof, y_values, strategy)


def comparison_table(results: Iterable[ValidationResult]) -> pd.DataFrame:
    table = pd.DataFrame([result.summary for result in results])
    if table.empty:
        return table
    return table.sort_values(["MAE", "RMSE"], ascending=True).reset_index(drop=True)
