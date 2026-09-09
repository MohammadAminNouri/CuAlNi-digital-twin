"""Explainability with explicit SHAP/permutation distinction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class ImportanceResult:
    method: str
    feature_names: list[str]
    importance: np.ndarray
    values: np.ndarray | None
    classification: str = "machine_learning_explanation"
    limitation: str = (
        "Feature importance explains this fitted empirical model; it does not establish causality."
    )

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_importance": self.importance}
        ).sort_values("mean_abs_importance", ascending=False, ignore_index=True)


def _transformed_data(pipeline, frame: pd.DataFrame) -> tuple[np.ndarray, list[str], Any]:
    preprocess = pipeline.named_steps["preprocess"]
    regressor = pipeline.named_steps["regressor"]
    values = np.asarray(preprocess.transform(frame), dtype=float)
    try:
        names = preprocess.get_feature_names_out().astype(str).tolist()
    except AttributeError:
        names = [f"feature_{index}" for index in range(values.shape[1])]
    return values, names, regressor


def shap_importance(
    fitted_pipeline,
    background: pd.DataFrame,
    explain: pd.DataFrame | None = None,
    *,
    max_background: int = 100,
) -> ImportanceResult:
    """Calculate SHAP values, failing explicitly if SHAP is not installed."""

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP explanation unavailable because the optional 'shap' package is not installed"
        ) from exc
    background = background.iloc[:max_background]
    explain = background if explain is None else explain
    background_values, names, regressor = _transformed_data(fitted_pipeline, background)
    explain_values, _, _ = _transformed_data(fitted_pipeline, explain)
    explainer = shap.Explainer(regressor, background_values, feature_names=names)
    explanation = explainer(explain_values)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[..., 0]
    return ImportanceResult("SHAP mean absolute value", names, np.mean(np.abs(values), axis=0), values)


def permutation_importance_report(
    fitted_pipeline,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 30,
    random_state: int = 42,
) -> ImportanceResult:
    """Model-agnostic fallback, intentionally not described as SHAP."""

    result = permutation_importance(
        fitted_pipeline,
        X,
        y,
        scoring="neg_mean_absolute_error",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return ImportanceResult(
        "permutation importance (MAE decrease; not SHAP)",
        X.columns.astype(str).tolist(),
        np.asarray(result.importances_mean, dtype=float),
        np.asarray(result.importances, dtype=float).T,
    )
