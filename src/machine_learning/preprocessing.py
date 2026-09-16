"""Leakage-resistant preprocessing for heterogeneous Cu-Al-Ni datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .dataset import COMPOSITION_COLUMNS, DatasetValidationError

CATEGORICAL_FEATURES = ("quench_medium",)


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def validate_feature_frame(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Validate model inputs without silently repairing physical quantities."""

    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    missing = [column for column in features if column not in frame]
    if missing:
        raise DatasetValidationError("Prediction input is missing: " + ", ".join(missing))
    out = frame[list(features)].copy()
    composition = [name for name in COMPOSITION_COLUMNS if name in features]
    if composition:
        if len(composition) != 3:
            raise DatasetValidationError("All three Cu, Al, and Ni wt% features must be supplied")
        numeric = out[composition].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any(axis=None):
            raise DatasetValidationError("Composition features must be numeric and non-missing")
        if ((numeric < 0.0) | (numeric > 100.0)).any(axis=None):
            raise DatasetValidationError("Composition features must lie in [0, 100] wt%")
        totals = numeric.sum(axis=1)
        if not np.allclose(totals, 100.0, atol=0.5, rtol=0.0):
            raise DatasetValidationError("Prediction compositions must total 100 +/- 0.5 wt%")
    return out


def split_feature_types(features: Sequence[str]) -> tuple[list[str], list[str]]:
    categorical = [column for column in features if column in CATEGORICAL_FEATURES]
    numeric = [column for column in features if column not in categorical]
    return numeric, categorical


def make_preprocessor(features: Sequence[str]) -> ColumnTransformer:
    """Create fold-local preprocessing suitable for use inside CV pipelines."""

    numeric, categorical = split_feature_types(features)
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", RobustScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )
    if not transformers:
        raise DatasetValidationError("No features were supplied to the preprocessor")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


@dataclass(frozen=True)
class FeatureDomain:
    """Training range audit used to flag, never conceal, extrapolation."""

    numeric_bounds: dict[str, tuple[float, float]]
    categorical_levels: dict[str, tuple[str, ...]]

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, features: Sequence[str]) -> "FeatureDomain":
        numeric, categorical = split_feature_types(features)
        bounds: dict[str, tuple[float, float]] = {}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                bounds[column] = (float(values.min()), float(values.max()))
        levels = {
            column: tuple(sorted(frame[column].dropna().astype(str).unique().tolist()))
            for column in categorical
        }
        return cls(bounds, levels)

    def assess(self, frame: pd.DataFrame) -> list[dict[str, object]]:
        findings: list[dict[str, object]] = []
        for row_index, row in frame.iterrows():
            outside: list[str] = []
            unknown: list[str] = []
            for column, (lower, upper) in self.numeric_bounds.items():
                value = row.get(column)
                if pd.notna(value) and not lower <= float(value) <= upper:
                    outside.append(f"{column}={value:g} outside [{lower:g}, {upper:g}]")
            for column, levels in self.categorical_levels.items():
                value = row.get(column)
                if pd.notna(value) and str(value) not in levels:
                    unknown.append(f"{column}={value!s} unseen in training")
            findings.append(
                {
                    "row": row_index,
                    "within_training_ranges": not outside and not unknown,
                    "out_of_range": outside,
                    "unseen_categories": unknown,
                }
            )
        return findings
