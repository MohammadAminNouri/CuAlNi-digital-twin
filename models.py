"""Reproducible model registry for Cu-Al-Ni property regression."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Sequence

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR

from .preprocessing import make_preprocessor


@dataclass(frozen=True)
class ModelAvailability:
    name: str
    available: bool
    reason: str | None = None


def model_availability() -> list[ModelAvailability]:
    return [
        ModelAvailability("linear", True),
        ModelAvailability("ridge", True),
        ModelAvailability("svr", True),
        ModelAvailability("random_forest", True),
        ModelAvailability("gaussian_process", True),
        ModelAvailability(
            "xgboost",
            importlib.util.find_spec("xgboost") is not None,
            None if importlib.util.find_spec("xgboost") else "xgboost is not installed",
        ),
        ModelAvailability(
            "lightgbm",
            importlib.util.find_spec("lightgbm") is not None,
            None if importlib.util.find_spec("lightgbm") else "lightgbm is not installed",
        ),
    ]


def build_regressor(name: str, *, random_state: int = 42) -> RegressorMixin:
    """Instantiate a documented, deterministic baseline model."""

    if name == "linear":
        return LinearRegression()
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "svr":
        return SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "gaussian_process":
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=1.0,
            length_scale_bounds=(1e-3, 1e3),
            nu=1.5,
        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-8, 1e3))
        return GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=random_state,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("Install xgboost to enable the XGBoost model") from exc
        return XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise ImportError("Install lightgbm to enable the LightGBM model") from exc
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise KeyError(f"Unknown regression model {name!r}")


def build_pipeline(name: str, features: Sequence[str], *, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocessor(features)),
            ("regressor", build_regressor(name, random_state=random_state)),
        ]
    )


def available_model_names() -> list[str]:
    return [item.name for item in model_availability() if item.available]
