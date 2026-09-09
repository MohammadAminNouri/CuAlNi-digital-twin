"""Cross-validated ensembles with provenance and applicability diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .dataset import (
    PROPERTY_TARGETS,
    TRANSFORMATION_TARGETS,
    DatasetBundle,
    DatasetValidationError,
    infer_feature_columns,
)
from .models import available_model_names, build_pipeline
from .preprocessing import FeatureDomain, validate_feature_frame
from .uncertainty import IntervalCalibration, conformal_residual_quantile
from .validation import ValidationResult, cross_validate_regressor


@dataclass
class PredictionBatch:
    target: str
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    unit: str
    interval: IntervalCalibration
    applicability: list[dict[str, object]]
    model_weights: dict[str, float]
    classification: str = "machine_learning_prediction"
    limitations: tuple[str, ...] = (
        "Predictions are empirical and are not CALPHAD or first-principles results.",
        "Validity is limited to processing routes and compositions represented in training.",
    )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                self.target: self.values,
                "lower": self.lower,
                "upper": self.upper,
                "nominal_coverage": self.interval.nominal_coverage,
                "within_training_ranges": [
                    item["within_training_ranges"] for item in self.applicability
                ],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "target": self.target,
            "unit": self.unit,
            "prediction": self.values.tolist(),
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "confidence": {
                "type": self.interval.method,
                "nominal_coverage": self.interval.nominal_coverage,
                "empirical_oof_coverage": self.interval.empirical_oof_coverage,
                "calibration_rows": self.interval.n_calibration,
            },
            "applicability": self.applicability,
            "model_weights": self.model_weights,
            "limitations": list(self.limitations + self.interval.limitations),
        }


@dataclass
class TransformationEnsemble:
    """One-target ensemble trained only after dataset provenance authorization."""

    target: str
    alpha: float = 0.10
    random_state: int = 42
    model_names: Sequence[str] | None = None
    n_splits: int = 5
    estimators_: dict[str, Any] = field(default_factory=dict, init=False)
    validations_: dict[str, ValidationResult] = field(default_factory=dict, init=False)
    failures_: dict[str, str] = field(default_factory=dict, init=False)
    weights_: dict[str, float] = field(default_factory=dict, init=False)
    interval_: IntervalCalibration | None = field(default=None, init=False)
    features_: list[str] = field(default_factory=list, init=False)
    domain_: FeatureDomain | None = field(default=None, init=False)
    dataset_sha256_: str | None = field(default=None, init=False)
    provenance_: dict[str, Any] | None = field(default=None, init=False)
    target_unit_: str | None = field(default=None, init=False)

    def fit(
        self,
        bundle: DatasetBundle,
        features: Sequence[str] | None = None,
        *,
        allow_synthetic: bool = False,
    ) -> "TransformationEnsemble":
        selected = list(features or infer_feature_columns(bundle.frame))
        X, y = bundle.training_frame(
            self.target,
            selected,
            allow_synthetic=allow_synthetic,
            minimum_rows=max(8, self.n_splits),
        )
        groups = None
        if "source_id" in bundle.frame:
            groups = bundle.frame.loc[X.index, "source_id"].to_numpy()
        candidates = list(self.model_names or available_model_names())
        self.estimators_.clear()
        self.validations_.clear()
        self.failures_.clear()
        for name in candidates:
            try:
                pipeline = build_pipeline(name, selected, random_state=self.random_state)
                validation = cross_validate_regressor(
                    name,
                    pipeline,
                    X.reset_index(drop=True),
                    y.reset_index(drop=True),
                    n_splits=self.n_splits,
                    groups=groups,
                    random_state=self.random_state,
                )
                pipeline.fit(X, y)
                self.estimators_[name] = pipeline
                self.validations_[name] = validation
            except Exception as exc:  # individual optional/ill-conditioned models may fail
                self.failures_[name] = f"{type(exc).__name__}: {exc}"
        if not self.estimators_:
            raise DatasetValidationError(
                "No model could be trained. Failures: "
                + "; ".join(f"{key}: {value}" for key, value in self.failures_.items())
            )
        errors = {
            name: float(result.summary["MAE"])
            for name, result in self.validations_.items()
        }
        finite_positive = [value for value in errors.values() if np.isfinite(value) and value > 0]
        scale = min(finite_positive) if finite_positive else 1.0
        raw = {
            name: 1.0 / max(error, scale * 1e-6, np.finfo(float).eps)
            for name, error in errors.items()
            if np.isfinite(error)
        }
        total = sum(raw.values())
        self.weights_ = {name: value / total for name, value in raw.items()}
        ensemble_oof = np.zeros(len(y), dtype=float)
        for name, weight in self.weights_.items():
            ensemble_oof += weight * self.validations_[name].oof_prediction
        self.interval_ = conformal_residual_quantile(
            y.to_numpy(), ensemble_oof, alpha=self.alpha
        )
        self.features_ = selected
        self.domain_ = FeatureDomain.from_frame(X, selected)
        self.dataset_sha256_ = bundle.sha256
        self.provenance_ = bundle.audit_record()["provenance"]
        self.target_unit_ = str(bundle.provenance.units[self.target])
        return self

    @property
    def is_fitted(self) -> bool:
        return bool(self.estimators_ and self.interval_ and self.domain_)

    def predict(self, frame: pd.DataFrame) -> PredictionBatch:
        if not self.is_fitted:
            raise RuntimeError(
                "Model unavailable: fit with a provenance-authorized experimental dataset first"
            )
        frame = validate_feature_frame(frame, self.features_)
        prediction = np.zeros(len(frame), dtype=float)
        for name, weight in self.weights_.items():
            prediction += weight * np.asarray(self.estimators_[name].predict(frame), dtype=float)
        assert self.interval_ is not None and self.domain_ is not None
        radius = self.interval_.residual_quantile
        return PredictionBatch(
            target=self.target,
            values=prediction,
            lower=prediction - radius,
            upper=prediction + radius,
            unit=self.target_unit_ or "not documented",
            interval=self.interval_,
            applicability=self.domain_.assess(frame),
            model_weights=dict(self.weights_),
        )

    def validation_table(self) -> pd.DataFrame:
        rows = [result.summary for result in self.validations_.values()]
        return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    def model_card(self) -> dict[str, Any]:
        if not self.is_fitted:
            return {
                "status": "unavailable",
                "reason": "No provenance-authorized training dataset has been fitted",
                "classification": "machine_learning_prediction",
            }
        assert self.interval_ is not None
        return {
            "status": "trained",
            "classification": "machine_learning_prediction",
            "target": self.target,
            "features": self.features_,
            "dataset_sha256": self.dataset_sha256_,
            "dataset_provenance": self.provenance_,
            "model_weights": self.weights_,
            "failed_models": self.failures_,
            "validation": [result.summary for result in self.validations_.values()],
            "uncertainty": asdict(self.interval_),
        }


@dataclass
class TransformationModelSet:
    """Fit and predict the available Ms/Mf/As/Af targets independently."""

    alpha: float = 0.10
    random_state: int = 42
    model_names: Sequence[str] | None = None
    models_: dict[str, TransformationEnsemble] = field(default_factory=dict, init=False)
    unavailable_: dict[str, str] = field(default_factory=dict, init=False)

    def fit(
        self,
        bundle: DatasetBundle,
        features: Sequence[str] | None = None,
        *,
        allow_synthetic: bool = False,
    ) -> "TransformationModelSet":
        self.models_.clear()
        self.unavailable_.clear()
        for target in TRANSFORMATION_TARGETS:
            if target not in bundle.available_targets():
                self.unavailable_[target] = "Target absent or contains no observations"
                continue
            model = TransformationEnsemble(
                target,
                alpha=self.alpha,
                random_state=self.random_state,
                model_names=self.model_names,
            )
            try:
                model.fit(bundle, features, allow_synthetic=allow_synthetic)
                self.models_[target] = model
            except Exception as exc:
                self.unavailable_[target] = f"{type(exc).__name__}: {exc}"
        return self

    def predict(self, frame: pd.DataFrame) -> dict[str, PredictionBatch]:
        return {target: model.predict(frame) for target, model in self.models_.items()}

    def status(self) -> dict[str, Any]:
        return {
            "classification": "machine_learning_prediction",
            "trained_targets": sorted(self.models_),
            "unavailable_targets": self.unavailable_,
            "constraint_note": (
                "Targets are fitted independently. The UI must flag predictions that violate "
                "Mf <= Ms or As <= Af; values are not silently reordered."
            ),
        }


@dataclass
class PropertyModelSet:
    """Fit hardness/yield-strength/UTS regressors with the same provenance gate."""

    alpha: float = 0.10
    random_state: int = 42
    model_names: Sequence[str] | None = None
    models_: dict[str, TransformationEnsemble] = field(default_factory=dict, init=False)
    unavailable_: dict[str, str] = field(default_factory=dict, init=False)

    def fit(
        self,
        bundle: DatasetBundle,
        features: Sequence[str] | None = None,
        *,
        allow_synthetic: bool = False,
    ) -> "PropertyModelSet":
        self.models_.clear()
        self.unavailable_.clear()
        for target in PROPERTY_TARGETS:
            if target not in bundle.available_targets():
                self.unavailable_[target] = "Target absent or contains no observations"
                continue
            model = TransformationEnsemble(
                target,
                alpha=self.alpha,
                random_state=self.random_state,
                model_names=self.model_names,
            )
            try:
                model.fit(bundle, features, allow_synthetic=allow_synthetic)
                self.models_[target] = model
            except Exception as exc:
                self.unavailable_[target] = f"{type(exc).__name__}: {exc}"
        return self

    def predict(self, frame: pd.DataFrame) -> dict[str, PredictionBatch]:
        return {target: model.predict(frame) for target, model in self.models_.items()}

    def status(self) -> dict[str, Any]:
        return {
            "classification": "machine_learning_prediction",
            "trained_targets": sorted(self.models_),
            "unavailable_targets": self.unavailable_,
            "units": {
                target: model.target_unit_ for target, model in self.models_.items()
            },
        }
