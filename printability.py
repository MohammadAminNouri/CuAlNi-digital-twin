"""User-auditable LPBF suitability scoring with no hidden scientific cutoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class PrintabilityError(ValueError):
    pass


@dataclass(frozen=True)
class ScreeningMetric:
    value: float
    unit: str
    classification: str
    provenance: str
    uncertainty: float | None = None


@dataclass(frozen=True)
class MetricRule:
    name: str
    preferred_value: float
    adverse_value: float
    weight: float
    unit: str
    rationale: str
    source: str

    def validate(self) -> None:
        if not np.isfinite(self.preferred_value) or not np.isfinite(self.adverse_value):
            raise PrintabilityError(f"Rule {self.name!r} bounds must be finite")
        if self.preferred_value == self.adverse_value:
            raise PrintabilityError(f"Rule {self.name!r} bounds cannot be equal")
        if self.weight <= 0:
            raise PrintabilityError(f"Rule {self.name!r} weight must be positive")
        if not self.source.strip() or not self.rationale.strip():
            raise PrintabilityError(f"Rule {self.name!r} requires source and rationale")


@dataclass(frozen=True)
class ScreeningProfile:
    name: str
    version: str
    rules: tuple[MetricRule, ...]
    provenance: str
    applicability: str
    grade_bands: tuple[tuple[float, str], ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ScreeningProfile":
        required = ("name", "version", "rules", "provenance", "applicability")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise PrintabilityError("Screening profile missing: " + ", ".join(missing))
        rules = tuple(MetricRule(**item) for item in raw["rules"])
        profile = cls(
            name=str(raw["name"]),
            version=str(raw["version"]),
            rules=rules,
            provenance=str(raw["provenance"]),
            applicability=str(raw["applicability"]),
            grade_bands=tuple((float(score), str(label)) for score, label in raw.get("grade_bands", ())),
        )
        profile.validate()
        return profile

    def validate(self) -> None:
        if not self.rules:
            raise PrintabilityError("Screening profile must contain at least one rule")
        if not self.provenance.strip() or not self.applicability.strip():
            raise PrintabilityError("Screening profile requires provenance and applicability")
        names = [rule.name for rule in self.rules]
        if len(names) != len(set(names)):
            raise PrintabilityError("Screening rule names must be unique")
        for rule in self.rules:
            rule.validate()


@dataclass
class PrintabilityAssessment:
    score_0_to_10: float | None
    grade: str | None
    components: dict[str, dict[str, Any]]
    status: str
    coverage: float
    missing_metrics: list[str]
    profile: dict[str, Any] | None
    classification: str = "engineering_screening"
    confidence: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "This is a screening index, not a calibrated probability of build success.",
        "A score cannot replace coupon builds, process-window development, or defect characterization.",
        "Results are conditional on the declared normalization profile and input provenance.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "status": self.status,
            "score_0_to_10": self.score_0_to_10,
            "grade": self.grade,
            "components": self.components,
            "coverage": self.coverage,
            "missing_metrics": self.missing_metrics,
            "profile": self.profile,
            "confidence": self.confidence,
            "limitations": list(self.limitations),
        }


def _component_score(value: float, rule: MetricRule) -> float:
    fraction = (value - rule.adverse_value) / (rule.preferred_value - rule.adverse_value)
    return float(np.clip(fraction, 0.0, 1.0) * 10.0)


def _grade(score: float, bands: Sequence[tuple[float, str]]) -> str | None:
    if not bands:
        return None
    for lower_bound, label in sorted(bands, reverse=True):
        if score >= lower_bound:
            return label
    return sorted(bands)[0][1]


def assess_printability(
    metrics: Mapping[str, ScreeningMetric | float],
    profile: ScreeningProfile | Mapping[str, Any] | None = None,
    *,
    minimum_weight_coverage: float = 1.0,
) -> PrintabilityAssessment:
    """Score metrics only when explicit, sourced normalization rules are supplied."""

    if not 0 < minimum_weight_coverage <= 1:
        raise PrintabilityError("minimum_weight_coverage must lie in (0, 1]")
    if profile is None:
        return PrintabilityAssessment(
            score_0_to_10=None,
            grade=None,
            components={
                name: {
                    "raw_value": metric.value if isinstance(metric, ScreeningMetric) else float(metric),
                    "normalized_score_0_to_10": None,
                }
                for name, metric in metrics.items()
            },
            status="unavailable_without_calibrated_screening_profile",
            coverage=0.0,
            missing_metrics=[],
            profile=None,
            confidence={
                "score_available": False,
                "reason": "No provenance-declared normalization bounds and weights were supplied",
            },
        )
    if isinstance(profile, Mapping):
        profile = ScreeningProfile.from_mapping(profile)
    profile.validate()
    total_weight = sum(rule.weight for rule in profile.rules)
    available_weight = 0.0
    weighted_score = 0.0
    components: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for rule in profile.rules:
        if rule.name not in metrics:
            missing.append(rule.name)
            continue
        raw = metrics[rule.name]
        metric = raw if isinstance(raw, ScreeningMetric) else ScreeningMetric(
            value=float(raw),
            unit=rule.unit,
            classification="unspecified_input",
            provenance="not supplied",
        )
        if metric.unit != rule.unit:
            raise PrintabilityError(
                f"Metric {rule.name!r} uses {metric.unit!r}; profile expects {rule.unit!r}"
            )
        if not np.isfinite(metric.value):
            raise PrintabilityError(f"Metric {rule.name!r} must be finite")
        score = _component_score(metric.value, rule)
        weighted_score += rule.weight * score
        available_weight += rule.weight
        components[rule.name] = {
            "raw_value": metric.value,
            "unit": metric.unit,
            "normalized_score_0_to_10": score,
            "weight": rule.weight,
            "preferred_value": rule.preferred_value,
            "adverse_value": rule.adverse_value,
            "classification": metric.classification,
            "input_provenance": metric.provenance,
            "input_uncertainty": metric.uncertainty,
            "rule_rationale": rule.rationale,
            "rule_source": rule.source,
        }
    coverage = available_weight / total_weight
    score = weighted_score / available_weight if available_weight and coverage >= minimum_weight_coverage else None
    status = "complete" if score is not None and not missing else "partial"
    if score is None:
        status = "insufficient_metric_coverage"
    profile_record = {
        "name": profile.name,
        "version": profile.version,
        "provenance": profile.provenance,
        "applicability": profile.applicability,
    }
    return PrintabilityAssessment(
        score,
        _grade(score, profile.grade_bands) if score is not None else None,
        components,
        status,
        float(coverage),
        missing,
        profile_record,
        confidence={
            "score_available": score is not None,
            "metric_weight_coverage": float(coverage),
            "profile_declares_grade_bands": bool(profile.grade_bands),
            "interpretation": "coverage/provenance audit; not statistical confidence",
        },
    )
