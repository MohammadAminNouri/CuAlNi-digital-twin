"""Derived solidification metrics from a supplied CALPHAD/experimental path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class SolidificationAnalysisError(ValueError):
    pass


@dataclass
class SolidificationAssessment:
    liquidus_proxy_C: float
    solidus_proxy_C: float
    freezing_range_C: float
    liquidus_fraction_definition: float
    solidus_fraction_definition: float
    segregation: dict[str, dict[str, float | None]]
    thermal_metrics: dict[str, float | None]
    source: dict[str, Any]
    path: pd.DataFrame
    classification: str = "physics_calculated_from_supplied_path"
    confidence: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "Liquidus/solidus proxies depend on the chosen solid-fraction cutoffs and path resolution.",
        "A Scheil path assumes local interfacial equilibrium, complete liquid mixing, and negligible solid diffusion unless its source states otherwise.",
        "Thermal-gradient metrics describe imposed conditions and do not by themselves predict melt-pool defects.",
    )

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output.pop("path")
        output["limitations"] = list(self.limitations)
        return output


def _prepare_path(
    temperature_C: Sequence[float], solid_fraction: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    temperature = np.asarray(temperature_C, dtype=float)
    fraction = np.asarray(solid_fraction, dtype=float)
    if temperature.ndim != 1 or fraction.ndim != 1 or len(temperature) != len(fraction):
        raise SolidificationAnalysisError("Temperature and solid fraction must be equal-length 1D arrays")
    finite = np.isfinite(temperature) & np.isfinite(fraction)
    temperature, fraction = temperature[finite], fraction[finite]
    if len(temperature) < 4:
        raise SolidificationAnalysisError("At least four finite solidification states are required")
    if ((fraction < 0) | (fraction > 1)).any():
        raise SolidificationAnalysisError("Solid fraction must lie in [0, 1]")
    order = np.argsort(fraction, kind="stable")
    frame = pd.DataFrame({"fraction": fraction[order], "temperature": temperature[order]})
    frame = frame.groupby("fraction", as_index=False)["temperature"].mean()
    fraction = frame["fraction"].to_numpy()
    temperature = frame["temperature"].to_numpy()
    if len(fraction) < 4:
        raise SolidificationAnalysisError("Need at least four distinct solid fractions")
    temperature_increases = np.diff(temperature) > max(1e-8, np.ptp(temperature) * 1e-6)
    if temperature_increases.any():
        raise SolidificationAnalysisError(
            "Temperature must be non-increasing as solid fraction increases; inspect the supplied path"
        )
    return temperature, fraction


def temperature_at_solid_fraction(
    temperature_C: Sequence[float], solid_fraction: Sequence[float], query_fraction: float
) -> float:
    temperature, fraction = _prepare_path(temperature_C, solid_fraction)
    if not fraction[0] <= query_fraction <= fraction[-1]:
        raise SolidificationAnalysisError(
            f"Requested solid fraction {query_fraction:g} is outside supplied range "
            f"[{fraction[0]:g}, {fraction[-1]:g}]"
        )
    return float(np.interp(query_fraction, fraction, temperature))


def _segregation_metrics(
    fraction: np.ndarray,
    composition_paths: Mapping[str, Sequence[float]] | None,
) -> dict[str, dict[str, float | None]]:
    if not composition_paths:
        return {}
    metrics: dict[str, dict[str, float | None]] = {}
    for element, raw in composition_paths.items():
        values = np.asarray(raw, dtype=float)
        if values.shape != fraction.shape:
            raise SolidificationAnalysisError(
                f"Composition path for {element} must have {len(fraction)} values after path preparation"
            )
        if not np.isfinite(values).all() or (values < 0).any():
            raise SolidificationAnalysisError(f"Composition path for {element} is invalid")
        initial = float(values[0])
        terminal = float(values[-1])
        minimum = float(np.min(values))
        metrics[str(element)] = {
            "initial": initial,
            "terminal": terminal,
            "terminal_to_initial_ratio": terminal / initial if initial > 0 else None,
            "max_to_min_ratio": float(np.max(values) / minimum) if minimum > 0 else None,
            "range": float(np.ptp(values)),
        }
    return metrics


def _align_composition_paths(
    original_temperature: Sequence[float],
    original_fraction: Sequence[float],
    prepared_fraction: np.ndarray,
    composition_paths: Mapping[str, Sequence[float]] | None,
) -> dict[str, np.ndarray] | None:
    if not composition_paths:
        return None
    raw_temperature = np.asarray(original_temperature, dtype=float)
    raw_fraction = np.asarray(original_fraction, dtype=float)
    base_finite = np.isfinite(raw_temperature) & np.isfinite(raw_fraction)
    aligned: dict[str, np.ndarray] = {}
    for element, raw_values in composition_paths.items():
        values = np.asarray(raw_values, dtype=float)
        if values.shape != raw_fraction.shape:
            raise SolidificationAnalysisError(
                f"Composition path for {element} must match the original solid-fraction array"
            )
        finite = base_finite & np.isfinite(values)
        if finite.sum() < 2 or (values[finite] < 0).any():
            raise SolidificationAnalysisError(f"Composition path for {element} is invalid")
        table = pd.DataFrame(
            {"fraction": raw_fraction[finite], "composition": values[finite]}
        ).groupby("fraction", as_index=False)["composition"].mean()
        aligned[str(element)] = np.interp(
            prepared_fraction,
            table["fraction"].to_numpy(),
            table["composition"].to_numpy(),
        )
    return aligned


def analyze_solidification_path(
    temperature_C: Sequence[float],
    solid_fraction: Sequence[float],
    *,
    composition_paths: Mapping[str, Sequence[float]] | None = None,
    liquidus_fraction: float = 0.01,
    solidus_fraction: float = 0.99,
    thermal_gradient_K_per_mm: float | None = None,
    interface_velocity_mm_per_s: float | None = None,
    source: Mapping[str, Any] | None = None,
) -> SolidificationAssessment:
    """Calculate path-derived metrics without assigning unsupported risk categories."""

    if not 0 <= liquidus_fraction < solidus_fraction <= 1:
        raise SolidificationAnalysisError("Solid-fraction definitions must be increasing in [0, 1]")
    temperature, fraction = _prepare_path(temperature_C, solid_fraction)
    if fraction[0] > liquidus_fraction or fraction[-1] < solidus_fraction:
        raise SolidificationAnalysisError(
            "Supplied path does not span the requested liquidus/solidus fraction definitions"
        )
    liquidus = float(np.interp(liquidus_fraction, fraction, temperature))
    solidus = float(np.interp(solidus_fraction, fraction, temperature))
    if thermal_gradient_K_per_mm is not None and thermal_gradient_K_per_mm <= 0:
        raise SolidificationAnalysisError("Thermal gradient must be positive")
    if interface_velocity_mm_per_s is not None and interface_velocity_mm_per_s <= 0:
        raise SolidificationAnalysisError("Interface velocity must be positive")
    gradient_velocity_ratio = None
    cooling_rate = None
    if thermal_gradient_K_per_mm is not None and interface_velocity_mm_per_s is not None:
        gradient_velocity_ratio = thermal_gradient_K_per_mm / interface_velocity_mm_per_s
        cooling_rate = thermal_gradient_K_per_mm * interface_velocity_mm_per_s
    path = pd.DataFrame({"solid_fraction": fraction, "temperature_C": temperature})
    confidence = {
        "basis": "input coverage and numerical resolution; not a probability",
        "n_states": int(len(fraction)),
        "solid_fraction_range": [float(fraction[0]), float(fraction[-1])],
        "source_documented": bool(source),
    }
    return SolidificationAssessment(
        liquidus_proxy_C=liquidus,
        solidus_proxy_C=solidus,
        freezing_range_C=liquidus - solidus,
        liquidus_fraction_definition=liquidus_fraction,
        solidus_fraction_definition=solidus_fraction,
        segregation=_segregation_metrics(
            fraction,
            _align_composition_paths(
                temperature_C, solid_fraction, fraction, composition_paths
            ),
        ),
        thermal_metrics={
            "thermal_gradient_K_per_mm": thermal_gradient_K_per_mm,
            "interface_velocity_mm_per_s": interface_velocity_mm_per_s,
            "G_over_R_K_s_per_mm2": gradient_velocity_ratio,
            "G_times_R_cooling_rate_K_per_s": cooling_rate,
        },
        source=dict(source or {"status": "not documented"}),
        path=path,
        confidence=confidence,
    )
