"""Solidification-cracking indicators without unsupported categorical claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.signal import savgol_filter

from .solidification import SolidificationAnalysisError, _prepare_path


@dataclass(frozen=True)
class CrackingIndicator:
    name: str
    value: float
    unit: str
    solid_fraction_interval: tuple[float, float]
    classification: str
    citation: str
    confidence: dict[str, Any]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["limitations"] = list(self.limitations)
        return output


def kou_cracking_index(
    temperature_C: Sequence[float],
    solid_fraction: Sequence[float],
    *,
    vulnerable_fraction_interval: tuple[float, float] = (0.90, 0.99),
    smoothing_window: int | None = None,
) -> CrackingIndicator:
    """Calculate max |dT/d(sqrt(fs))| in a declared terminal interval.

    This implements a Kou-type solidification cracking index.  It deliberately
    returns no low/medium/high category because those bounds need process- and
    alloy-family calibration.
    """

    temperature, fraction = _prepare_path(temperature_C, solid_fraction)
    lower, upper = vulnerable_fraction_interval
    if not 0 < lower < upper <= 1:
        raise SolidificationAnalysisError("Vulnerable fraction interval must lie in (0, 1]")
    mask = (fraction >= lower) & (fraction <= upper)
    if mask.sum() < 3:
        raise SolidificationAnalysisError(
            "At least three path states are required inside the vulnerable fraction interval"
        )
    local_temperature = temperature[mask]
    local_fraction = fraction[mask]
    if smoothing_window is not None:
        window = min(smoothing_window, len(local_temperature) if len(local_temperature) % 2 else len(local_temperature) - 1)
        if window >= 3:
            local_temperature = savgol_filter(local_temperature, window, min(2, window - 1))
    derivative = np.gradient(local_temperature, np.sqrt(local_fraction))
    value = float(np.max(np.abs(derivative)))
    return CrackingIndicator(
        name="Kou-type terminal solidification slope",
        value=value,
        unit="K",
        solid_fraction_interval=(lower, upper),
        classification="physics_calculated_indicator",
        citation="S. Kou, Acta Materialia 88 (2015) 366-374, https://doi.org/10.1016/j.actamat.2015.01.034",
        confidence={
            "basis": "path discretization; no probability calibration",
            "points_in_interval": int(mask.sum()),
            "smoothing_window": smoothing_window,
        },
        limitations=(
            "The indicator is comparative; no universal Cu-Al-Ni LPBF risk threshold is asserted.",
            "Results depend strongly on thermodynamic database, solidification model, and terminal path resolution.",
            "Strain rate, feeding geometry, grain structure, residual stress, and process conditions are outside this scalar indicator.",
        ),
    )
