"""Multiphase display-pattern construction from calculated stick patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crystallography.diffraction import XRDPattern
from scientific import ResultClassification, ScientificMetadata


@dataclass
class MultiphaseXRDPattern:
    profile: pd.DataFrame
    peaks: pd.DataFrame
    display_weights: Mapping[str, float]
    fwhm_deg: float
    metadata: ScientificMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(orient="records"),
            "peaks": self.peaks.to_dict(orient="records"),
            "display_weights": dict(self.display_weights),
            "fwhm_deg": self.fwhm_deg,
            "metadata": self.metadata.to_dict(),
        }


def simulate_multiphase_pattern(
    patterns: Mapping[str, XRDPattern],
    *,
    display_weights: Mapping[str, float] | None = None,
    fwhm_deg: float = 0.15,
    step_deg: float = 0.02,
) -> MultiphaseXRDPattern:
    """Broaden and sum normalized phase patterns for interactive comparison.

    The weights are display weights, not a quantitative phase-fraction/RIR
    model. The function intentionally refuses the name ``phase_fractions``.
    """

    if not patterns:
        raise ValueError("At least one phase pattern is required.")
    phase_names = list(patterns)
    weights = dict(display_weights or {phase: 1.0 for phase in phase_names})
    if set(weights) != set(phase_names):
        raise ValueError("display_weights must contain exactly the supplied pattern names.")
    if any(not np.isfinite(float(value)) or float(value) < 0 for value in weights.values()):
        raise ValueError("display weights must be finite and non-negative.")
    total = sum(float(value) for value in weights.values())
    if total <= 0:
        raise ValueError("At least one display weight must be positive.")
    normalized = {phase: float(value) / total for phase, value in weights.items()}
    fwhm = float(fwhm_deg)
    step = float(step_deg)
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("fwhm_deg must be finite and positive.")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step_deg must be finite and positive.")
    lower = min(pattern.two_theta_range_deg[0] for pattern in patterns.values())
    upper = max(pattern.two_theta_range_deg[1] for pattern in patterns.values())
    grid = np.arange(lower, upper + 0.5 * step, step)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    total_profile = np.zeros_like(grid)
    phase_profiles: dict[str, np.ndarray] = {}
    peak_tables: list[pd.DataFrame] = []
    for phase, pattern in patterns.items():
        profile = np.zeros_like(grid)
        for row in pattern.peaks.itertuples():
            amplitude = normalized[phase] * float(row.scaled_intensity)
            profile += amplitude * np.exp(-0.5 * ((grid - float(row.two_theta_deg)) / sigma) ** 2)
        phase_profiles[phase] = profile
        total_profile += profile
        table = pattern.peaks.copy()
        table["display_weight"] = normalized[phase]
        table["weighted_scaled_intensity"] = table["scaled_intensity"] * normalized[phase]
        peak_tables.append(table)
    profile_data: dict[str, Any] = {
        "two_theta_deg": grid,
        "summed_display_intensity": total_profile,
    }
    profile_data.update({f"intensity_{phase}": values for phase, values in phase_profiles.items()})
    return MultiphaseXRDPattern(
        profile=pd.DataFrame(profile_data),
        peaks=pd.concat(peak_tables, ignore_index=True),
        display_weights=normalized,
        fwhm_deg=fwhm,
        metadata=ScientificMetadata(
            classification=ResultClassification.ENGINEERING_SCREENING,
            confidence="qualitative visualization only",
            limitations=(
                "Display weights are not quantitative phase fractions.",
                "A common user-selected Gaussian width is applied to all reflections.",
                "Quantitative phase analysis requires calibrated scale factors, absorption, and phase-specific reference-intensity treatment or Rietveld refinement.",
            ),
            provenance=tuple(
                item for pattern in patterns.values() for item in pattern.metadata.provenance
            ),
        ),
    )
