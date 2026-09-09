"""Transparent peak-position matching without uncalibrated probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from crystallography.diffraction import XRDPattern
from scientific import ResultClassification, ScientificMetadata


@dataclass
class PeakMatchResult:
    phase_scores: pd.DataFrame
    assignments: pd.DataFrame
    tolerance_deg: float
    metadata: ScientificMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_scores": self.phase_scores.to_dict(orient="records"),
            "assignments": self.assignments.to_dict(orient="records"),
            "tolerance_deg": self.tolerance_deg,
            "metadata": self.metadata.to_dict(),
        }


def match_peaks(
    experimental_two_theta_deg: Sequence[float],
    references: Mapping[str, XRDPattern | pd.DataFrame],
    *,
    tolerance_deg: float = 0.2,
    experimental_intensity: Sequence[float] | None = None,
) -> PeakMatchResult:
    """Match experimental to reference peak positions and report similarity scores.

    Scores are deterministic coverage metrics in [0, 1], explicitly *not*
    calibrated phase probabilities.
    """

    observed = np.asarray(experimental_two_theta_deg, dtype=float)
    if observed.ndim != 1 or observed.size == 0 or not np.all(np.isfinite(observed)):
        raise ValueError("experimental_two_theta_deg must be a non-empty finite 1D sequence.")
    tolerance = float(tolerance_deg)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance_deg must be finite and positive.")
    if experimental_intensity is None:
        observed_intensity = np.ones_like(observed)
    else:
        observed_intensity = np.asarray(experimental_intensity, dtype=float)
        if observed_intensity.shape != observed.shape:
            raise ValueError("experimental_intensity must match experimental peak positions.")
        if not np.all(np.isfinite(observed_intensity)) or np.any(observed_intensity < 0):
            raise ValueError("experimental_intensity must be finite and non-negative.")
    if not references:
        raise ValueError("At least one reference pattern is required.")
    assignments: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for phase, reference in references.items():
        table = reference.peaks if isinstance(reference, XRDPattern) else reference
        required = {"two_theta_deg", "scaled_intensity"}
        if not required.issubset(table.columns):
            raise ValueError(f"Reference {phase!r} must contain {sorted(required)} columns.")
        ref_angles = table["two_theta_deg"].to_numpy(dtype=float)
        ref_intensity = table["scaled_intensity"].to_numpy(dtype=float)
        if ref_angles.size == 0:
            score_rows.append(
                {
                    "phase": phase,
                    "similarity_score": 0.0,
                    "matched_reference_peaks": 0,
                    "reference_peak_count": 0,
                    "rms_two_theta_residual_deg": np.nan,
                }
            )
            continue
        ref_weights = ref_intensity / max(float(ref_intensity.sum()), np.finfo(float).eps)
        matched_weight = 0.0
        residuals: list[float] = []
        for index, (angle, intensity, weight) in enumerate(
            zip(ref_angles, ref_intensity, ref_weights, strict=True)
        ):
            nearest_index = int(np.argmin(np.abs(observed - angle)))
            residual = float(observed[nearest_index] - angle)
            matched = abs(residual) <= tolerance
            if matched:
                matched_weight += float(weight)
                residuals.append(residual)
            assignments.append(
                {
                    "phase": phase,
                    "reference_two_theta_deg": float(angle),
                    "reference_scaled_intensity": float(intensity),
                    "experimental_two_theta_deg": float(observed[nearest_index]),
                    "experimental_intensity": float(observed_intensity[nearest_index]),
                    "residual_deg": residual,
                    "matched": matched,
                    "hkl": table.iloc[index].get("hkl"),
                }
            )
        score_rows.append(
            {
                "phase": phase,
                "similarity_score": matched_weight,
                "matched_reference_peaks": len(residuals),
                "reference_peak_count": int(ref_angles.size),
                "rms_two_theta_residual_deg": (
                    float(np.sqrt(np.mean(np.square(residuals)))) if residuals else np.nan
                ),
            }
        )
    return PeakMatchResult(
        phase_scores=pd.DataFrame(score_rows).sort_values(
            "similarity_score", ascending=False, ignore_index=True
        ),
        assignments=pd.DataFrame(assignments),
        tolerance_deg=tolerance,
        metadata=ScientificMetadata(
            classification=ResultClassification.ENGINEERING_SCREENING,
            confidence="screening-level; depends on reference completeness and user tolerance",
            limitations=(
                "Similarity score is intensity-weighted reference-peak coverage, not a calibrated probability.",
                "Peak overlap, preferred orientation, zero shift, broadening, fluorescence, and unmodelled phases can bias matches.",
                "Confirmation requires full-pattern refinement and appropriate experimental controls.",
            ),
            provenance=tuple(
                item
                for reference in references.values()
                if isinstance(reference, XRDPattern)
                for item in reference.metadata.provenance
            ),
        ),
    )
