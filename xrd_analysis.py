"""Experimental powder-XRD preprocessing, peak finding, and reference matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.sparse.linalg import spsolve


class XRDAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class ExperimentalPeak:
    two_theta_deg: float
    intensity: float
    prominence: float
    fwhm_deg: float | None
    d_spacing_angstrom: float | None


@dataclass(frozen=True)
class PeakMatch:
    experimental_two_theta_deg: float
    reference_two_theta_deg: float
    delta_two_theta_deg: float
    phase: str
    hkl: str | None
    reference_d_spacing_angstrom: float | None
    angular_likelihood: float


@dataclass
class XRDAnalysisResult:
    pattern: pd.DataFrame
    peaks: list[ExperimentalPeak]
    matches: list[PeakMatch]
    phase_matching_scores: dict[str, float]
    wavelength_angstrom: float | None
    parameters: dict[str, Any]
    classification: str = "experimental_data_analysis"
    confidence: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "Reference matching alone cannot distinguish all overlapping or textured phases.",
        "Reported phase scores are uncalibrated matching scores, not probabilities or phase fractions.",
        "Zero shift, specimen displacement, preferred orientation, and instrumental broadening are not refined.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "peaks": [asdict(item) for item in self.peaks],
            "matches": [asdict(item) for item in self.matches],
            "phase_matching_scores": self.phase_matching_scores,
            "phase_matching_probability": None,
            "confidence": self.confidence,
            "parameters": self.parameters,
            "limitations": list(self.limitations),
        }


def _read_table(data: str | Path | BinaryIO | pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise XRDAnalysisError("XRD arrays must contain at least two columns")
        return pd.DataFrame(data[:, :2], columns=["two_theta", "intensity"])
    try:
        return pd.read_csv(data, sep=None, engine="python", comment="#")
    except Exception as exc:
        raise XRDAnalysisError(f"Could not parse XRD table: {exc}") from exc


def _find_column(frame: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    normalized = {
        str(column).strip().lower().replace(" ", "_").replace("°", "deg"): column
        for column in frame.columns
    }
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def load_xrd_pattern(data: str | Path | BinaryIO | pd.DataFrame | np.ndarray) -> pd.DataFrame:
    frame = _read_table(data)
    angle_column = _find_column(
        frame, ("2theta", "two_theta", "2theta_deg", "two_theta_deg", "angle")
    )
    intensity_column = _find_column(
        frame, ("intensity", "counts", "count", "i", "intensity_counts")
    )
    if angle_column is None or intensity_column is None:
        if frame.shape[1] < 2:
            raise XRDAnalysisError("Need 2theta and intensity columns")
        angle_column, intensity_column = frame.columns[:2]
    clean = pd.DataFrame(
        {
            "two_theta_deg": pd.to_numeric(frame[angle_column], errors="coerce"),
            "intensity_raw": pd.to_numeric(frame[intensity_column], errors="coerce"),
        }
    ).dropna()
    clean = clean.groupby("two_theta_deg", as_index=False)["intensity_raw"].mean()
    clean = clean.sort_values("two_theta_deg", ignore_index=True)
    if len(clean) < 15:
        raise XRDAnalysisError("At least 15 finite pattern points are required")
    if (np.diff(clean["two_theta_deg"]) <= 0).any():
        raise XRDAnalysisError("2theta values must be strictly increasing after duplicate averaging")
    return clean


def asymmetric_least_squares_baseline(
    intensity: Sequence[float], *, smoothness: float = 1e6, asymmetry: float = 0.01, iterations: int = 10
) -> np.ndarray:
    """Eilers asymmetric least-squares baseline (algorithmic parameters exposed)."""

    y = np.asarray(intensity, dtype=float)
    if smoothness <= 0 or not 0 < asymmetry < 1 or iterations < 1:
        raise ValueError("Invalid asymmetric least-squares parameters")
    second_difference = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(len(y) - 2, len(y)))
    penalty = smoothness * second_difference.T @ second_difference
    weights = np.ones(len(y))
    baseline = np.zeros_like(y)
    for _ in range(iterations):
        weight_matrix = sparse.spdiags(weights, 0, len(y), len(y))
        baseline = np.asarray(spsolve((weight_matrix + penalty).tocsc(), weights * y))
        weights = asymmetry * (y > baseline) + (1.0 - asymmetry) * (y <= baseline)
    return baseline


def _odd_window(length: int, requested: int, polynomial_order: int) -> int:
    window = min(requested, length if length % 2 else length - 1)
    minimum = polynomial_order + 2 + ((polynomial_order + 2) % 2 == 0)
    window = max(window, minimum)
    if window >= length:
        window = length - 1 if length % 2 == 0 else length
    return max(3, window)


def preprocess_pattern(
    pattern: pd.DataFrame,
    *,
    baseline_smoothness: float = 1e6,
    baseline_asymmetry: float = 0.01,
    smoothing_window: int = 11,
    polynomial_order: int = 3,
) -> pd.DataFrame:
    out = pattern.copy()
    y = out["intensity_raw"].to_numpy(dtype=float)
    baseline = asymmetric_least_squares_baseline(
        y, smoothness=baseline_smoothness, asymmetry=baseline_asymmetry
    )
    corrected = np.clip(y - baseline, 0.0, None)
    window = _odd_window(len(corrected), smoothing_window, polynomial_order)
    smoothed = savgol_filter(corrected, window_length=window, polyorder=min(polynomial_order, window - 1))
    smoothed = np.clip(smoothed, 0.0, None)
    maximum = float(np.max(smoothed))
    normalized = 100.0 * smoothed / maximum if maximum > 0 else np.zeros_like(smoothed)
    out["baseline"] = baseline
    out["intensity_corrected"] = corrected
    out["intensity_normalized"] = normalized
    return out


def detect_peaks(
    pattern: pd.DataFrame,
    *,
    minimum_prominence_pct: float = 2.0,
    minimum_distance_deg: float = 0.15,
    wavelength_angstrom: float | None = 1.5406,
) -> list[ExperimentalPeak]:
    x = pattern["two_theta_deg"].to_numpy(dtype=float)
    y = pattern["intensity_normalized"].to_numpy(dtype=float)
    step = float(np.median(np.diff(x)))
    distance_points = max(1, int(np.ceil(minimum_distance_deg / step)))
    indices, properties = find_peaks(y, prominence=minimum_prominence_pct, distance=distance_points)
    if indices.size:
        widths = peak_widths(y, indices, rel_height=0.5)[0] * step
    else:
        widths = np.array([], dtype=float)
    peaks: list[ExperimentalPeak] = []
    for order, index in enumerate(indices):
        theta_rad = np.deg2rad(x[index] / 2.0)
        d_spacing = None
        if wavelength_angstrom is not None and np.sin(theta_rad) > 0:
            d_spacing = float(wavelength_angstrom / (2.0 * np.sin(theta_rad)))
        peaks.append(
            ExperimentalPeak(
                two_theta_deg=float(x[index]),
                intensity=float(y[index]),
                prominence=float(properties["prominences"][order]),
                fwhm_deg=float(widths[order]) if np.isfinite(widths[order]) else None,
                d_spacing_angstrom=d_spacing,
            )
        )
    return peaks


def _reference_frame(reference_peaks: pd.DataFrame | Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = reference_peaks.copy() if isinstance(reference_peaks, pd.DataFrame) else pd.DataFrame(reference_peaks)
    aliases = {
        "two_theta_deg": ("two_theta_deg", "two_theta", "2theta", "2theta_deg"),
        "phase": ("phase", "phase_name"),
        "hkl": ("hkl",),
        "d_spacing_angstrom": ("d_spacing_angstrom", "d_spacing", "d_angstrom"),
        "intensity": ("intensity", "relative_intensity"),
    }
    renamed: dict[str, str] = {}
    for canonical, choices in aliases.items():
        found = _find_column(frame, choices)
        if found is not None:
            renamed[found] = canonical
    frame = frame.rename(columns=renamed)
    if "two_theta_deg" not in frame or "phase" not in frame:
        raise XRDAnalysisError("Reference peaks require phase and two_theta_deg")
    frame["two_theta_deg"] = pd.to_numeric(frame["two_theta_deg"], errors="coerce")
    frame = frame.dropna(subset=["two_theta_deg", "phase"]).reset_index(drop=True)
    return frame


def match_reference_peaks(
    peaks: Sequence[ExperimentalPeak],
    reference_peaks: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    tolerance_deg: float = 0.30,
) -> tuple[list[PeakMatch], dict[str, float]]:
    """One-to-one peak assignment and an explicitly uncalibrated phase score."""

    reference = _reference_frame(reference_peaks)
    phases = sorted(reference["phase"].astype(str).unique())
    if not peaks or reference.empty:
        return [], {phase: 0.0 for phase in phases}
    experimental_angles = np.array([peak.two_theta_deg for peak in peaks])
    reference_angles = reference["two_theta_deg"].to_numpy(dtype=float)
    delta = np.abs(experimental_angles[:, None] - reference_angles[None, :])
    row_indices, column_indices = linear_sum_assignment(delta)
    matches: list[PeakMatch] = []
    explained: dict[str, float] = {phase: 0.0 for phase in phases}
    total_intensity = sum(peak.intensity for peak in peaks) or 1.0
    for row, column in zip(row_indices, column_indices, strict=True):
        if delta[row, column] > tolerance_deg:
            continue
        ref = reference.iloc[column]
        angular_likelihood = float(np.exp(-0.5 * (delta[row, column] / tolerance_deg) ** 2))
        phase = str(ref["phase"])
        explained[phase] += peaks[row].intensity * angular_likelihood
        matches.append(
            PeakMatch(
                experimental_two_theta_deg=peaks[row].two_theta_deg,
                reference_two_theta_deg=float(ref["two_theta_deg"]),
                delta_two_theta_deg=float(experimental_angles[row] - reference_angles[column]),
                phase=phase,
                hkl=str(ref["hkl"]) if "hkl" in ref and pd.notna(ref["hkl"]) else None,
                reference_d_spacing_angstrom=float(ref["d_spacing_angstrom"])
                if "d_spacing_angstrom" in ref and pd.notna(ref["d_spacing_angstrom"])
                else None,
                angular_likelihood=angular_likelihood,
            )
        )
    scores = {phase: float(value / total_intensity) for phase, value in explained.items()}
    return matches, scores


def analyze_xrd(
    data: str | Path | BinaryIO | pd.DataFrame | np.ndarray,
    reference_peaks: pd.DataFrame | Sequence[Mapping[str, Any]] | None = None,
    *,
    wavelength_angstrom: float | None = 1.5406,
    minimum_prominence_pct: float = 2.0,
    minimum_distance_deg: float = 0.15,
    matching_tolerance_deg: float = 0.30,
    baseline_smoothness: float = 1e6,
    baseline_asymmetry: float = 0.01,
) -> XRDAnalysisResult:
    raw = load_xrd_pattern(data)
    pattern = preprocess_pattern(
        raw,
        baseline_smoothness=baseline_smoothness,
        baseline_asymmetry=baseline_asymmetry,
    )
    peaks = detect_peaks(
        pattern,
        minimum_prominence_pct=minimum_prominence_pct,
        minimum_distance_deg=minimum_distance_deg,
        wavelength_angstrom=wavelength_angstrom,
    )
    matches: list[PeakMatch] = []
    scores: dict[str, float] = {}
    if reference_peaks is not None:
        matches, scores = match_reference_peaks(
            peaks, reference_peaks, tolerance_deg=matching_tolerance_deg
        )
    confidence = {
        "detected_peak_count": len(peaks),
        "matched_peak_count": len(matches),
        "median_step_deg": float(np.median(np.diff(pattern["two_theta_deg"]))),
        "reference_supplied": reference_peaks is not None,
        "calibrated_phase_probability_available": False,
    }
    return XRDAnalysisResult(
        pattern,
        peaks,
        matches,
        scores,
        wavelength_angstrom,
        {
            "minimum_prominence_pct": minimum_prominence_pct,
            "minimum_distance_deg": minimum_distance_deg,
            "matching_tolerance_deg": matching_tolerance_deg,
            "baseline_smoothness": baseline_smoothness,
            "baseline_asymmetry": baseline_asymmetry,
        },
        confidence=confidence,
    )
