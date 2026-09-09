"""DSC branch detection and operational martensitic transformation temperatures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


class DSCAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class DSCEvent:
    branch: str
    onset_C: float
    peak_C: float
    finish_C: float
    signal_polarity: str
    signal_to_baseline_noise: float | None
    integrated_abs_signal_heatflow_degC: float
    n_points: int
    method: str = "linear endpoint baseline; 5-95% cumulative absolute event area"


@dataclass
class DSCAnalysisResult:
    processed_curve: pd.DataFrame
    events: list[DSCEvent]
    temperatures_C: dict[str, float | None]
    classification: str = "experimental_data_analysis"
    confidence: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "Automatically detected temperatures are operational signal bounds, not a substitute for instrument-software tangent analysis.",
        "Baseline choice, scan rate, sample mass, heat-flow sign convention, and overlapping events affect the result.",
        "The dominant event per heating/cooling branch is selected; secondary transformations require manual review.",
        "Temperature calibration uncertainty is not available unless supplied with the experiment metadata.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "temperatures_C": self.temperatures_C,
            "events": [asdict(event) for event in self.events],
            "confidence": self.confidence,
            "limitations": list(self.limitations),
        }


def _read_dsc(data: str | Path | BinaryIO | pd.DataFrame | np.ndarray) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] < 2:
            raise DSCAnalysisError("DSC arrays require temperature and heat-flow columns")
        frame = pd.DataFrame(data[:, :2], columns=["temperature_C", "heat_flow"])
    else:
        try:
            frame = pd.read_csv(data, sep=None, engine="python", comment="#")
        except Exception as exc:
            raise DSCAnalysisError(f"Could not parse DSC table: {exc}") from exc
    normalized = {
        str(column).strip().lower().replace(" ", "_").replace("°", ""): column
        for column in frame.columns
    }
    temp_aliases = ("temperature_c", "temperature", "temp_c", "temp", "t_c")
    flow_aliases = ("heat_flow", "heatflow", "dsc", "signal", "mw", "heat_flow_mw")
    temp_column = next((normalized[name] for name in temp_aliases if name in normalized), None)
    flow_column = next((normalized[name] for name in flow_aliases if name in normalized), None)
    if temp_column is None or flow_column is None:
        if frame.shape[1] < 2:
            raise DSCAnalysisError("Need temperature and heat-flow columns")
        temp_column, flow_column = frame.columns[:2]
    clean = pd.DataFrame(
        {
            "temperature_C": pd.to_numeric(frame[temp_column], errors="coerce"),
            "heat_flow": pd.to_numeric(frame[flow_column], errors="coerce"),
        }
    ).dropna()
    if len(clean) < 20:
        raise DSCAnalysisError("At least 20 finite DSC points are required")
    return clean.reset_index(drop=True)


def _fill_zero_signs(signs: np.ndarray) -> np.ndarray:
    signs = signs.copy()
    for index in range(1, len(signs)):
        if signs[index] == 0:
            signs[index] = signs[index - 1]
    for index in range(len(signs) - 2, -1, -1):
        if signs[index] == 0:
            signs[index] = signs[index + 1]
    return signs


def _monotonic_segments(temperature: np.ndarray, minimum_points: int) -> list[np.ndarray]:
    signs = _fill_zero_signs(np.sign(np.diff(temperature)))
    if not np.any(signs):
        return []
    changes = np.where(signs[1:] != signs[:-1])[0] + 1
    boundaries = np.r_[0, changes + 1, len(temperature)]
    segments: list[np.ndarray] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        index = np.arange(start, stop)
        if len(index) >= minimum_points and abs(temperature[index[-1]] - temperature[index[0]]) > 0:
            segments.append(index)
    return segments


def _savgol(values: np.ndarray, window: int, order: int) -> np.ndarray:
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window = min(maximum, max(order + 2 + ((order + 2) % 2 == 0), window))
    if window <= order or window < 3:
        return values.copy()
    return savgol_filter(values, window_length=window, polyorder=order)


def _cumulative_event_bounds(
    temperature: np.ndarray, event_signal: np.ndarray, lower_fraction: float, upper_fraction: float
) -> tuple[int, int, float]:
    increments = 0.5 * (event_signal[:-1] + event_signal[1:]) * np.abs(np.diff(temperature))
    cumulative = np.r_[0.0, np.cumsum(increments)]
    total = float(cumulative[-1])
    if total <= 0:
        peak = int(np.argmax(event_signal))
        return peak, peak, 0.0
    start = int(np.searchsorted(cumulative, lower_fraction * total, side="left"))
    finish = int(np.searchsorted(cumulative, upper_fraction * total, side="left"))
    return min(start, len(temperature) - 1), min(finish, len(temperature) - 1), total


def _analyze_branch(
    temperature: np.ndarray,
    heat_flow: np.ndarray,
    *,
    baseline_fraction: float,
    area_bounds: tuple[float, float],
    smoothing_window: int,
) -> tuple[DSCEvent, np.ndarray, np.ndarray]:
    count = len(temperature)
    edge = max(3, int(np.ceil(count * baseline_fraction)))
    baseline_indices = np.r_[np.arange(edge), np.arange(count - edge, count)]
    coefficients = np.polyfit(temperature[baseline_indices], heat_flow[baseline_indices], deg=1)
    baseline = np.polyval(coefficients, temperature)
    residual = _savgol(heat_flow - baseline, smoothing_window, 3)
    interior = np.arange(edge, count - edge)
    if interior.size < 3:
        raise DSCAnalysisError("Branch is too short for endpoint baseline and event detection")
    peak_index = int(interior[np.argmax(np.abs(residual[interior]))])
    polarity = 1.0 if residual[peak_index] >= 0 else -1.0
    event_signal = np.clip(polarity * residual, 0.0, None)
    start, finish, total_area = _cumulative_event_bounds(
        temperature, event_signal, area_bounds[0], area_bounds[1]
    )
    baseline_residuals = np.r_[residual[:edge], residual[-edge:]]
    noise = float(np.std(baseline_residuals, ddof=1)) if len(baseline_residuals) > 1 else 0.0
    snr = float(abs(residual[peak_index]) / noise) if noise > 0 else None
    direction = "heating" if temperature[-1] > temperature[0] else "cooling"
    return (
        DSCEvent(
            branch=direction,
            onset_C=float(temperature[start]),
            peak_C=float(temperature[peak_index]),
            finish_C=float(temperature[finish]),
            signal_polarity="positive" if polarity > 0 else "negative",
            signal_to_baseline_noise=snr,
            integrated_abs_signal_heatflow_degC=total_area,
            n_points=count,
        ),
        baseline,
        residual,
    )


def analyze_dsc(
    data: str | Path | BinaryIO | pd.DataFrame | np.ndarray,
    *,
    baseline_fraction: float = 0.15,
    area_bounds: tuple[float, float] = (0.05, 0.95),
    smoothing_window: int = 11,
    minimum_branch_points: int = 20,
    temperature_calibration_uncertainty_C: float | None = None,
) -> DSCAnalysisResult:
    if not 0.05 <= baseline_fraction <= 0.30:
        raise DSCAnalysisError("baseline_fraction must lie between 0.05 and 0.30")
    if not 0 <= area_bounds[0] < area_bounds[1] <= 1:
        raise DSCAnalysisError("area_bounds must be increasing and lie in [0, 1]")
    curve = _read_dsc(data)
    curve["branch"] = "unassigned"
    curve["baseline"] = np.nan
    curve["residual"] = np.nan
    temperature = curve["temperature_C"].to_numpy(dtype=float)
    heat_flow = curve["heat_flow"].to_numpy(dtype=float)
    events: list[DSCEvent] = []
    for number, indices in enumerate(_monotonic_segments(temperature, minimum_branch_points), start=1):
        event, baseline, residual = _analyze_branch(
            temperature[indices],
            heat_flow[indices],
            baseline_fraction=baseline_fraction,
            area_bounds=area_bounds,
            smoothing_window=smoothing_window,
        )
        events.append(event)
        curve.loc[indices, "branch"] = f"{event.branch}_{number}"
        curve.loc[indices, "baseline"] = baseline
        curve.loc[indices, "residual"] = residual
    if not events:
        raise DSCAnalysisError("No sufficiently long monotonic heating or cooling branch was found")
    selected: dict[str, DSCEvent] = {}
    for direction in ("cooling", "heating"):
        candidates = [event for event in events if event.branch == direction]
        if candidates:
            selected[direction] = max(
                candidates,
                key=lambda event: event.signal_to_baseline_noise
                if event.signal_to_baseline_noise is not None
                else -np.inf,
            )
    cooling = selected.get("cooling")
    heating = selected.get("heating")
    temperatures = {
        "Ms": cooling.onset_C if cooling else None,
        "Mf": cooling.finish_C if cooling else None,
        "As": heating.onset_C if heating else None,
        "Af": heating.finish_C if heating else None,
    }
    temperature_steps = np.abs(np.diff(temperature))
    confidence = {
        "basis": "signal diagnostics; not a calibrated probability",
        "selected_branch_snr": {
            key: value.signal_to_baseline_noise for key, value in selected.items()
        },
        "median_temperature_step_C": float(np.median(temperature_steps[temperature_steps > 0])),
        "temperature_calibration_uncertainty_C": temperature_calibration_uncertainty_C,
        "heating_and_cooling_available": heating is not None and cooling is not None,
        "manual_review_recommended": True,
    }
    return DSCAnalysisResult(curve, events, temperatures, confidence=confidence)
