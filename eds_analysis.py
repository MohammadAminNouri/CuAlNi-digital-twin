"""EDS composition statistics with uncertainty and segregation diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import numpy as np
import pandas as pd

ATOMIC_WEIGHTS_G_MOL = {"Cu": 63.546, "Al": 26.9815385, "Ni": 58.6934}
ATOMIC_NUMBERS = {"Cu": 29, "Al": 13, "Ni": 28}


class EDSAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class ElementStatistics:
    element: str
    mean_wt_pct: float
    sample_sd_wt_pct: float | None
    standard_error_wt_pct: float | None
    min_wt_pct: float
    max_wt_pct: float
    coefficient_of_variation: float | None
    target_deviation_wt_pct: float | None


@dataclass
class EDSAnalysisResult:
    normalized_points_wt_pct: pd.DataFrame
    statistics: list[ElementStatistics]
    segregation_metrics: dict[str, Any]
    interaction_volume: dict[str, Any] | None
    classification: str = "experimental_data_analysis"
    confidence: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = (
        "EDS quantification depends on standards, matrix corrections, detector calibration, and surface condition.",
        "The interaction volume averages sub-surface material and can mix adjacent microstructural regions.",
        "Point-to-point variation can combine true microsegregation with counting and preparation error.",
        "Area statistics are not bulk composition or equilibrium phase fractions.",
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "statistics": [asdict(item) for item in self.statistics],
            "segregation_metrics": self.segregation_metrics,
            "interaction_volume": self.interaction_volume,
            "confidence": self.confidence,
            "limitations": list(self.limitations),
        }


def _read_eds(data: str | Path | BinaryIO | pd.DataFrame) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    try:
        return pd.read_csv(data, sep=None, engine="python", comment="#")
    except Exception as exc:
        raise EDSAnalysisError(f"Could not parse EDS table: {exc}") from exc


def _element_column(frame: pd.DataFrame, element: str, basis: str) -> str | None:
    aliases = {
        f"{element.lower()}_{basis}",
        f"{element.lower()}_{basis}_pct",
        f"{element.lower()}{basis}",
        f"{element.lower()} ({basis}%)",
        f"{element.lower()}_{'weight' if basis == 'wt' else 'atomic'}_percent",
    }
    for column in frame.columns:
        normalized = str(column).strip().lower().replace("%", "").replace(" ", "_")
        normalized = normalized.replace("__", "_")
        if normalized in {item.replace(" ", "_").replace("%", "") for item in aliases}:
            return column
    return None


def atomic_to_weight_percent(atomic_percent: pd.DataFrame) -> pd.DataFrame:
    weighted = atomic_percent.copy()
    for element in ATOMIC_WEIGHTS_G_MOL:
        weighted[element] = weighted[element] * ATOMIC_WEIGHTS_G_MOL[element]
    denominator = weighted.sum(axis=1)
    if (denominator <= 0).any():
        raise EDSAnalysisError("Atomic-percent rows must contain a positive total")
    return weighted.div(denominator, axis=0) * 100.0


def normalize_eds_composition(frame: pd.DataFrame, *, basis: str = "auto") -> pd.DataFrame:
    if basis not in {"auto", "wt", "at"}:
        raise EDSAnalysisError("basis must be 'auto', 'wt', or 'at'")
    wt_columns = {element: _element_column(frame, element, "wt") for element in ATOMIC_WEIGHTS_G_MOL}
    at_columns = {element: _element_column(frame, element, "at") for element in ATOMIC_WEIGHTS_G_MOL}
    if basis == "auto":
        basis = "wt" if all(wt_columns.values()) else "at" if all(at_columns.values()) else ""
    selected = wt_columns if basis == "wt" else at_columns
    if not basis or not all(selected.values()):
        raise EDSAnalysisError("Need Cu, Al, and Ni columns consistently labelled wt% or at%")
    composition = pd.DataFrame(
        {
            element: pd.to_numeric(frame[column], errors="coerce")
            for element, column in selected.items()
        }
    ).dropna()
    if composition.empty:
        raise EDSAnalysisError("No complete numeric EDS rows were found")
    if (composition < 0).any(axis=None):
        raise EDSAnalysisError("EDS composition cannot be negative")
    if basis == "at":
        composition = atomic_to_weight_percent(composition)
    totals = composition.sum(axis=1)
    if (totals <= 0).any():
        raise EDSAnalysisError("EDS rows must have positive composition totals")
    return composition.div(totals, axis=0) * 100.0


def kanaya_okayama_interaction_range_um(
    composition_wt_pct: Mapping[str, float], *, accelerating_voltage_kV: float, density_g_cm3: float
) -> float:
    """Kanaya-Okayama electron range estimate for a composition-weighted material.

    Citation: Kanaya and Okayama, J. Phys. D 5 (1972) 43,
    https://doi.org/10.1088/0022-3727/5/1/308.
    """

    if accelerating_voltage_kV <= 0 or density_g_cm3 <= 0:
        raise EDSAnalysisError("Accelerating voltage and density must be positive")
    wt = np.array([composition_wt_pct[element] for element in ATOMIC_WEIGHTS_G_MOL], dtype=float)
    wt = wt / wt.sum()
    mole = wt / np.array(list(ATOMIC_WEIGHTS_G_MOL.values()))
    atomic_fraction = mole / mole.sum()
    mean_a = float(np.dot(atomic_fraction, list(ATOMIC_WEIGHTS_G_MOL.values())))
    mean_z = float(np.dot(atomic_fraction, list(ATOMIC_NUMBERS.values())))
    return float(0.0276 * mean_a * accelerating_voltage_kV**1.67 / (density_g_cm3 * mean_z**0.89))


def analyze_eds(
    data: str | Path | BinaryIO | pd.DataFrame,
    target_wt_pct: Mapping[str, float] | None = None,
    *,
    basis: str = "auto",
    accelerating_voltage_kV: float | None = None,
    density_g_cm3: float | None = None,
    calibration_reference: str | None = None,
) -> EDSAnalysisResult:
    raw = _read_eds(data)
    composition = normalize_eds_composition(raw, basis=basis)
    if target_wt_pct is not None:
        if set(target_wt_pct) != set(ATOMIC_WEIGHTS_G_MOL):
            raise EDSAnalysisError("Target must contain exactly Cu, Al, and Ni")
        if not np.isclose(sum(target_wt_pct.values()), 100.0, atol=0.5):
            raise EDSAnalysisError("Target composition must total 100 +/- 0.5 wt%")
    statistics: list[ElementStatistics] = []
    segregation: dict[str, Any] = {}
    for element in ATOMIC_WEIGHTS_G_MOL:
        values = composition[element].to_numpy(dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
        sem = sd / np.sqrt(len(values)) if sd is not None else None
        cv = sd / mean if sd is not None and mean != 0 else None
        target_deviation = mean - float(target_wt_pct[element]) if target_wt_pct else None
        statistics.append(
            ElementStatistics(
                element,
                mean,
                sd,
                sem,
                float(np.min(values)),
                float(np.max(values)),
                cv,
                target_deviation,
            )
        )
        minimum = float(np.min(values))
        segregation[element] = {
            "range_wt_pct": float(np.ptp(values)),
            "coefficient_of_variation": cv,
            "max_to_min_ratio": float(np.max(values) / minimum) if minimum > 0 else None,
        }
    interaction = None
    if accelerating_voltage_kV is not None and density_g_cm3 is not None:
        means = {item.element: item.mean_wt_pct for item in statistics}
        interaction = {
            "range_um": kanaya_okayama_interaction_range_um(
                means,
                accelerating_voltage_kV=accelerating_voltage_kV,
                density_g_cm3=density_g_cm3,
            ),
            "model": "Kanaya-Okayama electron range estimate",
            "doi": "10.1088/0022-3727/5/1/308",
            "inputs": {
                "accelerating_voltage_kV": accelerating_voltage_kV,
                "density_g_cm3": density_g_cm3,
            },
            "limitation": "Electron range is not the same as lateral X-ray spatial resolution.",
        }
    confidence = {
        "n_points": int(len(composition)),
        "replicate_standard_errors_available": len(composition) > 1,
        "calibration_reference": calibration_reference,
        "calibration_documented": bool(calibration_reference),
    }
    return EDSAnalysisResult(composition, statistics, segregation, interaction, confidence=confidence)
