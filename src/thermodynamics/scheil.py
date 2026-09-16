"""Adapter to the official open-source ``pycalphad/scheil`` package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from composition.converter import AlloyComposition
from scientific import Provenance, ResultClassification, ScientificMetadata
from thermodynamics.calphad import CalphadEngine, DatabaseCoverageError

SCHEIL_REFERENCE = Provenance(
    title="pycalphad/scheil",
    citation="pycalphad contributors, scheil: a Scheil-Gulliver simulation tool using pycalphad",
    url="https://github.com/pycalphad/scheil",
)


@dataclass
class ScheilResult:
    phase_fractions: pd.DataFrame
    liquid_composition: Mapping[str, Sequence[float]]
    raw_result: Any
    metadata: ScientificMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_fractions": self.phase_fractions.to_dict(orient="records"),
            "liquid_composition": {key: list(value) for key, value in self.liquid_composition.items()},
            "metadata": self.metadata.to_dict(),
        }


def run_scheil(
    engine: CalphadEngine,
    composition: AlloyComposition,
    *,
    start_temperature_K: float,
    step_temperature_K: float = 1.0,
    phases: Sequence[str] | None = None,
    liquid_phase_name: str = "LIQUID",
    stop_liquid_fraction: float = 1e-4,
    adaptive: bool = True,
) -> ScheilResult:
    """Run Scheil-Gulliver solidification using a validated external database."""

    try:
        from scheil import simulate_scheil_solidification
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "The optional 'scheil' package is required for Scheil-Gulliver simulation."
        ) from exc
    start = float(start_temperature_K)
    step = float(step_temperature_K)
    stop = float(stop_liquid_fraction)
    if not np.isfinite(start) or start <= 0:
        raise ValueError("start_temperature_K must be a finite absolute temperature above 0 K.")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("step_temperature_K must be finite and positive.")
    if not (0 < stop < 1):
        raise ValueError("stop_liquid_fraction must lie strictly between 0 and 1.")
    selected = engine._select_phases(phases)
    if liquid_phase_name not in selected:
        raise DatabaseCoverageError(
            f"Liquid phase {liquid_phase_name!r} is not available in the selected database phases."
        )
    conditions = engine.composition_conditions(composition)
    raw = simulate_scheil_solidification(
        engine.dbf,
        engine._components(),
        selected,
        conditions,
        start,
        step_temperature=step,
        liquid_phase_name=liquid_phase_name,
        stop=stop,
        adaptive=bool(adaptive),
    )
    temperatures = np.asarray(raw.temperatures, dtype=float)
    liquid = np.asarray(raw.fraction_liquid, dtype=float)
    phase_series = {
        str(phase): np.asarray(values, dtype=float)
        for phase, values in raw.cum_phase_amounts.items()
    }
    rows: list[dict[str, float | str]] = []
    for index, temperature in enumerate(temperatures):
        rows.append(
            {
                "temperature_K": float(temperature),
                "temperature_C": float(temperature - 273.15),
                "phase": liquid_phase_name,
                "cumulative_molar_fraction": float(liquid[index]),
            }
        )
        for phase, values in phase_series.items():
            rows.append(
                {
                    "temperature_K": float(temperature),
                    "temperature_C": float(temperature - 273.15),
                    "phase": phase,
                    "cumulative_molar_fraction": float(values[index]),
                }
            )
    liquid_composition_raw = getattr(raw, "x_liquid", {})
    liquid_composition = {
        str(component): np.asarray(values, dtype=float).tolist()
        for component, values in liquid_composition_raw.items()
    }
    return ScheilResult(
        phase_fractions=pd.DataFrame(rows),
        liquid_composition=liquid_composition,
        raw_result=raw,
        metadata=ScientificMetadata(
            classification=ResultClassification.PHYSICS_CALCULATED,
            confidence="database- and model-dependent",
            limitations=(
                "Assumes perfect liquid mixing, local solid/liquid equilibrium, and no solid-state diffusion.",
                "Rapid-solidification solute trapping, melt convection, nucleation barriers, and solid transformations are omitted.",
                "The starting temperature must be inside the single-liquid region of the supplied database.",
                "Miscibility-gap phase multiplicities are not supported by the underlying package.",
            ),
            provenance=(SCHEIL_REFERENCE,),
        ),
    )
