"""Convenience APIs for equilibrium phase-fraction analyses."""

from __future__ import annotations

from typing import Iterable, Sequence

from composition.converter import AlloyComposition
from thermodynamics.calphad import CalphadEngine, EquilibriumResult


def phase_fractions_vs_temperature(
    engine: CalphadEngine,
    composition: AlloyComposition,
    temperatures_K: Iterable[float],
    *,
    pressure_Pa: float = 101_325.0,
    phases: Sequence[str] | None = None,
) -> EquilibriumResult:
    """Return database-calculated molar phase fractions along a temperature path."""

    return engine.equilibrium(
        composition,
        temperatures_K,
        pressure_Pa=pressure_Pa,
        phases=phases,
    )


def equilibrium_at_temperature(
    engine: CalphadEngine,
    composition: AlloyComposition,
    temperature_K: float,
    *,
    pressure_Pa: float = 101_325.0,
    phases: Sequence[str] | None = None,
) -> EquilibriumResult:
    """Return a single-temperature equilibrium result."""

    return engine.equilibrium(
        composition,
        temperature_K,
        pressure_Pa=pressure_Pa,
        phases=phases,
    )
