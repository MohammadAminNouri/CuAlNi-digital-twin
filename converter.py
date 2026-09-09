"""Mass/atomic/mole conversion for the closed Cu-Al-Ni system.

Atomic weights are the conventional values published by the CIAAW.  Atomic
percent and mole percent are numerically identical for an elemental alloy.
No thermodynamic assumptions enter these conversions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from scientific import Provenance, ResultClassification, ScientificMetadata

ELEMENTS = ("Cu", "Al", "Ni")
ATOMIC_WEIGHTS_G_MOL = {
    "Cu": 63.546,
    "Al": 26.9815384,
    "Ni": 58.6934,
}
ATOMIC_WEIGHT_PROVENANCE = Provenance(
    title="CIAAW Standard Atomic Weights",
    citation="Commission on Isotopic Abundances and Atomic Weights, standard atomic weights",
    url="https://ciaaw.org/atomic-weights.htm",
)


class CompositionError(ValueError):
    """Raised when an alloy composition is malformed or outside this system."""


def _canonical_element(name: str) -> str:
    cleaned = str(name).strip().capitalize()
    if cleaned not in ELEMENTS:
        raise CompositionError(
            f"Unsupported element {name!r}; this platform accepts only Cu, Al, and Ni."
        )
    return cleaned


def _validate_and_normalize(values: Mapping[str, float]) -> tuple[dict[str, float], float]:
    canonical: dict[str, float] = {element: 0.0 for element in ELEMENTS}
    seen: set[str] = set()
    for raw_name, raw_value in values.items():
        element = _canonical_element(raw_name)
        if element in seen:
            raise CompositionError(f"Element {element} was supplied more than once.")
        seen.add(element)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise CompositionError(f"Composition for {element} must be numeric.") from exc
        if not isfinite(value) or value < 0:
            raise CompositionError(f"Composition for {element} must be finite and non-negative.")
        canonical[element] = value
    total = sum(canonical.values())
    if total <= 0:
        raise CompositionError("At least one composition value must be greater than zero.")
    return ({element: value / total for element, value in canonical.items()}, total)


@dataclass(frozen=True)
class AlloyComposition:
    """Normalized Cu-Al-Ni composition represented on both mass and mole bases."""

    weight_fraction: Mapping[str, float]
    mole_fraction: Mapping[str, float]
    input_basis: str
    input_total: float
    metadata: ScientificMetadata

    @classmethod
    def from_wt_percent(cls, values: Mapping[str, float]) -> "AlloyComposition":
        weights, raw_total = _validate_and_normalize(values)
        mole_amounts = {
            element: weights[element] / ATOMIC_WEIGHTS_G_MOL[element] for element in ELEMENTS
        }
        mole_total = sum(mole_amounts.values())
        mole_fractions = {element: amount / mole_total for element, amount in mole_amounts.items()}
        limitations: tuple[str, ...] = ()
        if abs(raw_total - 100.0) > 1e-8:
            limitations = (
                f"Input values summed to {raw_total:g}, so they were normalized to 100 wt%.",
            )
        return cls(
            weight_fraction=weights,
            mole_fraction=mole_fractions,
            input_basis="wt_percent",
            input_total=raw_total,
            metadata=ScientificMetadata(
                classification=ResultClassification.PHYSICS_CALCULATED,
                confidence="high (deterministic unit conversion)",
                limitations=limitations,
                provenance=(ATOMIC_WEIGHT_PROVENANCE,),
            ),
        )

    @classmethod
    def from_atomic_percent(cls, values: Mapping[str, float]) -> "AlloyComposition":
        moles, raw_total = _validate_and_normalize(values)
        masses = {element: moles[element] * ATOMIC_WEIGHTS_G_MOL[element] for element in ELEMENTS}
        mass_total = sum(masses.values())
        weights = {element: mass / mass_total for element, mass in masses.items()}
        limitations: tuple[str, ...] = ()
        if abs(raw_total - 100.0) > 1e-8:
            limitations = (
                f"Input values summed to {raw_total:g}, so they were normalized to 100 at%.",
            )
        return cls(
            weight_fraction=weights,
            mole_fraction=moles,
            input_basis="atomic_percent",
            input_total=raw_total,
            metadata=ScientificMetadata(
                classification=ResultClassification.PHYSICS_CALCULATED,
                confidence="high (deterministic unit conversion)",
                limitations=limitations,
                provenance=(ATOMIC_WEIGHT_PROVENANCE,),
            ),
        )

    @classmethod
    def from_mole_fraction(cls, values: Mapping[str, float]) -> "AlloyComposition":
        composition = cls.from_atomic_percent(values)
        return cls(
            weight_fraction=composition.weight_fraction,
            mole_fraction=composition.mole_fraction,
            input_basis="mole_fraction",
            input_total=composition.input_total,
            metadata=composition.metadata,
        )

    @property
    def weight_percent(self) -> dict[str, float]:
        return {element: 100.0 * self.weight_fraction[element] for element in ELEMENTS}

    @property
    def atomic_percent(self) -> dict[str, float]:
        return {element: 100.0 * self.mole_fraction[element] for element in ELEMENTS}

    @property
    def dependent_element(self) -> str:
        """The largest mole-fraction element, useful as CALPHAD dependent component."""

        return max(ELEMENTS, key=lambda element: self.mole_fraction[element])

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_percent": self.weight_percent,
            "atomic_percent": self.atomic_percent,
            "mole_fraction": dict(self.mole_fraction),
            "input_basis": self.input_basis,
            "input_total": self.input_total,
            "metadata": self.metadata.to_dict(),
        }


def convert_wt_to_atomic(values: Mapping[str, float]) -> dict[str, float]:
    """Return atomic percent for a Cu-Al-Ni weight-percent mapping."""

    return AlloyComposition.from_wt_percent(values).atomic_percent


def convert_atomic_to_wt(values: Mapping[str, float]) -> dict[str, float]:
    """Return weight percent for a Cu-Al-Ni atomic-percent mapping."""

    return AlloyComposition.from_atomic_percent(values).weight_percent
