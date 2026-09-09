"""Source-defined composition-window screening.

No built-in empirical phase boundaries are included: ternary phase boundaries
are temperature- and database-dependent.  Users may register a cited window,
or use the CALPHAD modules for database-backed equilibrium results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from composition.converter import ELEMENTS, AlloyComposition
from scientific import Provenance, ResultClassification, ScientificMetadata


@dataclass(frozen=True)
class CompositionWindow:
    name: str
    basis: str
    minima: Mapping[str, float]
    maxima: Mapping[str, float]
    provenance: Provenance
    temperature_K: float | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.basis not in {"wt_percent", "atomic_percent"}:
            raise ValueError("basis must be 'wt_percent' or 'atomic_percent'")
        for element in ELEMENTS:
            if element not in self.minima or element not in self.maxima:
                raise ValueError(f"Window bounds must include {element}.")
            if float(self.minima[element]) > float(self.maxima[element]):
                raise ValueError(f"Minimum exceeds maximum for {element}.")


@dataclass(frozen=True)
class WindowAssessment:
    window_name: str
    inside: bool
    signed_margin_percent: Mapping[str, float]
    metadata: ScientificMetadata


def assess_against_window(
    composition: AlloyComposition, window: CompositionWindow
) -> WindowAssessment:
    """Test a composition against a fully source-defined rectangular window."""

    values = composition.weight_percent if window.basis == "wt_percent" else composition.atomic_percent
    margins: dict[str, float] = {}
    inside = True
    for element in ELEMENTS:
        lower = float(window.minima[element])
        upper = float(window.maxima[element])
        value = values[element]
        margin = min(value - lower, upper - value)
        margins[element] = margin
        inside &= lower <= value <= upper
    limitations = [
        "A rectangular composition window is a screening construct, not a phase-equilibrium calculation."
    ]
    if window.temperature_K is None:
        limitations.append("The supplied window has no temperature attached.")
    return WindowAssessment(
        window_name=window.name,
        inside=inside,
        signed_margin_percent=margins,
        metadata=ScientificMetadata(
            classification=ResultClassification.ENGINEERING_SCREENING,
            confidence="limited by the supplied literature window",
            limitations=tuple(limitations),
            provenance=(window.provenance,),
        ),
    )
