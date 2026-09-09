"""Database-backed ternary and isopleth sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from composition.converter import AlloyComposition
from scientific import ResultClassification, ScientificMetadata
from thermodynamics.calphad import CalphadEngine


@dataclass
class PhaseMapResult:
    points: pd.DataFrame
    temperature_K: float
    database_sha256: str
    metadata: ScientificMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points.to_dict(orient="records"),
            "temperature_K": self.temperature_K,
            "database_sha256": self.database_sha256,
            "metadata": self.metadata.to_dict(),
        }


def ternary_isothermal_phase_map(
    engine: CalphadEngine,
    *,
    temperature_K: float,
    divisions: int = 20,
    pressure_Pa: float = 101_325.0,
    phases: Sequence[str] | None = None,
    display_fraction_cutoff: float = 1e-6,
) -> PhaseMapResult:
    """Sample stable phase assemblages on a regular ternary mole-fraction grid.

    The cutoff changes only displayed phase labels; raw fractions are retained
    in the ``phase_fractions`` mapping at every point.
    """

    if not isinstance(divisions, int) or not 2 <= divisions <= 100:
        raise ValueError("divisions must be an integer from 2 to 100.")
    if not 0 <= display_fraction_cutoff < 1:
        raise ValueError("display_fraction_cutoff must be in [0, 1).")
    rows: list[dict[str, Any]] = []
    for i_cu in range(divisions + 1):
        for i_al in range(divisions + 1 - i_cu):
            i_ni = divisions - i_cu - i_al
            mole = {
                "Cu": i_cu / divisions,
                "Al": i_al / divisions,
                "Ni": i_ni / divisions,
            }
            composition = AlloyComposition.from_mole_fraction(mole)
            result = engine.equilibrium(
                composition,
                temperature_K,
                pressure_Pa=pressure_Pa,
                phases=phases,
                dependent_component=composition.dependent_element.upper(),
            )
            fractions = {
                str(row.phase): float(row.molar_phase_fraction)
                for row in result.phase_fractions.itertuples()
            }
            visible = sorted(
                phase for phase, fraction in fractions.items() if fraction >= display_fraction_cutoff
            )
            rows.append(
                {
                    "x_Cu": mole["Cu"],
                    "x_Al": mole["Al"],
                    "x_Ni": mole["Ni"],
                    "phase_assemblage": " + ".join(visible) if visible else "unresolved",
                    "phase_fractions": fractions,
                }
            )
    return PhaseMapResult(
        points=pd.DataFrame(rows),
        temperature_K=float(temperature_K),
        database_sha256=engine.info.sha256,
        metadata=ScientificMetadata(
            classification=ResultClassification.PHYSICS_CALCULATED,
            confidence="database-dependent grid sampling",
            limitations=(
                "Grid resolution can miss narrow phase fields and does not trace tie-lines.",
                "Displayed assemblages suppress fractions below the user-selected visualization cutoff.",
                "Use a dedicated pycalphad mapping strategy for publication-quality phase boundaries.",
            ),
            provenance=(),
        ),
    )
