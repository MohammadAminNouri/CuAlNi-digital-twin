"""Composition-only descriptors with explicit definitional conventions."""

from __future__ import annotations

from math import isfinite
from typing import Mapping

from composition.converter import ELEMENTS, AlloyComposition
from scientific import Provenance, ResultClassification, ScientificMetadata, ScientificValue

# Group numbers are an explicit, reproducible VEC convention; this must not be
# confused with a fitted Hume-Rothery effective electron count.
GROUP_NUMBER_VALENCE = {"Cu": 11.0, "Al": 3.0, "Ni": 10.0}

# Approximate room-temperature elemental bulk densities from the Royal Society
# of Chemistry element data pages. The resulting reciprocal rule-of-mixtures
# value is only an ideal-mixture screening estimate, not an alloy measurement.
ROOM_TEMPERATURE_ELEMENT_DENSITY_G_CM3 = {"Cu": 8.96, "Al": 2.70, "Ni": 8.90}
RSC_DENSITY_PROVENANCE = Provenance(
    title="Royal Society of Chemistry Periodic Table",
    citation="RSC element data pages for Cu, Al and Ni; elemental densities near room temperature",
    url="https://www.rsc.org/periodic-table",
)


def electron_concentration(
    composition: AlloyComposition,
    valence_map: Mapping[str, float],
    *,
    convention_name: str,
    source: Provenance | None = None,
) -> ScientificValue:
    """Calculate an average electron count for an explicitly supplied convention.

    There is no universal effective valence assignment for transition-metal
    Hume-Rothery e/a. Requiring a complete map prevents an arbitrary convention
    from being silently presented as a measured electronic property.
    """

    missing = set(ELEMENTS) - set(valence_map)
    if missing:
        raise ValueError(f"valence_map is missing: {', '.join(sorted(missing))}")
    values = {element: float(valence_map[element]) for element in ELEMENTS}
    if not all(isfinite(value) for value in values.values()):
        raise ValueError("Effective valence values must be finite.")
    value = sum(composition.mole_fraction[element] * values[element] for element in ELEMENTS)
    provenance = (source,) if source is not None else ()
    return ScientificValue(
        name="electron_concentration",
        value=value,
        unit="electrons/atom",
        details={"convention": convention_name, "valence_map": values},
        metadata=ScientificMetadata(
            classification=ResultClassification.PHYSICS_CALCULATED,
            confidence="high for the stated arithmetic convention; not an experimental band count",
            limitations=(
                "The value depends on the selected valence convention.",
                "It is not a direct electronic-structure calculation.",
            ),
            provenance=provenance,
        ),
    )


def valence_electron_concentration(composition: AlloyComposition) -> ScientificValue:
    """Return VEC under the explicit periodic-table group-number convention."""

    return electron_concentration(
        composition,
        GROUP_NUMBER_VALENCE,
        convention_name="periodic-table group number",
        source=Provenance(
            title="IUPAC Periodic Table",
            citation="IUPAC periodic-table group assignments",
            url="https://iupac.org/what-we-do/periodic-table-of-elements/",
        ),
    )


def unavailable_hume_rothery_e_a() -> ScientificValue:
    """Represent intentionally unavailable e/a when no convention is provided."""

    return ScientificValue(
        name="hume_rothery_e_a",
        value=None,
        unit="electrons/atom",
        metadata=ScientificMetadata(
            classification=ResultClassification.NOT_AVAILABLE,
            confidence="not calculated",
            limitations=(
                "A transition-metal effective-valence convention must be selected and cited before e/a is calculated.",
            ),
        ),
    )


def estimate_density(
    composition: AlloyComposition,
    elemental_densities_g_cm3: Mapping[str, float] | None = None,
) -> ScientificValue:
    """Estimate ideal-mixture density using ``rho = 1 / sum(w_i/rho_i)``."""

    densities = dict(elemental_densities_g_cm3 or ROOM_TEMPERATURE_ELEMENT_DENSITY_G_CM3)
    missing = set(ELEMENTS) - set(densities)
    if missing:
        raise ValueError(f"elemental_densities_g_cm3 is missing: {', '.join(sorted(missing))}")
    if any(float(densities[element]) <= 0 for element in ELEMENTS):
        raise ValueError("All elemental densities must be positive.")
    density = 1.0 / sum(
        composition.weight_fraction[element] / float(densities[element]) for element in ELEMENTS
    )
    using_reference = elemental_densities_g_cm3 is None
    provenance = (RSC_DENSITY_PROVENANCE,) if using_reference else ()
    return ScientificValue(
        name="ideal_mixture_density",
        value=density,
        unit="g/cm^3",
        details={"mixing_rule": "inverse mass-fraction rule", "elemental_densities": densities},
        metadata=ScientificMetadata(
            classification=ResultClassification.ENGINEERING_SCREENING,
            confidence="low-to-moderate (ideal mixture estimate)",
            limitations=(
                "Ignores excess volume, phase constitution, porosity, defects, and temperature.",
                "Do not substitute this estimate for measured or CALPHAD-derived density.",
            ),
            provenance=provenance,
        ),
    )


def compute_descriptors(
    composition: AlloyComposition,
    *,
    e_a_valence_map: Mapping[str, float] | None = None,
    e_a_convention_name: str = "user supplied",
    e_a_source: Provenance | None = None,
) -> dict[str, ScientificValue]:
    """Compute reproducible composition descriptors without hidden fitted values."""

    e_a = (
        unavailable_hume_rothery_e_a()
        if e_a_valence_map is None
        else electron_concentration(
            composition,
            e_a_valence_map,
            convention_name=e_a_convention_name,
            source=e_a_source,
        )
    )
    return {
        "hume_rothery_e_a": e_a,
        "valence_electron_concentration": valence_electron_concentration(composition),
        "density_estimate": estimate_density(composition),
    }
