"""18R martensite literature metadata."""

from phases._base import PhaseReference

MARTENSITE_18R = PhaseReference(
    key="18R",
    display_name="18R β₁′ martensite",
    temperature_role="low-temperature martensitic product",
    crystallographic_description="Long-period close-packed martensite; reported setting may be monoclinic or orthorhombic depending on model/convention.",
    space_group=None,
    stacking_sequence="AB'CB'CA'CA'BA'BC'BC'AC'AB'",
    formula=None,
    source="Otsuka, Nakamura & Shimizu (1974), Trans. JIM 15, 200-210",
    source_url="https://doi.org/10.2320/matertrans1960.15.200",
    limitations=(
        "The cited cell/stacking data concern a specific Cu-14.2Al-4.3Ni wt% specimen.",
        "A full CIF is required; missing angles or atom sites are never inferred.",
    ),
)


def get_phase_reference() -> PhaseReference:
    return MARTENSITE_18R

