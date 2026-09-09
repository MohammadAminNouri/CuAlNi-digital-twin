"""Ordered beta1 parent reference."""

from phases._base import PhaseReference

BETA1_PHASE = PhaseReference(
    key="beta1",
    display_name="β₁ ordered parent",
    temperature_role="ordered parent of martensitic transformations",
    crystallographic_description="Ordered BCC-related parent commonly described with DO3/L21-type ordering in Cu-Al-Ni literature.",
    space_group=None,
    stacking_sequence=None,
    formula=None,
    source="Otsuka, Nakamura & Shimizu (1974), Trans. JIM 15, 200-210",
    source_url="https://doi.org/10.2320/matertrans1960.15.200",
    limitations=(
        "Ordering state and site occupancy depend on composition and thermal history.",
        "No universal lattice parameter or atomic basis is assigned here.",
    ),
)


def get_phase_reference() -> PhaseReference:
    return BETA1_PHASE

