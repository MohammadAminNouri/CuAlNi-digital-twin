"""Gamma2 Cu9Al4 reference metadata."""

from phases._base import PhaseReference

GAMMA2_PHASE = PhaseReference(
    key="gamma2",
    display_name="γ₂ Cu₉Al₄",
    temperature_role="secondary/intermetallic phase",
    crystallographic_description="Gamma-brass D8_3 prototype, Pearson cP52.",
    space_group="P-43m (No. 215)",
    stacking_sequence=None,
    formula="Cu9Al4",
    source="AFLOW prototype A4B9_cP52_215_ei_3efgi-001; cited refinement DOI 10.1107/S0567739478000807",
    source_url="https://aflow.org/p/A4B9_cP52_215_ei_3efgi-001",
    limitations=(
        "Prototype identity alone does not provide the specimen-specific lattice/internal coordinates used by this app.",
        "Presence and fraction must be calculated with a suitable TDB or measured experimentally.",
    ),
)


def get_phase_reference() -> PhaseReference:
    return GAMMA2_PHASE

