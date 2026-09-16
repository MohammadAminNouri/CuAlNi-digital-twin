"""Disordered high-temperature beta-parent reference."""

from phases._base import PhaseReference

BETA_PHASE = PhaseReference(
    key="beta",
    display_name="β parent phase",
    temperature_role="high-temperature parent phase",
    crystallographic_description="BCC-related disordered parent; the exact database phase/model name is TDB-dependent.",
    space_group=None,
    stacking_sequence=None,
    formula=None,
    source="Cu-Al-Ni CALPHAD assessment: Miettinen (2005), CALPHAD 29, 40-48",
    source_url="https://doi.org/10.1016/j.calphad.2005.02.002",
    limitations=(
        "This description is not a quantitative stability field.",
        "A specimen-appropriate CIF and a compatible TDB are required for XRD and equilibrium fractions.",
    ),
)


def get_phase_reference() -> PhaseReference:
    return BETA_PHASE

