"""2H martensite literature metadata."""

from phases._base import PhaseReference

MARTENSITE_2H = PhaseReference(
    key="2H",
    display_name="2H γ′ martensite",
    temperature_role="low-temperature martensitic product",
    crystallographic_description="Two-layer martensitic stacking described in the cited Cu-Al-Ni electron-microscopy study.",
    space_group="Pnmm (reported setting)",
    stacking_sequence="2H",
    formula=None,
    source="Xie et al. (2006), Journal of Alloys and Compounds 417, 170-175",
    source_url="https://doi.org/10.1016/j.jallcom.2005.09.049",
    limitations=(
        "The cited phase assignment concerns Cu-11.92Al-3.78Ni wt%.",
        "This registry does not promote incomplete literature metadata to a diffraction structure.",
    ),
)


def get_phase_reference() -> PhaseReference:
    return MARTENSITE_2H

