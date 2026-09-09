"""Curated Cu-Al-Ni phase reference registry."""

from phases.beta1_phase import BETA1_PHASE
from phases.beta_phase import BETA_PHASE
from phases.gamma2_phase import GAMMA2_PHASE
from phases.martensite_2H import MARTENSITE_2H
from phases.martensite_18R import MARTENSITE_18R


def phase_reference_catalog() -> dict[str, object]:
    """Return phase descriptions, never equilibrium predictions."""

    references = (BETA_PHASE, BETA1_PHASE, MARTENSITE_18R, MARTENSITE_2H, GAMMA2_PHASE)
    return {
        "status": "reference_only",
        "classification": "physics_calculated",
        "confidence": "phase descriptions only; no specimen phase assignment",
        "phases": {reference.key: reference.to_dict() for reference in references},
        "limitations": [
            "This catalog documents candidate phases; it does not predict stability or fraction.",
            "Use a validated CALPHAD database and/or experimental characterization for phase assignment.",
        ],
    }


__all__ = [
    "BETA_PHASE",
    "BETA1_PHASE",
    "MARTENSITE_18R",
    "MARTENSITE_2H",
    "GAMMA2_PHASE",
    "phase_reference_catalog",
]
