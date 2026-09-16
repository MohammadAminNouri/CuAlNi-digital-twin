"""Structure-factor powder XRD simulation through pymatgen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from crystallography.crystal_database import CrystalStructureRecord, coerce_structure
from scientific import Provenance, ResultClassification, ScientificMetadata


@dataclass
class XRDPattern:
    phase: str
    peaks: pd.DataFrame
    wavelength_angstrom: float
    radiation: str
    two_theta_range_deg: tuple[float, float]
    metadata: ScientificMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "peaks": self.peaks.to_dict(orient="records"),
            "wavelength_angstrom": self.wavelength_angstrom,
            "radiation": self.radiation,
            "two_theta_range_deg": list(self.two_theta_range_deg),
            "metadata": self.metadata.to_dict(),
        }


def simulate_powder_pattern(
    structure: Any | CrystalStructureRecord | Mapping[str, Any],
    *,
    phase_name: str | None = None,
    radiation: str | float = "CuKa",
    wavelength_angstrom: float | None = None,
    two_theta_range_deg: tuple[float, float] = (10.0, 100.0),
    scaled_intensity_cutoff: float = 0.0,
) -> XRDPattern:
    """Calculate ideal kinematic powder peaks from a complete periodic structure.

    ``radiation`` may be a pymatgen radiation key (e.g. ``"CuKa"``) or an
    explicit wavelength in ångström. ``wavelength_angstrom`` is an explicit,
    report-friendly alias; it cannot be combined with a non-default
    ``radiation`` value. Pymatgen's resolved wavelength is returned rather than
    duplicated as a hidden project constant.
    """

    try:
        from pymatgen.analysis.diffraction.xrd import XRDCalculator
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pymatgen is required for structure-factor XRD simulation.") from exc
    if isinstance(structure, CrystalStructureRecord):
        record = structure
        crystal = coerce_structure(record)
        inherited_provenance = record.metadata.provenance
        inherited_limitations = record.metadata.limitations
        default_name = record.name
    else:
        crystal = coerce_structure(structure)
        inherited_provenance = ()
        inherited_limitations = ()
        default_name = getattr(getattr(crystal, "composition", None), "reduced_formula", "structure")
    if wavelength_angstrom is not None:
        wavelength = float(wavelength_angstrom)
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("wavelength_angstrom must be finite and positive.")
        if radiation != "CuKa":
            raise ValueError("Specify either radiation or wavelength_angstrom, not both.")
        radiation = wavelength
    lower, upper = map(float, two_theta_range_deg)
    if not (0 <= lower < upper <= 180):
        raise ValueError("two_theta_range_deg must satisfy 0 <= lower < upper <= 180.")
    cutoff = float(scaled_intensity_cutoff)
    if not 0 <= cutoff <= 100:
        raise ValueError("scaled_intensity_cutoff must be in [0, 100].")
    try:
        calculator = XRDCalculator(wavelength=radiation)
    except Exception as exc:
        raise ValueError(f"Invalid diffraction radiation/wavelength {radiation!r}: {exc}") from exc
    pattern = calculator.get_pattern(crystal, two_theta_range=(lower, upper), scaled=True)
    rows: list[dict[str, Any]] = []
    for angle, intensity, families, spacing in zip(
        pattern.x, pattern.y, pattern.hkls, pattern.d_hkls, strict=True
    ):
        if float(intensity) < cutoff:
            continue
        hkls = [tuple(int(index) for index in family["hkl"]) for family in families]
        multiplicities = [int(family.get("multiplicity", 1)) for family in families]
        rows.append(
            {
                "two_theta_deg": float(angle),
                "scaled_intensity": float(intensity),
                "d_angstrom": float(spacing),
                "hkl": "; ".join(str(hkl) for hkl in hkls),
                "hkl_families": hkls,
                "multiplicity": int(sum(multiplicities)),
                "phase": phase_name or str(default_name),
            }
        )
    wavelength = float(calculator.wavelength)
    radiation_label = str(radiation) if isinstance(radiation, str) else "user wavelength"
    return XRDPattern(
        phase=phase_name or str(default_name),
        peaks=pd.DataFrame(
            rows,
            columns=[
                "two_theta_deg",
                "scaled_intensity",
                "d_angstrom",
                "hkl",
                "hkl_families",
                "multiplicity",
                "phase",
            ],
        ),
        wavelength_angstrom=wavelength,
        radiation=radiation_label,
        two_theta_range_deg=(lower, upper),
        metadata=ScientificMetadata(
            classification=ResultClassification.PHYSICS_CALCULATED,
            confidence="high for the supplied ideal structural model and kinematic approximation",
            limitations=(
                "Calculated intensities assume an ideal powder and kinematic diffraction.",
                "Texture, absorption, instrument optics, crystallite size/strain broadening, defects, and temperature factors may differ experimentally.",
                "Phase identification is valid only if the supplied structure is appropriate to the specimen.",
                *inherited_limitations,
            ),
            provenance=(
                Provenance(
                    title="pymatgen XRDCalculator",
                    citation=(
                        "Ong, S. P. et al. (2013), Python Materials Genomics "
                        "(pymatgen), Computational Materials Science 68, 314-319"
                    ),
                    url="https://doi.org/10.1016/j.commatsci.2012.10.028",
                ),
                *inherited_provenance,
            ),
        ),
    )
