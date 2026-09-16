"""spglib-backed symmetry analysis with disclosed numerical tolerance."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

from scientific import Provenance, ResultClassification, ScientificMetadata


@dataclass(frozen=True)
class SymmetryResult:
    international_symbol: str
    hall_symbol: str
    space_group_number: int
    point_group: str
    equivalent_atoms: tuple[int, ...]
    symprec_angstrom: float
    angle_tolerance_deg: float
    metadata: ScientificMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "international_symbol": self.international_symbol,
            "hall_symbol": self.hall_symbol,
            "space_group_number": self.space_group_number,
            "point_group": self.point_group,
            "equivalent_atoms": list(self.equivalent_atoms),
            "symprec_angstrom": self.symprec_angstrom,
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "metadata": self.metadata.to_dict(),
        }


def _spglib_cell(structure: Any) -> tuple[Any, Any, Any]:
    if hasattr(structure, "lattice") and hasattr(structure, "frac_coords"):
        if hasattr(structure, "is_ordered") and not structure.is_ordered:
            raise ValueError("spglib symmetry requires an ordered site model, not partial occupancies.")
        numbers = [int(site.specie.Z) for site in structure]
        return structure.lattice.matrix, structure.frac_coords, numbers
    if hasattr(structure, "get_cell") and hasattr(structure, "get_scaled_positions"):
        return structure.get_cell(), structure.get_scaled_positions(), structure.get_atomic_numbers()
    if isinstance(structure, tuple) and len(structure) == 3:
        return structure
    raise TypeError("Expected a pymatgen Structure, ASE Atoms, or spglib (lattice, positions, numbers) tuple.")


def analyze_symmetry(
    structure: Any, *, symprec_angstrom: float = 1e-3, angle_tolerance_deg: float = -1.0
) -> SymmetryResult:
    """Determine crystallographic symmetry with spglib at explicit tolerances."""

    try:
        import spglib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("spglib is required for symmetry analysis.") from exc
    symprec = float(symprec_angstrom)
    angle_tolerance = float(angle_tolerance_deg)
    if symprec <= 0:
        raise ValueError("symprec_angstrom must be positive.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Set OLD_ERROR_HANDLING to false.*",
            category=DeprecationWarning,
            module="spglib.*",
        )
        dataset = spglib.get_symmetry_dataset(
            _spglib_cell(structure), symprec=symprec, angle_tolerance=angle_tolerance
        )
    if dataset is None:
        raise RuntimeError("spglib could not determine symmetry at the selected tolerances.")
    # Current spglib exposes attributes; older versions use mapping keys.
    def field(name: str) -> Any:
        return getattr(dataset, name) if hasattr(dataset, name) else dataset[name]

    return SymmetryResult(
        international_symbol=str(field("international")),
        hall_symbol=str(field("hall")),
        space_group_number=int(field("number")),
        point_group=str(field("pointgroup")),
        equivalent_atoms=tuple(int(value) for value in field("equivalent_atoms")),
        symprec_angstrom=symprec,
        angle_tolerance_deg=angle_tolerance,
        metadata=ScientificMetadata(
            classification=ResultClassification.PHYSICS_CALCULATED,
            confidence="tolerance-dependent numerical symmetry assignment",
            limitations=(
                "Space-group assignment can change with atomic-coordinate precision and selected tolerances.",
                "Disordered/partially occupied structures must be represented by an explicit ordered model.",
            ),
            provenance=(
                Provenance(
                    title="spglib",
                    citation="Togo, A. et al., spglib: a software library for crystal symmetry search",
                    url="https://spglib.readthedocs.io/",
                ),
            ),
        ),
    )
