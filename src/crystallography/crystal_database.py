"""Provenance-first structure loading and literature reference registry.

Only a complete CIF or a complete user-supplied lattice/site model is marked
diffraction-ready. Literature records that report cell lengths without angles
or atomic coordinates remain metadata and cannot silently generate XRD peaks.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

from scientific import Provenance, ResultClassification, ScientificMetadata

OTSUKA_1974 = Provenance(
    title="Electron Microscopy Study of Stress-induced Acicular beta1-prime Martensite in Cu-Al-Ni Alloy",
    citation=(
        "Otsuka, K., Nakamura, T. & Shimizu, K. (1974), Transactions of the "
        "Japan Institute of Metals 15, 200-210"
    ),
    url="https://doi.org/10.2320/matertrans1960.15.200",
    identifier="10.2320/matertrans1960.15.200",
)

XIE_2006 = Provenance(
    title="Electron microscopy study of 2H and 18R martensites in Cu-11.92 wt% Al-3.78 wt% Ni shape memory alloy",
    citation="Xie et al. (2006), Journal of Alloys and Compounds 417, 170-175",
    url="https://doi.org/10.1016/j.jallcom.2005.09.049",
    identifier="10.1016/j.jallcom.2005.09.049",
)

GAMMA_BRASS_REFERENCE = Provenance(
    title="gamma-Cu9Al4 AFLOW prototype",
    citation=(
        "gamma-Cu9Al4, Strukturbericht D8_3, Pearson cP52, space group P-43m; "
        "original refinement doi:10.1107/S0567739478000807"
    ),
    url="https://aflow.org/p/A4B9_cP52_215_ei_3efgi-001",
    identifier="A4B9_cP52_215_ei_3efgi-001",
)


@dataclass(frozen=True)
class LiteratureCrystalReference:
    phase: str
    material: str
    composition: str | None
    lattice_lengths_angstrom: Mapping[str, float] | None
    lattice_angles_deg: Mapping[str, float] | None
    space_group: str | None
    stacking_sequence: str | None
    provenance: Provenance
    limitations: tuple[str, ...]

    @property
    def diffraction_ready(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe reference metadata without implying a structure model."""

        return {
            "phase": self.phase,
            "material": self.material,
            "composition": self.composition,
            "lattice_lengths_angstrom": (
                dict(self.lattice_lengths_angstrom)
                if self.lattice_lengths_angstrom is not None
                else None
            ),
            "lattice_angles_deg": (
                dict(self.lattice_angles_deg) if self.lattice_angles_deg is not None else None
            ),
            "space_group": self.space_group,
            "stacking_sequence": self.stacking_sequence,
            "diffraction_ready": False,
            "metadata": ScientificMetadata(
                classification=ResultClassification.REFERENCE_DATA,
                confidence="literature metadata; not a complete crystallographic model",
                limitations=self.limitations,
                provenance=(self.provenance,),
            ).to_dict(),
        }


LITERATURE_REFERENCES: dict[str, LiteratureCrystalReference] = {
    "18R": LiteratureCrystalReference(
        phase="18R martensite (beta1-prime)",
        material="Cu-Al-Ni",
        composition="Cu-14.2Al-4.3Ni wt%",
        lattice_lengths_angstrom={"a": 4.382, "b": 5.356, "c": 38.00},
        lattice_angles_deg=None,
        space_group=None,
        stacking_sequence="AB'CB'CA'CA'BA'BC'BC'AC'AB'",
        provenance=OTSUKA_1974,
        limitations=(
            "The indexed source summary provides cell lengths and stacking sequence but not a complete CIF here.",
            "Angles and atomic coordinates are intentionally not inferred; this record cannot generate diffraction intensities.",
            "Reported values apply to the cited Cu-14.2Al-4.3Ni wt% specimen, not every Cu-Al-Ni composition.",
        ),
    ),
    "2H": LiteratureCrystalReference(
        phase="2H martensite (gamma-prime)",
        material="Cu-Al-Ni",
        composition="Cu-11.92Al-3.78Ni wt%",
        lattice_lengths_angstrom=None,
        lattice_angles_deg=None,
        space_group="Pnmm (as reported by source)",
        stacking_sequence="2H",
        provenance=XIE_2006,
        limitations=(
            "The cited phase assignment does not provide a complete structure model in this registry.",
            "A specimen-appropriate CIF is required for visualization and XRD.",
        ),
    ),
    "GAMMA2": LiteratureCrystalReference(
        phase="gamma2 Cu9Al4",
        material="Cu9Al4",
        composition="Cu9Al4",
        lattice_lengths_angstrom=None,
        lattice_angles_deg=None,
        space_group="P-43m (No. 215), cP52, D8_3 prototype",
        stacking_sequence=None,
        provenance=GAMMA_BRASS_REFERENCE,
        limitations=(
            "Prototype metadata alone is not a complete atomic structure.",
            "Load a cited CIF containing its refined lattice and internal coordinates for XRD.",
        ),
    ),
}


@dataclass
class CrystalStructureRecord:
    name: str
    structure: Any
    source_path: str | None
    source_sha256: str | None
    metadata: ScientificMetadata

    @property
    def diffraction_ready(self) -> bool:
        return self.structure is not None

    def to_dict(self) -> dict[str, Any]:
        if self.structure is None:
            formula = None
            site_count = 0
            lattice = None
            serialized_structure = None
        else:
            formula = str(self.structure.composition.reduced_formula)
            site_count = len(self.structure)
            lattice = {
                "a_angstrom": float(self.structure.lattice.a),
                "b_angstrom": float(self.structure.lattice.b),
                "c_angstrom": float(self.structure.lattice.c),
                "alpha_deg": float(self.structure.lattice.alpha),
                "beta_deg": float(self.structure.lattice.beta),
                "gamma_deg": float(self.structure.lattice.gamma),
                "volume_angstrom3": float(self.structure.lattice.volume),
            }
            serialized_structure = structure_to_mapping(self.structure)
        return {
            "name": self.name,
            "formula": formula,
            "site_count": site_count,
            "lattice": lattice,
            "structure": serialized_structure,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "diffraction_ready": self.diffraction_ready,
            "metadata": self.metadata.to_dict(),
        }


def _pymatgen_structure() -> Any:
    try:
        from pymatgen.core import Structure
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pymatgen is required to load or construct crystal structures.") from exc
    return Structure


def _species_symbol(specie: Any) -> str:
    """Return an element/species label without discarding oxidation information in source data."""

    element = getattr(specie, "element", None)
    if element is not None and hasattr(element, "symbol"):
        return str(element.symbol)
    if hasattr(specie, "symbol"):
        return str(specie.symbol)
    return str(specie)


def structure_to_mapping(structure: Any) -> dict[str, Any]:
    """Serialize a pymatgen-like structure for the UI and report pipeline.

    Fractional coordinates and all site occupancies are retained.  This canonical
    representation is deliberately smaller and more stable than pymatgen's
    implementation-specific ``as_dict`` payload, while remaining reconstructable
    by :func:`structure_from_mapping`.
    """

    if structure is None or not hasattr(structure, "lattice"):
        raise TypeError("Expected a complete pymatgen-like periodic structure.")
    matrix = [[float(value) for value in row] for row in structure.lattice.matrix]
    sites: list[dict[str, Any]] = []
    for site in structure:
        species_entries = [
            {"element": _species_symbol(specie), "occupancy": float(occupancy)}
            for specie, occupancy in site.species.items()
        ]
        site_payload: dict[str, Any] = {
            "species": species_entries,
            "fractional_coordinates": [float(value) for value in site.frac_coords],
        }
        if len(species_entries) == 1:
            site_payload.update(species_entries[0])
        sites.append(site_payload)
    composition = getattr(structure, "composition", None)
    return {
        "format": "cualni.structure.v1",
        "formula": str(getattr(composition, "reduced_formula", "")) or None,
        "ordered": bool(getattr(structure, "is_ordered", False)),
        "lattice": {
            "matrix": matrix,
            "a_angstrom": float(structure.lattice.a),
            "b_angstrom": float(structure.lattice.b),
            "c_angstrom": float(structure.lattice.c),
            "alpha_deg": float(structure.lattice.alpha),
            "beta_deg": float(structure.lattice.beta),
            "gamma_deg": float(structure.lattice.gamma),
            "volume_angstrom3": float(structure.lattice.volume),
        },
        "sites": sites,
    }


def structure_from_mapping(payload: Mapping[str, Any]) -> Any:
    """Reconstruct a pymatgen Structure from a canonical or pymatgen mapping."""

    Structure = _pymatgen_structure()
    candidate: Mapping[str, Any] = payload
    nested = candidate.get("structure")
    if isinstance(nested, Mapping):
        candidate = nested
    if "@module" in candidate and "lattice" in candidate:
        try:
            return Structure.from_dict(dict(candidate))
        except Exception:
            # Continue with the implementation-independent parser below.
            pass
    lattice_data = candidate.get("lattice", candidate.get("lattice_matrix"))
    if isinstance(lattice_data, Mapping):
        lattice_data = lattice_data.get("matrix")
    sites = candidate.get("sites")
    if lattice_data is None or not isinstance(sites, Sequence) or isinstance(sites, (str, bytes)):
        raise ValueError("Structure mapping must contain lattice.matrix and a sites sequence.")
    species: list[Any] = []
    fractional_coordinates: list[Sequence[float]] = []
    for index, site in enumerate(sites):
        if not isinstance(site, Mapping):
            raise ValueError(f"Structure site {index} must be a mapping.")
        coordinates = site.get(
            "fractional_coordinates", site.get("frac_coords", site.get("abc"))
        )
        if coordinates is None:
            raise ValueError(f"Structure site {index} lacks fractional coordinates.")
        raw_species = site.get("species", site.get("element"))
        if isinstance(raw_species, Sequence) and not isinstance(raw_species, (str, bytes)):
            occupancy_map: dict[str, float] = {}
            for entry in raw_species:
                if not isinstance(entry, Mapping):
                    raise ValueError(f"Structure site {index} contains an invalid species entry.")
                label = entry.get("element", entry.get("name"))
                if label is None:
                    raise ValueError(f"Structure site {index} species entry lacks an element.")
                occupancy_map[str(label)] = occupancy_map.get(str(label), 0.0) + float(
                    entry.get("occupancy", entry.get("occu", 1.0))
                )
            raw_species = occupancy_map
        elif isinstance(raw_species, Mapping):
            raw_species = {str(key): float(value) for key, value in raw_species.items()}
        elif raw_species is not None and "occupancy" in site:
            raw_species = {str(raw_species): float(site["occupancy"])}
        if raw_species is None:
            raise ValueError(f"Structure site {index} lacks species information.")
        species.append(raw_species)
        fractional_coordinates.append(coordinates)
    try:
        return Structure(lattice_data, species, fractional_coordinates)
    except Exception as exc:
        raise ValueError(f"Could not reconstruct structure mapping: {exc}") from exc


def coerce_structure(value: Any) -> Any:
    """Return a pymatgen structure from a record, structure, or serialized mapping."""

    if isinstance(value, CrystalStructureRecord):
        if value.structure is None:
            raise ValueError("The structure record contains metadata only.")
        return value.structure
    if isinstance(value, Mapping):
        return structure_from_mapping(value)
    if hasattr(value, "lattice") and hasattr(value, "frac_coords"):
        return value
    raise TypeError("Expected a CrystalStructureRecord, pymatgen Structure, or structure mapping.")


def get_literature_reference(phase: str) -> LiteratureCrystalReference:
    key = str(phase).strip().upper().replace("MARTENSITE", "").strip()
    aliases = {"18R": "18R", "2H": "2H", "GAMMA2": "GAMMA2", "GAMMA_2": "GAMMA2"}
    if key not in aliases:
        raise KeyError(f"No curated reference metadata for {phase!r}.")
    return LITERATURE_REFERENCES[aliases[key]]


def list_literature_references() -> dict[str, dict[str, Any]]:
    """Return all curated metadata records; none is promoted to a complete structure."""

    return {name: reference.to_dict() for name, reference in LITERATURE_REFERENCES.items()}


def load_cif(
    path: str | Path,
    *,
    name: str | None = None,
    citation: str | None = None,
    source_url: str | None = None,
) -> CrystalStructureRecord:
    """Load a local CIF and preserve its content hash and supplied citation."""

    cif_path = Path(path).expanduser().resolve()
    if not cif_path.is_file():
        raise FileNotFoundError(f"CIF not found: {cif_path}")
    Structure = _pymatgen_structure()
    try:
        structure = Structure.from_file(str(cif_path))
    except Exception as exc:
        raise ValueError(f"pymatgen could not parse CIF {cif_path.name}: {exc}") from exc
    digest = sha256(cif_path.read_bytes()).hexdigest()
    provenance = Provenance(
        title=cif_path.name,
        citation=citation or "User-supplied CIF; original crystallographic citation not provided.",
        url=source_url,
        identifier=str(cif_path),
        sha256=digest,
    )
    limitations = []
    if citation is None:
        limitations.append("Original crystallographic citation was not supplied; verify the CIF pedigree.")
    if not structure.is_ordered:
        limitations.append("Structure contains partial occupancies; symmetry/XRD handling is model-dependent.")
    return CrystalStructureRecord(
        name=name or cif_path.stem,
        structure=structure,
        source_path=str(cif_path),
        source_sha256=digest,
        metadata=ScientificMetadata(
            classification=ResultClassification.USER_SUPPLIED,
            confidence="dependent on CIF refinement quality and applicability to the specimen",
            limitations=tuple(limitations),
            provenance=(provenance,),
        ),
    )


def load_cif_data(
    payload: str | bytes,
    *,
    source_filename: str = "uploaded.cif",
    name: str | None = None,
    citation: str | None = None,
    source_url: str | None = None,
) -> CrystalStructureRecord:
    """Load CIF text/bytes without creating a temporary file.

    The original bytes are hashed so an uploaded structure remains auditable in
    exported reports.  Pymatgen performs the CIF parsing and structural checks.
    """

    if isinstance(payload, bytes):
        raw = payload
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = payload.decode("latin-1")
    elif isinstance(payload, str):
        text = payload
        raw = payload.encode("utf-8")
    else:
        raise TypeError("CIF payload must be text or bytes.")
    if not text.strip():
        raise ValueError("CIF payload is empty.")
    Structure = _pymatgen_structure()
    try:
        structure = Structure.from_str(text, fmt="cif")
    except Exception as exc:
        raise ValueError(f"pymatgen could not parse CIF {source_filename}: {exc}") from exc
    digest = sha256(raw).hexdigest()
    provenance = Provenance(
        title=source_filename,
        citation=citation or "User-supplied CIF; original crystallographic citation not provided.",
        url=source_url,
        identifier=source_filename,
        sha256=digest,
    )
    limitations: list[str] = []
    if citation is None:
        limitations.append("Original crystallographic citation was not supplied; verify the CIF pedigree.")
    if not structure.is_ordered:
        limitations.append("Structure contains partial occupancies; symmetry/XRD handling is model-dependent.")
    return CrystalStructureRecord(
        name=name or Path(source_filename).stem,
        structure=structure,
        source_path=None,
        source_sha256=digest,
        metadata=ScientificMetadata(
            classification=ResultClassification.USER_SUPPLIED,
            confidence="dependent on CIF refinement quality and applicability to the specimen",
            limitations=tuple(limitations),
            provenance=(provenance,),
        ),
    )


def structure_from_user_data(
    *,
    name: str,
    lattice: Sequence[Sequence[float]] | Sequence[float],
    species: Sequence[Any],
    fractional_coordinates: Sequence[Sequence[float]],
    citation: str | None = None,
) -> CrystalStructureRecord:
    """Construct a complete periodic structure from explicit user inputs."""

    Structure = _pymatgen_structure()
    try:
        structure = Structure(lattice, species, fractional_coordinates)
    except Exception as exc:
        raise ValueError(f"Invalid lattice/species/fractional-coordinate model: {exc}") from exc
    return CrystalStructureRecord(
        name=name,
        structure=structure,
        source_path=None,
        source_sha256=None,
        metadata=ScientificMetadata(
            classification=ResultClassification.USER_SUPPLIED,
            confidence="dependent on the supplied structural model",
            limitations=(
                "No structural relaxation or diffraction refinement is performed.",
                "The model is not automatically representative of a different alloy composition or heat treatment.",
            ),
            provenance=(
                Provenance(
                    title=name,
                    citation=citation or "User-supplied lattice and fractional atomic coordinates.",
                ),
            ),
        ),
    )
