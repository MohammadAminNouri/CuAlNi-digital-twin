"""Crystallographic geometry, symmetry, texture, and diffraction APIs."""

from .crystal_database import (
    CrystalStructureRecord,
    LiteratureCrystalReference,
    get_literature_reference,
    list_literature_references,
    load_cif,
    load_cif_data,
    structure_from_user_data,
)
from .diffraction import XRDPattern, simulate_powder_pattern
from .lattice import bragg_two_theta, d_spacing, lattice_matrix, unit_cell_volume
from .orientation import (
    bunge_euler_matrix,
    crystal_direction_cartesian,
    minimum_misorientation_deg,
    misorientation_angle_deg,
    plane_normal_cartesian,
)
from .symmetry import SymmetryResult, analyze_symmetry
from .texture import PoleFigureResult, pole_density_histogram, pole_figure_points

__all__ = [
    "CrystalStructureRecord",
    "LiteratureCrystalReference",
    "SymmetryResult",
    "XRDPattern",
    "analyze_symmetry",
    "bragg_two_theta",
    "d_spacing",
    "get_literature_reference",
    "lattice_matrix",
    "load_cif",
    "load_cif_data",
    "list_literature_references",
    "bunge_euler_matrix",
    "crystal_direction_cartesian",
    "minimum_misorientation_deg",
    "misorientation_angle_deg",
    "plane_normal_cartesian",
    "PoleFigureResult",
    "pole_density_histogram",
    "pole_figure_points",
    "simulate_powder_pattern",
    "structure_from_user_data",
    "unit_cell_volume",
]
