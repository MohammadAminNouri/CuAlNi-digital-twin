import numpy as np
import pytest

from crystallography.crystal_database import list_literature_references
from crystallography.lattice import (
    bragg_two_theta,
    cell_parameters,
    d_spacing,
    lattice_matrix,
    unit_cell_volume,
)
from crystallography.orientation import (
    bunge_euler_matrix,
    crystal_direction_cartesian,
    misorientation_angle_deg,
    plane_normal_cartesian,
)
from crystallography.texture import pole_figure_points


def test_cubic_lattice_geometry_and_bragg_law():
    cell = lattice_matrix(3.0, 3.0, 3.0)
    assert np.isclose(unit_cell_volume(cell), 27.0)
    assert np.isclose(d_spacing(cell, (1, 1, 0)), 3.0 / np.sqrt(2))
    angle = bragg_two_theta(d_spacing(cell, (1, 1, 0)), 1.5406)
    assert angle is not None and 0 < angle < 180


def test_triclinic_round_trip_parameters():
    cell = lattice_matrix(4.0, 5.0, 6.0, 80.0, 95.0, 110.0)
    parameters = cell_parameters(cell)
    assert np.isclose(parameters["a_angstrom"], 4.0)
    assert np.isclose(parameters["alpha_deg"], 80.0)
    assert unit_cell_volume(cell) > 0


def test_orientation_identity_and_vector_conventions():
    identity = bunge_euler_matrix(0, 0, 0)
    assert np.allclose(identity, np.eye(3))
    assert np.isclose(misorientation_angle_deg(identity, identity), 0.0)
    cell = np.diag([2.0, 3.0, 4.0])
    assert np.allclose(crystal_direction_cartesian((1, 0, 0), cell), (1, 0, 0))
    assert np.allclose(plane_normal_cartesian((0, 0, 1), cell), (0, 0, 1))


def test_pole_projection_stays_inside_equal_area_circle():
    result = pole_figure_points([np.eye(3)], (0, 0, 1), np.eye(3))
    assert np.hypot(result.x[0], result.y[0]) <= 1.0
    assert result.to_dict()["classification"] == "physics_calculated"


def test_literature_references_are_not_diffraction_ready():
    records = list_literature_references()
    assert {"18R", "2H", "GAMMA2"}.issubset(records)
    assert all(record["diffraction_ready"] is False for record in records.values())


def test_zero_miller_index_is_rejected():
    with pytest.raises(ValueError):
        d_spacing(np.eye(3), (0, 0, 0))

