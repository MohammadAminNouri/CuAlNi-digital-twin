"""Orientation and crystallographic vector calculations.

Conventions are explicit: lattice vectors are rows and an orientation matrix maps
crystal Cartesian column vectors into the sample frame.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("Expected a finite three-component vector.")
    norm = float(np.linalg.norm(value))
    if norm <= 0:
        raise ValueError("A zero vector has no crystallographic direction.")
    return value / norm


def validate_rotation_matrix(matrix: Sequence[Sequence[float]], *, atol: float = 1e-7) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("Orientation matrix must be finite and 3 x 3.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0):
        raise ValueError("Orientation matrix is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=atol, rtol=0):
        raise ValueError("Orientation matrix must be a proper rotation (determinant +1).")
    return rotation


def bunge_euler_matrix(phi1: float, Phi: float, phi2: float, *, degrees: bool = True) -> np.ndarray:
    """Return the passive Bunge ZXZ crystal-to-sample orientation matrix."""

    angles = np.asarray([phi1, Phi, phi2], dtype=float)
    if not np.all(np.isfinite(angles)):
        raise ValueError("Euler angles must be finite.")
    if degrees:
        angles = np.deg2rad(angles)
    p1, p, p2 = angles
    c1, c, c2 = np.cos([p1, p, p2])
    s1, s, s2 = np.sin([p1, p, p2])
    matrix = np.array(
        [
            [c1 * c2 - s1 * s2 * c, s1 * c2 + c1 * s2 * c, s2 * s],
            [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
            [s1 * s, -c1 * s, c],
        ]
    )
    return validate_rotation_matrix(matrix)


def crystal_direction_cartesian(
    uvw: Sequence[float], lattice_matrix: Sequence[Sequence[float]]
) -> np.ndarray:
    """Convert a direct-lattice [uvw] direction to a Cartesian unit vector."""

    lattice = np.asarray(lattice_matrix, dtype=float)
    if lattice.shape != (3, 3) or abs(float(np.linalg.det(lattice))) < 1e-12:
        raise ValueError("lattice_matrix must be a nonsingular 3 x 3 row-vector matrix.")
    return _unit(np.asarray(uvw, dtype=float) @ lattice)


def plane_normal_cartesian(
    hkl: Sequence[float], lattice_matrix: Sequence[Sequence[float]]
) -> np.ndarray:
    """Convert plane indices (hkl) to a Cartesian reciprocal-space unit normal."""

    lattice = np.asarray(lattice_matrix, dtype=float)
    if lattice.shape != (3, 3) or abs(float(np.linalg.det(lattice))) < 1e-12:
        raise ValueError("lattice_matrix must be a nonsingular 3 x 3 row-vector matrix.")
    reciprocal_rows = np.linalg.inv(lattice).T
    return _unit(np.asarray(hkl, dtype=float) @ reciprocal_rows)


def misorientation_angle_deg(
    first: Sequence[Sequence[float]], second: Sequence[Sequence[float]]
) -> float:
    """Return disorientation angle without applying crystal symmetry operators."""

    g1 = validate_rotation_matrix(first)
    g2 = validate_rotation_matrix(second)
    delta = g1 @ g2.T
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def minimum_misorientation_deg(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
    symmetry_operators: Iterable[Sequence[Sequence[float]]],
) -> float:
    """Return the minimum angle over explicitly supplied proper symmetry rotations."""

    g1 = validate_rotation_matrix(first)
    g2 = validate_rotation_matrix(second)
    angles = [
        misorientation_angle_deg(g1, g2 @ validate_rotation_matrix(operator))
        for operator in symmetry_operators
    ]
    if not angles:
        raise ValueError("At least one symmetry operator is required.")
    return min(angles)

