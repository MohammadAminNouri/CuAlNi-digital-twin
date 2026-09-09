"""Dependency-free triclinic lattice and Bragg-law calculations."""

from __future__ import annotations

from math import acos, asin, cos, degrees, sin, sqrt
from typing import Iterable

import numpy as np


def lattice_matrix(
    a_angstrom: float,
    b_angstrom: float,
    c_angstrom: float,
    alpha_deg: float = 90.0,
    beta_deg: float = 90.0,
    gamma_deg: float = 90.0,
) -> np.ndarray:
    """Return a row-vector crystallographic lattice matrix in ångström."""

    lengths = np.asarray([a_angstrom, b_angstrom, c_angstrom], dtype=float)
    angles = np.asarray([alpha_deg, beta_deg, gamma_deg], dtype=float)
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Lattice lengths must be finite and positive.")
    if not np.all(np.isfinite(angles)) or np.any(angles <= 0) or np.any(angles >= 180):
        raise ValueError("Lattice angles must lie strictly between 0 and 180 degrees.")
    a, b, c = lengths
    alpha, beta, gamma = np.radians(angles)
    sin_gamma = sin(gamma)
    if abs(sin_gamma) < 1e-12:
        raise ValueError("Degenerate lattice: sin(gamma) is zero.")
    c_x = c * cos(beta)
    c_y = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin_gamma
    c_z_squared = c * c - c_x * c_x - c_y * c_y
    if c_z_squared <= 0:
        raise ValueError("Lattice lengths and angles do not define a positive-volume cell.")
    return np.asarray(
        [
            [a, 0.0, 0.0],
            [b * cos(gamma), b * sin_gamma, 0.0],
            [c_x, c_y, sqrt(c_z_squared)],
        ],
        dtype=float,
    )


def cell_parameters(matrix: Iterable[Iterable[float]]) -> dict[str, float]:
    """Recover conventional lengths and angles from a row-vector matrix."""

    cell = _validate_matrix(matrix)
    a, b, c = (float(np.linalg.norm(vector)) for vector in cell)

    def angle(first: np.ndarray, second: np.ndarray) -> float:
        cosine = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        return degrees(acos(float(np.clip(cosine, -1.0, 1.0))))

    return {
        "a_angstrom": a,
        "b_angstrom": b,
        "c_angstrom": c,
        "alpha_deg": angle(cell[1], cell[2]),
        "beta_deg": angle(cell[0], cell[2]),
        "gamma_deg": angle(cell[0], cell[1]),
    }


def _validate_matrix(matrix: Iterable[Iterable[float]]) -> np.ndarray:
    cell = np.asarray(matrix, dtype=float)
    if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
        raise ValueError("lattice matrix must be a finite 3x3 array.")
    if abs(np.linalg.det(cell)) < 1e-14:
        raise ValueError("lattice matrix is singular.")
    return cell


def unit_cell_volume(matrix: Iterable[Iterable[float]]) -> float:
    """Return the absolute unit-cell volume in Å³."""

    return float(abs(np.linalg.det(_validate_matrix(matrix))))


def metric_tensor(matrix: Iterable[Iterable[float]]) -> np.ndarray:
    """Return direct-space metric tensor ``G = A A.T``."""

    cell = _validate_matrix(matrix)
    return cell @ cell.T


def reciprocal_lattice_matrix(
    matrix: Iterable[Iterable[float]], *, include_2pi: bool = False
) -> np.ndarray:
    """Return reciprocal basis as rows, optionally using the physics 2π convention."""

    reciprocal = np.linalg.inv(_validate_matrix(matrix)).T
    return reciprocal * (2.0 * np.pi if include_2pi else 1.0)


def d_spacing(matrix: Iterable[Iterable[float]], hkl: Iterable[int]) -> float:
    """Calculate interplanar spacing for arbitrary lattice and Miller indices."""

    indices = np.asarray(list(hkl), dtype=float)
    if indices.shape != (3,) or not np.all(np.isfinite(indices)):
        raise ValueError("hkl must contain three finite indices.")
    if np.allclose(indices, 0):
        raise ValueError("(0, 0, 0) is not a reflecting plane.")
    reciprocal_vector = indices @ reciprocal_lattice_matrix(matrix)
    magnitude = float(np.linalg.norm(reciprocal_vector))
    return 1.0 / magnitude


def bragg_two_theta(
    d_angstrom: float, wavelength_angstrom: float, *, order: int = 1
) -> float | None:
    """Calculate 2θ in degrees from Bragg's law, or ``None`` if inaccessible."""

    d_value = float(d_angstrom)
    wavelength = float(wavelength_angstrom)
    if not np.isfinite(d_value) or d_value <= 0:
        raise ValueError("d_angstrom must be finite and positive.")
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength_angstrom must be finite and positive.")
    if not isinstance(order, int) or order <= 0:
        raise ValueError("order must be a positive integer.")
    sine_theta = order * wavelength / (2.0 * d_value)
    if sine_theta > 1.0:
        return None
    return 2.0 * degrees(asin(sine_theta))
