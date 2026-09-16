"""Pole-figure projections from explicit orientation data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from crystallography.orientation import plane_normal_cartesian, validate_rotation_matrix


@dataclass(frozen=True)
class PoleFigureResult:
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    hkl: tuple[float, float, float]
    projection: str = "Lambert equal-area, upper hemisphere"

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x.tolist(),
            "y": self.y.tolist(),
            "weights": self.weights.tolist(),
            "hkl": list(self.hkl),
            "projection": self.projection,
            "classification": "physics_calculated",
            "confidence": "deterministic for the supplied orientations and convention",
            "limitations": [
                "No crystal symmetry expansion, defocusing correction, or ODF inversion is applied automatically.",
                "Statistical representativeness depends on the supplied orientation measurements and weights.",
            ],
        }


def pole_figure_points(
    orientation_matrices: Sequence[Sequence[Sequence[float]]],
    hkl: Sequence[float],
    lattice_matrix: Sequence[Sequence[float]],
    *,
    weights: Sequence[float] | None = None,
) -> PoleFigureResult:
    """Project one crystal plane normal per orientation to an upper-hemisphere pole figure."""

    normal = plane_normal_cartesian(hkl, lattice_matrix)
    orientations = [validate_rotation_matrix(matrix) for matrix in orientation_matrices]
    if not orientations:
        raise ValueError("At least one orientation matrix is required.")
    poles = np.asarray([matrix @ normal for matrix in orientations], dtype=float)
    poles[poles[:, 2] < 0] *= -1.0
    denominator = np.sqrt(np.maximum(1.0 + poles[:, 2], np.finfo(float).eps))
    x = poles[:, 0] / denominator
    y = poles[:, 1] / denominator
    if weights is None:
        point_weights = np.ones(len(poles), dtype=float)
    else:
        point_weights = np.asarray(weights, dtype=float)
        if point_weights.shape != (len(poles),) or not np.all(np.isfinite(point_weights)):
            raise ValueError("weights must contain one finite value per orientation.")
        if np.any(point_weights < 0) or not np.any(point_weights > 0):
            raise ValueError("weights must be non-negative with at least one positive value.")
    return PoleFigureResult(
        x,
        y,
        point_weights,
        tuple(float(value) for value in hkl),
    )


def pole_density_histogram(
    result: PoleFigureResult, *, bins: int = 36
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a weighted 2D histogram; this is not a corrected ODF."""

    if not isinstance(bins, int) or bins < 4:
        raise ValueError("bins must be an integer of at least 4.")
    return np.histogram2d(
        result.x,
        result.y,
        bins=bins,
        range=((-1.0, 1.0), (-1.0, 1.0)),
        weights=result.weights,
    )

