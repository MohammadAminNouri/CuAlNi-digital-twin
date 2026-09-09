"""Auditable pycalphad adapter for Cu-Al-Ni equilibrium calculations.

This module contains no fabricated Gibbs-energy parameters and no embedded TDB.
A calculation can run only after a user supplies a parseable external database
that contains Al, Cu and Ni.  Phase names are always those in that database;
the code does not guess that e.g. ``BCC_B2`` is a particular martensite.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from composition.converter import ELEMENTS, AlloyComposition
from scientific import Provenance, ResultClassification, ScientificMetadata

PYCALPHAD_REFERENCE = Provenance(
    title="pycalphad",
    citation=(
        "Otis, R. & Liu, Z.-K. (2017), pycalphad: CALPHAD-based Computational "
        "Thermodynamics in Python, Journal of Open Research Software 5(1), 1"
    ),
    url="https://doi.org/10.5334/jors.140",
)


class CalphadDependencyError(RuntimeError):
    """Raised when the optional pycalphad runtime is unavailable."""


class DatabaseCoverageError(ValueError):
    """Raised when a database cannot support the requested system/calculation."""


@dataclass(frozen=True)
class DatabaseInfo:
    path: str
    sha256: str
    elements: tuple[str, ...]
    phases: tuple[str, ...]
    pycalphad_version: str
    citation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "elements": list(self.elements),
            "phases": list(self.phases),
            "pycalphad_version": self.pycalphad_version,
            "citation": self.citation,
        }


@dataclass
class EquilibriumResult:
    """Tidy molar phase fractions and the original pycalphad Dataset."""

    phase_fractions: pd.DataFrame
    raw_dataset: Any
    database: DatabaseInfo
    conditions: Mapping[str, Any]
    metadata: ScientificMetadata

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase_fractions": self.phase_fractions.to_dict(orient="records"),
            "database": self.database.to_dict(),
            "conditions": dict(self.conditions),
            "metadata": self.metadata.to_dict(),
        }


def _load_pycalphad() -> tuple[Any, Any, Any, str]:
    try:
        import pycalphad
        from pycalphad import Database, equilibrium, variables
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CalphadDependencyError(
            "pycalphad is required for thermodynamic calculations. Install the "
            "project's CALPHAD dependencies and supply a compatible .tdb/.dat file."
        ) from exc
    return Database, equilibrium, variables, str(pycalphad.__version__)


def _temperatures(values: float | Iterable[float]) -> np.ndarray:
    if np.isscalar(values):
        array = np.asarray([values], dtype=float)
    else:
        array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("temperatures_K must be a non-empty one-dimensional sequence.")
    if not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError("temperatures_K must contain finite absolute temperatures above 0 K.")
    return array


class CalphadEngine:
    """Validated interface around a user-supplied pycalphad database."""

    def __init__(self, tdb_path: str | Path, *, database_citation: str | None = None):
        path = Path(tdb_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Thermodynamic database not found: {path}")
        if path.suffix.lower() not in {".tdb", ".dat", ".xml"}:
            raise ValueError("Expected a pycalphad-readable .tdb, .dat, or plugin-supported .xml file.")
        Database, equilibrium, variables, version = _load_pycalphad()
        try:
            database = Database(str(path))
        except Exception as exc:
            raise DatabaseCoverageError(f"pycalphad could not parse {path.name}: {exc}") from exc
        self.path = path
        self.dbf = database
        self._equilibrium = equilibrium
        self.variables = variables
        self.database_citation = database_citation or (
            "User-supplied thermodynamic database; assessment citation not provided."
        )
        self.info = DatabaseInfo(
            path=str(path),
            sha256=sha256(path.read_bytes()).hexdigest(),
            elements=tuple(sorted(str(element).upper() for element in database.elements)),
            phases=tuple(sorted(str(phase) for phase in database.phases)),
            pycalphad_version=version,
            citation=self.database_citation,
        )
        self.validate_system()

    @property
    def available_phases(self) -> tuple[str, ...]:
        return self.info.phases

    def validate_system(self) -> DatabaseInfo:
        missing = {element.upper() for element in ELEMENTS} - set(self.info.elements)
        if missing:
            raise DatabaseCoverageError(
                "Database does not cover the complete Cu-Al-Ni system; missing "
                + ", ".join(sorted(missing))
            )
        if not self.info.phases:
            raise DatabaseCoverageError("Database contains no phases.")
        return self.info

    def _components(self) -> list[str]:
        components = [element.upper() for element in ELEMENTS]
        if "VA" in self.info.elements:
            components.append("VA")
        return components

    def _select_phases(self, requested: Sequence[str] | None) -> list[str]:
        if requested is None:
            candidates = list(self.available_phases)
        else:
            candidates = [str(phase) for phase in requested]
            missing = set(candidates) - set(self.available_phases)
            if missing:
                raise DatabaseCoverageError(
                    "Requested phase(s) absent from database: " + ", ".join(sorted(missing))
                )
        try:
            from pycalphad.core.utils import filter_phases

            selected = list(filter_phases(self.dbf, self._components(), candidates))
        except ImportError:  # pragma: no cover - protected by constructor import
            selected = candidates
        if not selected:
            raise DatabaseCoverageError("No database phases are valid for Cu-Al-Ni components.")
        return selected

    def composition_conditions(
        self, composition: AlloyComposition, *, dependent_component: str = "CU"
    ) -> dict[Any, float]:
        dependent = dependent_component.strip().upper()
        physical = {element.upper(): element for element in ELEMENTS}
        if dependent not in physical:
            raise ValueError("dependent_component must be CU, AL, or NI")
        return {
            self.variables.X(component): float(composition.mole_fraction[title])
            for component, title in physical.items()
            if component != dependent
        }

    def equilibrium(
        self,
        composition: AlloyComposition,
        temperatures_K: float | Iterable[float],
        *,
        pressure_Pa: float = 101_325.0,
        phases: Sequence[str] | None = None,
        dependent_component: str = "CU",
        output: str | Sequence[str] | None = None,
        calc_opts: Mapping[str, Any] | None = None,
    ) -> EquilibriumResult:
        """Minimize database Gibbs energies and return molar phase fractions."""

        temperatures = _temperatures(temperatures_K)
        pressure = float(pressure_Pa)
        if not np.isfinite(pressure) or pressure <= 0:
            raise ValueError("pressure_Pa must be finite and positive.")
        selected_phases = self._select_phases(phases)
        conditions: dict[Any, Any] = self.composition_conditions(
            composition, dependent_component=dependent_component
        )
        conditions.update(
            {
                self.variables.T: temperatures,
                self.variables.P: pressure,
                self.variables.N: 1.0,
            }
        )
        try:
            dataset = self._equilibrium(
                self.dbf,
                self._components(),
                selected_phases,
                conditions,
                output=output,
                calc_opts=dict(calc_opts or {}),
            )
        except Exception as exc:
            raise RuntimeError(
                "CALPHAD equilibrium failed. Check the database validity range, phase "
                f"selection, and composition. pycalphad reported: {exc}"
            ) from exc
        fractions = _tidy_phase_fractions(dataset, temperatures)
        database_provenance = Provenance(
            title=self.path.name,
            citation=self.database_citation,
            identifier=str(self.path),
            sha256=self.info.sha256,
        )
        conditions_for_export = {
            "temperature_K": temperatures.tolist(),
            "pressure_Pa": pressure,
            "mole_fraction": dict(composition.mole_fraction),
            "dependent_component": dependent_component.upper(),
            "phases": selected_phases,
        }
        return EquilibriumResult(
            phase_fractions=fractions,
            raw_dataset=dataset,
            database=self.info,
            conditions=conditions_for_export,
            metadata=ScientificMetadata(
                classification=ResultClassification.PHYSICS_CALCULATED,
                confidence=(
                    "database-dependent; numerical equilibrium solved by pycalphad, "
                    "but database assessment quality is not independently certified"
                ),
                limitations=(
                    "Assumes homogeneous bulk equilibrium at the specified pressure.",
                    "Results are valid only within the supplied database's assessed composition/temperature range.",
                    "Martensitic transformations require explicit metastable phase models and are not inferred from phase names.",
                    "Molar phase fractions are not automatically mass or volume fractions.",
                ),
                provenance=(PYCALPHAD_REFERENCE, database_provenance),
            ),
        )


def _tidy_phase_fractions(dataset: Any, requested_temperatures: np.ndarray) -> pd.DataFrame:
    """Convert pycalphad's vertex representation to one row per T and phase."""

    if "Phase" not in dataset or "NP" not in dataset:
        raise RuntimeError("pycalphad result lacks required Phase or NP variables.")
    phase_da = dataset["Phase"]
    fraction_da = dataset["NP"]
    select = {dim: 0 for dim in phase_da.dims if dim not in {"T", "vertex"}}
    if select:
        phase_da = phase_da.isel(select)
        fraction_da = fraction_da.isel(select)
    if "T" not in phase_da.dims:
        phase_da = phase_da.expand_dims(T=requested_temperatures[:1])
        fraction_da = fraction_da.expand_dims(T=requested_temperatures[:1])
    if "vertex" not in phase_da.dims:
        phase_da = phase_da.expand_dims(vertex=[0])
        fraction_da = fraction_da.expand_dims(vertex=[0])
    phase_values = np.asarray(phase_da.transpose("T", "vertex").values)
    fraction_values = np.asarray(fraction_da.transpose("T", "vertex").values, dtype=float)
    temperatures = np.asarray(phase_da.coords["T"].values, dtype=float).reshape(-1)
    rows: list[dict[str, Any]] = []
    for index, temperature in enumerate(temperatures):
        by_phase: dict[str, float] = {}
        for phase_raw, fraction in zip(
            phase_values[index], fraction_values[index], strict=True
        ):
            if isinstance(phase_raw, bytes):
                phase = phase_raw.decode(errors="replace").strip()
            else:
                phase = str(phase_raw).strip()
            if not phase or not np.isfinite(fraction):
                continue
            by_phase[phase] = by_phase.get(phase, 0.0) + float(fraction)
        for phase, fraction in sorted(by_phase.items()):
            rows.append(
                {
                    "temperature_K": float(temperature),
                    "temperature_C": float(temperature - 273.15),
                    "phase": phase,
                    "molar_phase_fraction": fraction,
                }
            )
    return pd.DataFrame(
        rows,
        columns=["temperature_K", "temperature_C", "phase", "molar_phase_fraction"],
    )
