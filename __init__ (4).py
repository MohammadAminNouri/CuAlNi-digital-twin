"""Database-gated CALPHAD equilibrium and solidification calculations."""

from .calphad import (
    CalphadDependencyError,
    CalphadEngine,
    DatabaseCoverageError,
    DatabaseInfo,
    EquilibriumResult,
)
from .phase_equilibrium import equilibrium_at_temperature, phase_fractions_vs_temperature
from .scheil import ScheilResult, run_scheil

__all__ = [
    "CalphadDependencyError",
    "CalphadEngine",
    "DatabaseCoverageError",
    "DatabaseInfo",
    "EquilibriumResult",
    "ScheilResult",
    "equilibrium_at_temperature",
    "phase_fractions_vs_temperature",
    "run_scheil",
]
