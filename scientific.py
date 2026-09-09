"""Shared scientific-result metadata and explicit provenance helpers.

The project deliberately separates numerical calculations from predictions and
engineering screens.  Every public result carries one of the classifications
defined here so that a UI or report cannot silently present a heuristic as a
thermodynamic calculation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class ResultClassification(str, Enum):
    """Permitted evidence classes in CuAlNi-DigitalTwin Pro."""

    PHYSICS_CALCULATED = "physics_calculated"
    MACHINE_LEARNING_PREDICTION = "machine_learning_prediction"
    ENGINEERING_SCREENING = "engineering_screening"
    REFERENCE_DATA = "reference_data"
    USER_SUPPLIED = "user_supplied"
    NOT_AVAILABLE = "not_available"


@dataclass(frozen=True)
class Provenance:
    """Trace one input, model, database, or literature source."""

    title: str
    citation: str
    url: str | None = None
    identifier: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class ScientificMetadata:
    """Metadata attached to every scientific output."""

    classification: ResultClassification
    confidence: str
    limitations: tuple[str, ...] = ()
    provenance: tuple[Provenance, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["classification"] = self.classification.value
        return data


@dataclass(frozen=True)
class ScientificValue:
    """A scalar or structured value together with units and scientific status."""

    value: Any
    unit: str | None
    metadata: ScientificMetadata
    name: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "details": dict(self.details),
            "metadata": self.metadata.to_dict(),
        }
