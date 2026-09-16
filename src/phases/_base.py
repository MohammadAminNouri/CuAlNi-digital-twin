"""Reference-only phase descriptions; never substitutes for CALPHAD or a CIF."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PhaseReference:
    key: str
    display_name: str
    temperature_role: str
    crystallographic_description: str
    space_group: str | None
    stacking_sequence: str | None
    formula: str | None
    source: str
    source_url: str
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "status": "reference_only",
                "classification": "physics_calculated",
                "diffraction_ready": False,
                "confidence": "literature phase description; specimen applicability not established",
                "provenance": self.source,
                "limitations": list(self.limitations),
            }
        )
        return payload

