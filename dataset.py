"""Dataset loading, provenance, and scientific validity checks.

The platform deliberately ships without fabricated training observations.  A model
can only be fitted from a table accompanied by explicit provenance metadata.  This
module makes that policy executable rather than relying on a UI warning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

COMPOSITION_COLUMNS = ("Cu_wt_pct", "Al_wt_pct", "Ni_wt_pct")
TRANSFORMATION_TARGETS = ("Ms_C", "Mf_C", "As_C", "Af_C")
PROPERTY_TARGETS = ("hardness_HV", "yield_strength_MPa", "uts_MPa")
PROCESS_COLUMNS = (
    "solution_treatment_C",
    "solution_treatment_min",
    "quench_medium",
    "aging_C",
    "aging_min",
    "grain_size_um",
    "beta_fraction",
    "martensite_fraction",
    "gamma2_fraction",
)
SUPPORTED_TARGETS = TRANSFORMATION_TARGETS + PROPERTY_TARGETS


class DatasetValidationError(ValueError):
    """Raised when data cannot support an auditable scientific model."""


@dataclass(frozen=True)
class DatasetProvenance:
    """Auditable origin and usage constraints for an experimental dataset."""

    title: str
    source: str
    license: str
    measurement_method: str
    units: Mapping[str, str]
    retrieved_at: str
    citation: str | None = None
    doi: str | None = None
    url: str | None = None
    curator: str | None = None
    notes: str | None = None
    synthetic: bool = False
    allow_training: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DatasetProvenance":
        required = (
            "title",
            "source",
            "license",
            "measurement_method",
            "units",
            "retrieved_at",
        )
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise DatasetValidationError(
                "Provenance metadata is incomplete; missing: " + ", ".join(missing)
            )
        if not isinstance(raw["units"], Mapping):
            raise DatasetValidationError("provenance.units must map column names to units")
        try:
            date.fromisoformat(str(raw["retrieved_at"])[:10])
        except ValueError as exc:
            raise DatasetValidationError("retrieved_at must use ISO-8601 format") from exc
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in raw.items() if key in known})

    def training_authorized(self, *, allow_synthetic: bool = False) -> bool:
        return bool(self.allow_training and (allow_synthetic or not self.synthetic))


@dataclass
class DatasetBundle:
    """Data and provenance kept together through the model lifecycle."""

    frame: pd.DataFrame
    provenance: DatasetProvenance
    sha256: str
    warnings: list[str] = field(default_factory=list)

    def require_training_authorization(self, *, allow_synthetic: bool = False) -> None:
        if not self.provenance.allow_training:
            raise DatasetValidationError("Dataset metadata explicitly disallows training")
        if self.provenance.synthetic and not allow_synthetic:
            raise DatasetValidationError(
                "Synthetic/example data are disabled for scientific predictions. "
                "Pass allow_synthetic=True only for an explicitly labelled software test."
            )

    def available_targets(self) -> list[str]:
        return [
            column
            for column in SUPPORTED_TARGETS
            if column in self.frame and self.frame[column].notna().sum() > 0
        ]

    def training_frame(
        self,
        target: str,
        features: Sequence[str] | None = None,
        *,
        allow_synthetic: bool = False,
        minimum_rows: int = 8,
    ) -> tuple[pd.DataFrame, pd.Series]:
        self.require_training_authorization(allow_synthetic=allow_synthetic)
        if target not in SUPPORTED_TARGETS:
            raise DatasetValidationError(f"Unsupported target {target!r}")
        if target not in self.frame:
            raise DatasetValidationError(f"Target {target!r} is absent")
        selected = list(features or infer_feature_columns(self.frame))
        if not selected:
            raise DatasetValidationError("No valid predictor columns were found")
        missing = [name for name in selected if name not in self.frame]
        if missing:
            raise DatasetValidationError("Missing features: " + ", ".join(missing))
        if target not in self.provenance.units:
            raise DatasetValidationError(
                f"Provenance must declare the measurement unit for target {target!r}"
            )
        numeric_without_units = [
            name
            for name in selected
            if name != "quench_medium" and name not in self.provenance.units
        ]
        if numeric_without_units:
            raise DatasetValidationError(
                "Provenance must declare feature units for: " + ", ".join(numeric_without_units)
            )
        usable = self.frame[selected + [target]].dropna(subset=[target]).copy()
        if len(usable) < minimum_rows:
            raise DatasetValidationError(
                f"{target} has {len(usable)} usable rows; at least {minimum_rows} are required"
            )
        if usable[target].nunique(dropna=True) < 2:
            raise DatasetValidationError(f"{target} contains no measurable variation")
        return usable[selected], usable[target].astype(float)

    def audit_record(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "rows": int(len(self.frame)),
            "columns": list(self.frame.columns),
            "provenance": asdict(self.provenance),
            "warnings": self.warnings,
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_metadata(metadata: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(metadata, Mapping):
        return metadata
    path = Path(metadata)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency-message path
            raise DatasetValidationError("PyYAML is required for YAML metadata") from exc
        parsed = yaml.safe_load(raw)
    else:
        parsed = json.loads(raw)
    if not isinstance(parsed, Mapping):
        raise DatasetValidationError("Metadata root must be an object")
    return parsed


def _validate_composition(frame: pd.DataFrame, warnings: list[str]) -> None:
    present = [name for name in COMPOSITION_COLUMNS if name in frame]
    if len(present) != len(COMPOSITION_COLUMNS):
        raise DatasetValidationError(
            "Training data require Cu_wt_pct, Al_wt_pct, and Ni_wt_pct"
        )
    values = frame[list(COMPOSITION_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if values.isna().any(axis=None):
        rows = (values.isna().any(axis=1)).to_numpy().nonzero()[0].tolist()
        raise DatasetValidationError(f"Non-numeric composition in zero-based rows {rows[:10]}")
    if ((values < 0) | (values > 100)).any(axis=None):
        raise DatasetValidationError("Composition values must lie in [0, 100] wt%")
    totals = values.sum(axis=1)
    bad = ~np.isclose(totals, 100.0, atol=0.5, rtol=0.0)
    if bad.any():
        rows = bad.to_numpy().nonzero()[0].tolist()
        raise DatasetValidationError(
            "Cu-Al-Ni totals must be 100 +/- 0.5 wt%; invalid zero-based rows "
            + str(rows[:10])
        )
    if not np.allclose(totals, 100.0, atol=1e-8, rtol=0.0):
        warnings.append("Composition totals are within tolerance but are not exactly 100 wt%")


def load_dataset(
    csv_path: str | Path,
    metadata: str | Path | Mapping[str, Any],
    *,
    allow_synthetic: bool = False,
) -> DatasetBundle:
    """Load a CSV only when its provenance and physical fields pass validation."""

    path = Path(csv_path)
    payload = path.read_bytes()
    frame = pd.read_csv(path)
    provenance = DatasetProvenance.from_mapping(_read_metadata(metadata))
    warnings: list[str] = []
    _validate_composition(frame, warnings)
    for column, unit in provenance.units.items():
        if column not in frame.columns:
            warnings.append(f"Unit metadata refers to absent column {column!r}")
        if not str(unit).strip():
            raise DatasetValidationError(f"Empty unit for {column!r}")
    bundle = DatasetBundle(frame=frame, provenance=provenance, sha256=_sha256_bytes(payload), warnings=warnings)
    if provenance.synthetic and not allow_synthetic:
        warnings.append("Synthetic/example dataset: scientific model training is disabled")
    return bundle


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return supported features that exist, excluding leakage-prone targets."""

    ordered = list(COMPOSITION_COLUMNS + PROCESS_COLUMNS)
    return [name for name in ordered if name in frame.columns]


def combine_datasets(bundles: Iterable[DatasetBundle]) -> DatasetBundle:
    """Combine auditable datasets while retaining source IDs for grouped CV."""

    bundles = list(bundles)
    if not bundles:
        raise DatasetValidationError("At least one dataset is required")
    frames: list[pd.DataFrame] = []
    for bundle in bundles:
        part = bundle.frame.copy()
        part["source_id"] = bundle.sha256[:16]
        frames.append(part)
    manifest = [bundle.audit_record() for bundle in bundles]
    combined_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    provenance = DatasetProvenance(
        title="Combined Cu-Al-Ni experimental datasets",
        source="; ".join(bundle.provenance.title for bundle in bundles),
        license="; ".join(sorted({bundle.provenance.license for bundle in bundles})),
        measurement_method="Multiple; see component provenance manifest",
        units={
            key: value
            for bundle in bundles
            for key, value in bundle.provenance.units.items()
        },
        retrieved_at=date.today().isoformat(),
        notes=json.dumps(manifest, sort_keys=True),
        synthetic=any(bundle.provenance.synthetic for bundle in bundles),
        allow_training=all(bundle.provenance.allow_training for bundle in bundles),
    )
    warnings = [warning for bundle in bundles for warning in bundle.warnings]
    return DatasetBundle(
        pd.concat(frames, ignore_index=True, sort=False),
        provenance,
        _sha256_bytes(combined_bytes),
        warnings,
    )
