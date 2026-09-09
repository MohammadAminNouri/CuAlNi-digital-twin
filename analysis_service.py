"""Scientific-module orchestration for the Streamlit application.

Adapters in this module do not contain fallback scientific correlations.  A failed
database/model/import produces an explicit unavailable result, preserving the
platform rule that missing science must never be replaced with plausible-looking
numbers.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict, dataclass, field, is_dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

ProgressCallback = Callable[[str, float], None]


@dataclass
class AnalysisInputs:
    composition_wt_percent: Mapping[str, float]
    e_a_valence_map: Mapping[str, float] | None = None
    e_a_convention_name: str = "user supplied"
    e_a_citation: str | None = None
    temperature_min_K: float = 273.15
    temperature_max_K: float = 1473.15
    temperature_steps: int = 121
    pressure_Pa: float = 101325.0
    tdb_bytes: bytes | None = None
    tdb_filename: str | None = None
    tdb_citation: str | None = None
    run_scheil: bool = False
    scheil_start_temperature_K: float | None = None
    scheil_step_temperature_K: float = 2.0
    run_phase_map: bool = False
    phase_map_temperature_K: float | None = None
    phase_map_divisions: int = 12
    structure_files: Mapping[str, bytes] = field(default_factory=dict)
    structure_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    xrd_wavelength_angstrom: float = 1.5406
    ml_dataset_bytes: bytes | None = None
    ml_metadata: Mapping[str, Any] | None = None
    processing: Mapping[str, Any] = field(default_factory=dict)
    experimental_files: Mapping[str, bytes] = field(default_factory=dict)
    experimental_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    am_normalization_profile: Mapping[str, Any] | None = None


def _serializable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _serializable(value.to_dict())
    if hasattr(value, "as_dict"):
        return _serializable(value.as_dict())
    if is_dataclass(value):
        return _serializable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "tolist"):
        return _serializable(value.tolist())
    if hasattr(value, "__dict__"):
        return _serializable(vars(value))
    return str(value)


def _classification(value: Any, default: str) -> str:
    aliases = {
        "experimental_data_analysis": "experimental_measurement",
        "physics-based": "physics_calculated",
        "ml_prediction": "machine_learning_prediction",
        "screening": "engineering_screening",
    }
    candidate = str(value or default)
    candidate = aliases.get(candidate, candidate)
    allowed = {
        "physics_calculated",
        "machine_learning_prediction",
        "engineering_screening",
        "experimental_measurement",
        "unavailable",
    }
    return candidate if candidate in allowed else default


def _available_result(
    value: Any,
    *,
    classification: str,
    provenance: str,
    default_limitations: tuple[str, ...],
) -> dict[str, Any]:
    mapping = _serializable(value)
    if not isinstance(mapping, dict):
        mapping = {"value": mapping}
    metadata = mapping.get("metadata") if isinstance(mapping.get("metadata"), Mapping) else {}
    metadata_provenance = metadata.get("provenance") if isinstance(metadata, Mapping) else None
    if metadata_provenance:
        provenance_parts = []
        for item in metadata_provenance:
            if isinstance(item, Mapping):
                title = item.get("title", "source")
                citation = item.get("citation")
                provenance_parts.append(f"{title}: {citation}" if citation else str(title))
            else:
                provenance_parts.append(str(item))
        provenance = "; ".join(provenance_parts)
    mapping["classification"] = _classification(
        mapping.get(
            "classification",
            mapping.get("result_classification", metadata.get("classification") if isinstance(metadata, Mapping) else None),
        ),
        classification,
    )
    mapping.setdefault(
        "confidence",
        metadata.get("confidence", "Not quantified by the underlying method")
        if isinstance(metadata, Mapping)
        else "Not quantified by the underlying method",
    )
    mapping.setdefault("provenance", provenance)
    mapping.setdefault(
        "limitations",
        list(metadata.get("limitations") or default_limitations)
        if isinstance(metadata, Mapping)
        else list(default_limitations),
    )
    method_status = mapping.get("status")
    if method_status not in (None, "available"):
        mapping["method_status"] = method_status
    # ``status`` in the report envelope describes whether the method executed.
    # Method-specific states such as ``partial`` remain in ``method_status``.
    mapping["status"] = "available"
    return mapping


def _unavailable(reason: str, *, provenance: str = "Not executed") -> dict[str, Any]:
    return {
        "status": "unavailable",
        "classification": "unavailable",
        "confidence": "Not applicable",
        "provenance": provenance,
        "limitations": [reason],
        "reason": reason,
    }


def _temp_file(payload: bytes, suffix: str):
    """Create a closed Windows-compatible temporary file context."""

    class _TemporaryPath:
        path: Path | None = None

        def __enter__(self) -> Path:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as stream:
                stream.write(payload)
                self.path = Path(stream.name)
            return self.path

        def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
            if self.path is not None:
                try:
                    self.path.unlink(missing_ok=True)
                except OSError:
                    pass

    return _TemporaryPath()


def _run_composition(
    composition: Mapping[str, float],
    *,
    e_a_valence_map: Mapping[str, float] | None = None,
    e_a_convention_name: str = "user supplied",
    e_a_citation: str | None = None,
) -> tuple[dict[str, Any], Any | None]:
    try:
        from composition.converter import AlloyComposition
        from composition.descriptors import compute_descriptors
    except ImportError as exc:
        return _unavailable(f"Composition engine is not installed: {exc}", provenance="Python import"), None
    try:
        alloy = AlloyComposition.from_wt_percent(composition)
        e_a_source = None
        if e_a_valence_map is not None:
            if not e_a_citation:
                raise ValueError("An effective-valence citation is required to calculate Hume-Rothery e/a.")
            from scientific import Provenance

            e_a_source = Provenance(
                title=e_a_convention_name,
                citation=e_a_citation,
            )
        descriptors = compute_descriptors(
            alloy,
            e_a_valence_map=e_a_valence_map,
            e_a_convention_name=e_a_convention_name,
            e_a_source=e_a_source,
        )
        payload = {
            "input_wt_percent": dict(composition),
            "weight_percent": _serializable(getattr(alloy, "weight_percent", composition)),
            "atomic_percent": _serializable(getattr(alloy, "atomic_percent", None)),
            "mole_fraction": _serializable(getattr(alloy, "mole_fraction", None)),
            "descriptors": _serializable(descriptors),
            "classification": "physics_calculated",
            "confidence": "Deterministic composition conversion; descriptor uncertainty follows input uncertainty",
            "provenance": "composition.converter.AlloyComposition and composition.descriptors.compute_descriptors",
            "limitations": [
                "Composition conversion assumes the atomic-weight constants documented by the composition module.",
                "Density and electronic descriptors, if present, are mixture-rule descriptors rather than measured bulk properties.",
            ],
            "status": "available",
        }
        return payload, alloy
    except Exception as exc:
        return _unavailable(
            f"Composition calculation failed: {type(exc).__name__}: {exc}",
            provenance="composition.converter / composition.descriptors",
        ), None


def _run_thermodynamics(inputs: AnalysisInputs, alloy: Any | None) -> dict[str, Any]:
    if inputs.tdb_bytes is None:
        return _unavailable(
            "Upload a licensed/open Cu-Al-Ni TDB containing compatible Cu, Al, and Ni phases to calculate equilibrium.",
            provenance="No thermodynamic database supplied",
        )
    if alloy is None:
        return _unavailable("A valid composition object is required before equilibrium can be calculated.")
    try:
        from thermodynamics.calphad import CalphadEngine
    except ImportError as exc:
        return _unavailable(f"CALPHAD engine is not installed: {exc}", provenance="Python import")
    suffix = Path(inputs.tdb_filename or "database.tdb").suffix or ".tdb"
    temperatures = np.linspace(inputs.temperature_min_K, inputs.temperature_max_K, inputs.temperature_steps)
    try:
        with _temp_file(inputs.tdb_bytes, suffix) as database_path:
            engine = CalphadEngine(database_path, database_citation=inputs.tdb_citation)
            validation = engine.validate_system() if hasattr(engine, "validate_system") else None
            result = engine.equilibrium(
                alloy,
                temperatures,
                pressure_Pa=inputs.pressure_Pa,
            )
            payload = _available_result(
                result,
                classification="physics_calculated",
                provenance=f"pycalphad equilibrium using user-supplied database {inputs.tdb_filename or 'database.tdb'}",
                default_limitations=(
                    "Equilibrium predictions are limited by the supplied TDB assessment and selected phases.",
                    "Metastable martensite and kinetic suppression may not be represented by an equilibrium calculation.",
                ),
            )
            if inputs.run_scheil:
                from thermodynamics.scheil import run_scheil

                start_temperature = inputs.scheil_start_temperature_K or inputs.temperature_max_K
                payload["scheil"] = _available_result(
                    run_scheil(
                        engine,
                        alloy,
                        start_temperature_K=start_temperature,
                        step_temperature_K=inputs.scheil_step_temperature_K,
                    ),
                    classification="physics_calculated",
                    provenance="pycalphad/scheil with the same user-supplied TDB",
                    default_limitations=(
                        "Scheil-Gulliver assumptions exclude solid-state diffusion and solute trapping.",
                    ),
                )
            else:
                payload["scheil"] = _unavailable(
                    "Scheil-Gulliver simulation was not requested for this run.",
                    provenance="User option",
                )
            if inputs.run_phase_map:
                from thermodynamics.phase_diagrams import ternary_isothermal_phase_map

                map_temperature = inputs.phase_map_temperature_K or float(
                    0.5 * (inputs.temperature_min_K + inputs.temperature_max_K)
                )
                payload["composition_phase_map"] = _available_result(
                    ternary_isothermal_phase_map(
                        engine,
                        temperature_K=map_temperature,
                        divisions=inputs.phase_map_divisions,
                        pressure_Pa=inputs.pressure_Pa,
                    ),
                    classification="physics_calculated",
                    provenance="pycalphad equilibrium sampled on a ternary mole-fraction grid",
                    default_limitations=(
                        "A sampled grid can miss narrow fields and is not a traced phase boundary.",
                    ),
                )
            else:
                payload["composition_phase_map"] = _unavailable(
                    "The optional ternary CALPHAD grid was not requested for this run.",
                    provenance="User option",
                )
        if validation is not None:
            payload["database_validation"] = _serializable(validation)
        payload.setdefault("temperature_K", temperatures.tolist())
        return payload
    except Exception as exc:
        return _unavailable(
            f"CALPHAD calculation failed: {type(exc).__name__}: {exc}",
            provenance=f"thermodynamics.calphad.CalphadEngine with {inputs.tdb_filename or 'uploaded TDB'}",
        )


def _run_crystallography(inputs: AnalysisInputs) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load user CIFs and keep incomplete literature records explicitly separate."""

    try:
        from crystallography import crystal_database as database
    except ImportError as exc:
        unavailable = _unavailable(f"Crystallography module is not installed: {exc}", provenance="Python import")
        return unavailable, _unavailable("No structure is available for diffraction simulation.")

    literature = database.list_literature_references()
    records: dict[str, Any] = {}
    raw_records: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for phase, cif_bytes in inputs.structure_files.items():
        metadata = dict(inputs.structure_metadata.get(phase, {}))
        try:
            record = database.load_cif_data(
                cif_bytes,
                source_filename=str(metadata.get("filename") or f"{phase}.cif"),
                name=phase,
                citation=metadata.get("citation"),
                source_url=metadata.get("source_url"),
            )
            serialized = _serializable(record)
            if isinstance(serialized, dict):
                serialized.setdefault("phase_name", phase)
                try:
                    from crystallography.symmetry import analyze_symmetry

                    serialized["symmetry"] = _serializable(analyze_symmetry(record.structure))
                    serialized["space_group"] = (
                        f"{serialized['symmetry']['international_symbol']} "
                        f"(No. {serialized['symmetry']['space_group_number']})"
                    )
                except Exception as symmetry_exc:
                    serialized["symmetry"] = _unavailable(
                        f"Symmetry analysis failed: {type(symmetry_exc).__name__}: {symmetry_exc}",
                        provenance="spglib",
                    )
            records[phase] = serialized
            raw_records[phase] = record
        except Exception as exc:
            errors[phase] = f"{type(exc).__name__}: {exc}"
    if not records:
        crystal_payload = _unavailable(
            "No complete CIF was supplied. Literature metadata are shown, but atomic coordinates are required for 3D visualization and structure-factor XRD.",
            provenance="crystallography.crystal_database literature registry",
        )
        crystal_payload["literature_references"] = literature
        crystal_payload["load_errors"] = errors
        return crystal_payload, _unavailable("Diffraction simulation requires a complete validated CIF.")
    payload = {
        "status": "available",
        "classification": "physics_calculated",
        "confidence": "Structure-record dependent; consult each record citation",
        "provenance": "crystallography.crystal_database citation-backed reference records",
        "limitations": [
            "Reference structures are representative crystallographic models and may not match composition, order, defects, or modulation of a specific specimen.",
            "Verify lattice parameters and site occupancies against the cited source before quantitative interpretation.",
        ],
        "structures": records,
        "literature_references": literature,
        "load_errors": errors,
    }
    xrd = _simulate_reference_xrd(
        raw_records or records,
        wavelength_angstrom=inputs.xrd_wavelength_angstrom,
    )
    return payload, xrd


def _simulate_reference_xrd(
    records: Mapping[str, Any], *, wavelength_angstrom: float = 1.5406
) -> dict[str, Any]:
    try:
        from crystallography.diffraction import simulate_powder_pattern
    except ImportError as exc:
        return _unavailable(f"Diffraction simulator is not installed: {exc}", provenance="Python import")
    patterns: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for phase, record in records.items():
        structure = record
        if isinstance(record, Mapping):
            structure = record.get("structure", record.get("cif", record))
        try:
            patterns[phase] = _serializable(
                simulate_powder_pattern(
                    structure,
                    phase_name=phase,
                    wavelength_angstrom=wavelength_angstrom,
                )
            )
        except Exception as exc:
            errors[phase] = f"{type(exc).__name__}: {exc}"
    if not patterns:
        return _unavailable(
            "Reference structures were loaded, but none were compatible with the diffraction simulator. "
            + "; ".join(f"{phase}: {error}" for phase, error in errors.items()),
            provenance="crystallography.diffraction.simulate_powder_pattern",
        )
    return {
        "status": "available",
        "classification": "physics_calculated",
        "confidence": "Ideal powder-pattern simulation",
        "provenance": (
            "crystallography.diffraction.simulate_powder_pattern; "
            f"user-selected wavelength {wavelength_angstrom:g} Å"
        ),
        "limitations": [
            "Ideal kinematic powder intensities do not include instrumental broadening, texture, strain, absorption, or specimen displacement.",
            "Phase matching scores are not phase probabilities unless explicitly calibrated by a separate model.",
        ],
        "wavelength_angstrom": wavelength_angstrom,
        "patterns": patterns,
        "simulation_errors": errors,
    }


def _run_ml(inputs: AnalysisInputs) -> dict[str, Any]:
    if inputs.ml_dataset_bytes is None or inputs.ml_metadata is None:
        return _unavailable(
            "Upload a provenance metadata JSON and compatible experimental CSV to train transformation/property models.",
            provenance="No authorized training dataset supplied",
        )
    try:
        import pandas as pd

        from machine_learning.dataset import PROPERTY_TARGETS, TRANSFORMATION_TARGETS, load_dataset
        from machine_learning.ensemble import PropertyModelSet, TransformationModelSet
        from machine_learning.shap_analysis import (
            permutation_importance_report,
            shap_importance,
        )
    except ImportError as exc:
        return _unavailable(f"ML subsystem is not installed: {exc}", provenance="Python import")
    try:
        with _temp_file(inputs.ml_dataset_bytes, ".csv") as dataset_path:
            bundle = load_dataset(dataset_path, inputs.ml_metadata)
            transformation_models = TransformationModelSet().fit(bundle)
            property_models = PropertyModelSet().fit(bundle)
        feature_values: dict[str, Any] = {
            "Cu_wt_pct": float(inputs.composition_wt_percent["Cu"]),
            "Al_wt_pct": float(inputs.composition_wt_percent["Al"]),
            "Ni_wt_pct": float(inputs.composition_wt_percent["Ni"]),
            **dict(inputs.processing),
        }
        predictions: dict[str, Any] = {}
        validations: dict[str, Any] = {}
        explanations: dict[str, Any] = {}
        property_maps: dict[str, Any] = {}
        unavailable = {
            **transformation_models.unavailable_,
            **property_models.unavailable_,
        }
        fitted_models = {
            **transformation_models.models_,
            **property_models.models_,
        }
        for target, model in fitted_models.items():
            missing = [feature for feature in model.features_ if feature not in feature_values]
            if missing:
                unavailable[target] = "Missing application inputs: " + ", ".join(missing)
                continue
            frame = pd.DataFrame(
                [{feature: feature_values[feature] for feature in model.features_}]
            )
            predictions[target] = model.predict(frame).to_dict()
            validations[target] = _serializable(
                model.validation_table().to_dict(orient="records")
            )

            # Explain one auditable constituent of the ensemble. SHAP is attempted
            # only when installed and compatible; the fallback is explicitly named
            # permutation importance and is never presented as SHAP.
            training_X, training_y = bundle.training_frame(target, model.features_)
            representative_name = max(model.weights_, key=model.weights_.get)
            representative = model.estimators_[representative_name]
            shap_error: str | None = None
            try:
                importance = shap_importance(representative, training_X)
            except Exception as exc:
                shap_error = f"{type(exc).__name__}: {exc}"
                importance = permutation_importance_report(
                    representative,
                    training_X,
                    training_y,
                    n_repeats=20,
                    random_state=model.random_state,
                )
            explanations[target] = {
                "status": "available",
                "classification": importance.classification,
                "method": importance.method,
                "representative_model": representative_name,
                "representative_model_weight": model.weights_[representative_name],
                "importance": importance.table().to_dict(orient="records"),
                "limitation": importance.limitation,
                "shap_unavailable_reason": shap_error,
            }

            if target in PROPERTY_TARGETS:
                property_map = _property_map_for_model(
                    model,
                    feature_values,
                    target=target,
                    unit=str(model.target_unit_ or "not documented"),
                )
                if property_map is not None:
                    property_maps[target] = property_map
        if not predictions:
            return _unavailable(
                "Models were evaluated, but no target could be predicted: "
                + "; ".join(f"{key}: {value}" for key, value in unavailable.items()),
                provenance=f"Dataset SHA-256 {bundle.sha256}",
            )
        transformation_predictions = {
            target: predictions[target]
            for target in TRANSFORMATION_TARGETS
            if target in predictions
        }
        property_predictions = {
            target: predictions[target]
            for target in PROPERTY_TARGETS
            if target in predictions
        }
        first_explanation = next(iter(explanations.values()), {})
        first_property_map = next(iter(property_maps.values()), None)
        scalar_predictions = {
            target: float(result["prediction"][0])
            for target, result in predictions.items()
            if isinstance(result.get("prediction"), list) and result["prediction"]
        }
        ordering_checks: dict[str, dict[str, Any]] = {}
        for label, lower_target, upper_target in (
            ("Mf <= Ms", "Mf_C", "Ms_C"),
            ("As <= Af", "As_C", "Af_C"),
        ):
            if lower_target in scalar_predictions and upper_target in scalar_predictions:
                ordering_checks[label] = {
                    "passes": scalar_predictions[lower_target] <= scalar_predictions[upper_target],
                    "lower_target_value_C": scalar_predictions[lower_target],
                    "upper_target_value_C": scalar_predictions[upper_target],
                }
        applicability_flags = [
            bool(item.get("within_training_ranges"))
            for result in predictions.values()
            for item in result.get("applicability", [])
        ]
        return {
            "status": "available",
            "classification": "machine_learning_prediction",
            "confidence": "Target-specific conformal intervals and cross-validation metrics are provided below",
            "provenance": f"Provenance-authorized dataset: {bundle.provenance.title}; SHA-256 {bundle.sha256}",
            "limitations": [
                "Predictions are empirical and valid only within the documented dataset domain and processing routes.",
                "Small or heterogeneous datasets can make cross-validation and conformal coverage estimates unstable.",
                "Transformation targets are fit independently; physical ordering constraints are flagged, not silently enforced.",
            ],
            "targets": predictions,
            "transformation_targets": transformation_predictions,
            "property_targets": property_predictions,
            "validation": validations,
            "explainability": explanations,
            "feature_importance": first_explanation.get("importance", []),
            "property_maps": property_maps,
            "property_map": first_property_map,
            "transformation_consistency": {
                "checks": ordering_checks,
                "all_available_checks_pass": (
                    all(item["passes"] for item in ordering_checks.values())
                    if ordering_checks
                    else None
                ),
                "action": (
                    "Predictions are displayed unchanged. A failed check requires model/data review."
                ),
            },
            "applicability_summary": {
                "all_predictions_within_univariate_training_ranges": (
                    all(applicability_flags) if applicability_flags else None
                ),
                "note": "Univariate range checks are not a multivariate convex-hull test.",
            },
            "model_status": {
                "classification": "machine_learning_prediction",
                "trained_targets": sorted(predictions),
                "unavailable_targets": unavailable,
                "transformation": transformation_models.status(),
                "properties": property_models.status(),
                "constraint_note": (
                    "Transformation targets are fitted independently. Mf <= Ms and As <= Af "
                    "must be checked; predictions are not silently reordered."
                ),
            },
            "dataset_audit": bundle.audit_record(),
        }
    except Exception as exc:
        return _unavailable(
            f"ML training/prediction failed: {type(exc).__name__}: {exc}",
            provenance="machine_learning.dataset / machine_learning.ensemble",
        )


def _property_map_for_model(
    model: Any,
    feature_values: Mapping[str, Any],
    *,
    target: str,
    unit: str,
    points_per_axis: int = 17,
) -> dict[str, Any] | None:
    """Predict an in-range ternary slice at the user-selected process state.

    The grid is restricted to the fitted model's one-dimensional training bounds.
    This is an applicability screen, not a convex-hull guarantee and not a phase
    diagram. The limitation is carried with the serialized map.
    """

    import pandas as pd

    composition_features = ("Cu_wt_pct", "Al_wt_pct", "Ni_wt_pct")
    if not all(name in model.features_ for name in composition_features):
        return None
    if model.domain_ is None or not all(
        name in model.domain_.numeric_bounds for name in composition_features
    ):
        return None
    if any(feature not in feature_values for feature in model.features_):
        return None
    cu_bounds = model.domain_.numeric_bounds["Cu_wt_pct"]
    al_bounds = model.domain_.numeric_bounds["Al_wt_pct"]
    ni_bounds = model.domain_.numeric_bounds["Ni_wt_pct"]
    rows: list[dict[str, Any]] = []
    for al in np.linspace(*al_bounds, points_per_axis):
        for ni in np.linspace(*ni_bounds, points_per_axis):
            cu = 100.0 - float(al) - float(ni)
            if not cu_bounds[0] <= cu <= cu_bounds[1]:
                continue
            row = {feature: feature_values[feature] for feature in model.features_}
            row.update(
                {
                    "Cu_wt_pct": cu,
                    "Al_wt_pct": float(al),
                    "Ni_wt_pct": float(ni),
                }
            )
            rows.append(row)
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=model.features_)
    batch = model.predict(frame)
    valid_indices = [
        index
        for index, item in enumerate(batch.applicability)
        if bool(item.get("within_training_ranges"))
    ]
    if not valid_indices:
        return None
    return {
        "classification": "machine_learning_prediction",
        "target": target,
        "property_name": target,
        "unit": unit,
        "compositions": [
            {
                "Cu": float(frame.iloc[index]["Cu_wt_pct"]),
                "Al": float(frame.iloc[index]["Al_wt_pct"]),
                "Ni": float(frame.iloc[index]["Ni_wt_pct"]),
            }
            for index in valid_indices
        ],
        "values": [float(batch.values[index]) for index in valid_indices],
        "interval_radius": float(batch.interval.residual_quantile),
        "fixed_features": {
            feature: feature_values[feature]
            for feature in model.features_
            if feature not in composition_features
        },
        "limitations": [
            "The map is an empirical model slice at the displayed fixed processing inputs.",
            "Points are inside per-feature training ranges, but that does not guarantee they lie inside the multivariate training-data convex hull.",
            "The map is not a CALPHAD phase diagram and must not be extrapolated beyond the displayed domain.",
        ],
    }


def _reference_peaks_from_xrd(xrd: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten calculated phase peak tables for experimental matching."""

    if xrd.get("status") != "available":
        return []
    patterns = xrd.get("patterns")
    if not isinstance(patterns, Mapping):
        return []
    reference_rows: list[dict[str, Any]] = []
    for phase, raw_pattern in patterns.items():
        pattern = _serializable(raw_pattern)
        if not isinstance(pattern, Mapping):
            continue
        peaks = pattern.get("peaks", pattern.get("peak_table", []))
        if not isinstance(peaks, list):
            continue
        for raw_peak in peaks:
            if not isinstance(raw_peak, Mapping):
                continue
            two_theta = raw_peak.get("two_theta_deg", raw_peak.get("two_theta"))
            try:
                two_theta_value = float(two_theta)
            except (TypeError, ValueError):
                continue
            row: dict[str, Any] = {
                "two_theta_deg": two_theta_value,
                "phase": str(raw_peak.get("phase") or phase),
            }
            hkl = raw_peak.get("hkl")
            if hkl is not None:
                row["hkl"] = str(hkl)
            spacing = raw_peak.get(
                "d_spacing_angstrom",
                raw_peak.get("d_angstrom", raw_peak.get("d_spacing")),
            )
            if spacing is not None:
                try:
                    row["d_spacing_angstrom"] = float(spacing)
                except (TypeError, ValueError):
                    pass
            intensity = raw_peak.get(
                "intensity",
                raw_peak.get("scaled_intensity", raw_peak.get("relative_intensity")),
            )
            if intensity is not None:
                try:
                    row["intensity"] = float(intensity)
                except (TypeError, ValueError):
                    pass
            reference_rows.append(row)
    return reference_rows


def _run_characterization(inputs: AnalysisInputs, reference_peaks: Any = None) -> dict[str, Any]:
    if not inputs.experimental_files:
        return _unavailable(
            "Upload XRD, EDS, DSC, or SEM data to run experimental characterization.",
            provenance="No experimental data supplied",
        )
    outputs: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for kind, payload in inputs.experimental_files.items():
        options = dict(inputs.experimental_options.get(kind, {}))
        try:
            if kind == "eds":
                from characterization.eds_analysis import analyze_eds

                result = analyze_eds(BytesIO(payload), target_wt_pct=inputs.composition_wt_percent, **options)
            elif kind == "xrd":
                from characterization.xrd_analysis import analyze_xrd

                options.pop("reference_peaks", None)
                result = analyze_xrd(BytesIO(payload), reference_peaks=reference_peaks, **options)
            elif kind == "dsc":
                from characterization.dsc_analysis import analyze_dsc

                result = analyze_dsc(BytesIO(payload), **options)
            elif kind == "sem":
                from characterization.sem_analysis import analyze_sem

                result = analyze_sem(BytesIO(payload), **options)
            else:
                errors[kind] = "Unsupported characterization file type"
                continue
            outputs[kind] = _available_result(
                result,
                classification="experimental_measurement",
                provenance=f"characterization.{kind}_analysis applied to user-uploaded data",
                default_limitations=("Interpretation is limited by the supplied instrument data and analysis parameters.",),
            )
        except Exception as exc:
            errors[kind] = f"{type(exc).__name__}: {exc}"
    if not outputs:
        return _unavailable(
            "No characterization pipeline completed: " + "; ".join(f"{key}: {value}" for key, value in errors.items()),
            provenance="characterization package",
        )
    return {
        "status": "available",
        "classification": "experimental_measurement",
        "confidence": "Technique-specific diagnostics reported per uploaded dataset",
        "provenance": "User-uploaded experimental data analyzed by characterization modules",
        "limitations": [
            "Instrument calibration, specimen history, and sampling representativeness must be reviewed by the user.",
            "Experimental analysis does not itself establish equilibrium phase fractions or causal process relationships.",
        ],
        "analyses": outputs,
        "errors": errors,
        "calculated_xrd_reference_peak_count": len(reference_peaks or []),
    }


def _run_am(inputs: AnalysisInputs, thermo: Mapping[str, Any]) -> dict[str, Any]:
    if inputs.am_normalization_profile is None:
        return _unavailable(
            "A documented AM normalization profile is required before a comparable 0-10 suitability score can be calculated.",
            provenance="No engineering-screening normalization profile supplied",
        )
    try:
        from additive_manufacturing.printability import ScreeningMetric, assess_printability
    except ImportError as exc:
        return _unavailable(f"AM screening module is not installed: {exc}", provenance="Python import")
    metrics = inputs.am_normalization_profile.get("metrics")
    profile = inputs.am_normalization_profile.get("profile", inputs.am_normalization_profile)
    if not isinstance(metrics, Mapping):
        return _unavailable(
            "The AM profile JSON must contain a documented `metrics` mapping; metrics are not inferred from missing thermodynamic outputs.",
            provenance="User-supplied AM profile",
        )
    try:
        parsed_metrics: dict[str, Any] = {}
        for name, raw in metrics.items():
            if isinstance(raw, Mapping):
                parsed_metrics[name] = ScreeningMetric(
                    value=float(raw["value"]),
                    unit=str(raw["unit"]),
                    classification=str(raw.get("classification", "user_supplied")),
                    provenance=str(raw.get("provenance", "User-supplied AM metric")),
                    uncertainty=(
                        float(raw["uncertainty"])
                        if raw.get("uncertainty") is not None
                        else None
                    ),
                )
            else:
                parsed_metrics[name] = float(raw)
        result = assess_printability(parsed_metrics, profile=profile)
        return _available_result(
            result,
            classification="engineering_screening",
            provenance="additive_manufacturing.printability.assess_printability with user-supplied normalization profile",
            default_limitations=(
                "The score is a transparent engineering screen, not a process qualification or build-success probability.",
                "Machine, powder, geometry, scan strategy, atmosphere, and defect inspection remain outside composition-only screening.",
            ),
        )
    except Exception as exc:
        return _unavailable(
            f"AM screening failed: {type(exc).__name__}: {exc}",
            provenance="additive_manufacturing.printability.assess_printability",
        )


def run_complete_analysis(
    inputs: AnalysisInputs,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run all available engines and return a report-ready result envelope."""

    callback = progress or (lambda _message, _value: None)
    callback("Converting composition and calculating descriptors", 0.08)
    composition, alloy = _run_composition(
        inputs.composition_wt_percent,
        e_a_valence_map=inputs.e_a_valence_map,
        e_a_convention_name=inputs.e_a_convention_name,
        e_a_citation=inputs.e_a_citation,
    )
    callback("Evaluating thermodynamic database", 0.24)
    thermodynamics = _run_thermodynamics(inputs, alloy)
    callback("Loading citation-backed crystal structures", 0.40)
    crystallography, xrd = _run_crystallography(inputs)
    callback("Training provenance-authorized predictive models", 0.58)
    ml = _run_ml(inputs)
    callback("Analyzing uploaded characterization data", 0.72)
    reference_peaks = _reference_peaks_from_xrd(xrd)
    characterization = _run_characterization(inputs, reference_peaks=reference_peaks or None)
    callback("Computing additive-manufacturing screen", 0.88)
    am = _run_am(inputs, thermodynamics)
    # Phase outputs are intentionally separate from equilibrium to avoid describing
    # an equilibrium state as a martensitic transformation prediction.
    try:
        from phases import phase_reference_catalog

        phase_catalog = phase_reference_catalog()
    except Exception as exc:
        phase_catalog = {"status": "unavailable", "reason": f"Phase registry unavailable: {exc}"}
    phases = {
        "status": thermodynamics.get("status", "unavailable"),
        "classification": thermodynamics.get("classification", "unavailable"),
        "confidence": thermodynamics.get("confidence", "Not applicable"),
        "provenance": thermodynamics.get("provenance", "Not executed"),
        "limitations": thermodynamics.get("limitations", ["No phase calculation is available."]),
        "equilibrium_result": thermodynamics if thermodynamics.get("status") == "available" else None,
        "phase_reference_catalog": phase_catalog,
        "risks": {
            "gamma2": "Requires a database-calculated γ₂ phase fraction or a cited screening profile.",
            "brittleness": "Requires measured properties or a cited engineering screening profile.",
            "segregation": "Requires Scheil/kinetic results or spatial experimental composition data.",
        },
    }
    callback("Analysis complete", 1.0)
    return {
        "schema_version": "1.0",
        "composition": composition,
        "thermodynamics": thermodynamics,
        "phases": phases,
        "crystallography": crystallography,
        "xrd": xrd,
        "ml": ml,
        "characterization": characterization,
        "am": am,
    }
