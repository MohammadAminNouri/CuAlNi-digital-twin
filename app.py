"""CuAlNi-DigitalTwin Pro Streamlit application."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.analysis_service import AnalysisInputs, run_complete_analysis  # noqa: E402
from reporting.report_generator import AnalysisReportGenerator, ReportGenerationError  # noqa: E402
from visualization.crystal_visualizer import crystal_structure_figure  # noqa: E402
from visualization.dashboard import (  # noqa: E402
    CLASSIFICATION_LABELS,
    STREAMLIT_CSS,
    am_suitability_radar,
    classification_badge,
    result_card_html,
)
from visualization.phase_plots import (  # noqa: E402
    composition_property_map,
    phase_fraction_temperature,
    phase_stability_map,
    ternary_composition_plot,
    ternary_phase_assemblage_map,
)
from visualization.xrd_plots import (  # noqa: E402
    transformation_temperature_chart,
    uncertainty_plot,
    xrd_pattern_plot,
)

st.set_page_config(
    page_title="CuAlNi-DigitalTwin Pro",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/pycalphad/pycalphad",
        "About": "CuAlNi-DigitalTwin Pro - provenance-aware Cu-Al-Ni alloy analysis.",
    },
)
st.markdown(STREAMLIT_CSS, unsafe_allow_html=True)


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return dict(value) if isinstance(value, Mapping) else {}


def _first(mapping: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        value: Any = mapping
        found = True
        for token in path.split("."):
            if isinstance(value, Mapping) and token in value:
                value = value[token]
            else:
                found = False
                break
        if found and value is not None:
            return value
    return default


def _status(section: Mapping[str, Any]) -> bool:
    return section.get("status") == "available" and section.get("classification") != "unavailable"


def _classification(section: Mapping[str, Any]) -> str:
    value = str(section.get("classification", section.get("result_classification", "unavailable")))
    aliases = {"experimental_data_analysis": "experimental_measurement"}
    value = aliases.get(value, value)
    return value if value in CLASSIFICATION_LABELS else "unavailable"


def _section_intro(section: Mapping[str, Any]) -> None:
    classification = _classification(section)
    confidence = section.get("confidence")
    st.markdown(classification_badge(classification, confidence), unsafe_allow_html=True)
    provenance = section.get("provenance")
    if provenance:
        st.caption(f"Provenance: {provenance}")
    if not _status(section):
        st.info(str(section.get("reason") or "This result is unavailable for the current analysis."), icon="ℹ️")


def _limitations(section: Mapping[str, Any], *, title: str = "Confidence and limitations") -> None:
    limitations = section.get("limitations") or []
    if isinstance(limitations, str):
        limitations = [limitations]
    with st.expander(title, expanded=False):
        st.markdown(classification_badge(_classification(section), section.get("confidence")), unsafe_allow_html=True)
        st.write(f"**Provenance:** {section.get('provenance', 'Not reported')}")
        if limitations:
            for item in limitations:
                st.markdown(f"- {item}")
        else:
            st.warning("No limitation statement was supplied; do not treat the result as validated.")


def _safe_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame([value])
    if isinstance(value, list):
        return pd.DataFrame(value)
    return pd.DataFrame()


def _display_mapping_table(mapping: Mapping[str, Any], *, value_header: str = "Value") -> None:
    rows = []
    for key, value in mapping.items():
        if key in {"classification", "result_classification", "confidence", "provenance", "limitations", "status", "figures"}:
            continue
        if isinstance(value, Mapping) and "value" in value:
            metadata = _mapping(value.get("metadata"))
            display = value.get("value")
            unit = value.get("unit")
            rows.append(
                {
                    "Result": str(key).replace("_", " ").title(),
                    value_header: "Unavailable" if display is None else str(display),
                    "Unit": unit or "",
                    "Classification": metadata.get("classification", "not reported"),
                }
            )
            continue
        if isinstance(value, (Mapping, list, tuple)):
            continue
        rows.append({"Result": str(key).replace("_", " ").title(), value_header: value})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _parse_json_upload(upload: Any, label: str) -> tuple[dict[str, Any] | None, str | None]:
    if upload is None:
        return None, None
    try:
        value = json.loads(upload.getvalue().decode("utf-8-sig"))
        if not isinstance(value, dict):
            return None, f"{label} must contain a JSON object."
        return value, None
    except Exception as exc:
        return None, f"Could not read {label}: {exc}"


def _build_sidebar() -> tuple[bool, AnalysisInputs | None]:
    st.sidebar.markdown("## Analysis setup")
    st.sidebar.caption("Uploads are processed by the server hosting this app for your session.")

    with st.sidebar.form("analysis_form", border=False):
        st.markdown("### Enter alloy composition")
        c1, c2, c3 = st.columns(3)
        cu = c1.number_input("Cu", min_value=0.0, max_value=100.0, value=81.4, step=0.1, format="%.4f", help="Copper concentration in weight percent.")
        al = c2.number_input("Al", min_value=0.0, max_value=100.0, value=14.3, step=0.1, format="%.4f", help="Aluminium concentration in weight percent.")
        ni = c3.number_input("Ni", min_value=0.0, max_value=100.0, value=4.3, step=0.1, format="%.4f", help="Nickel concentration in weight percent.")
        st.caption("Basis: wt%. Components must total 100.000 wt%.")

        with st.expander("Optional Hume-Rothery e/a convention", expanded=False):
            use_e_a = st.checkbox(
                "Calculate e/a from a cited effective-valence convention",
                help="Transition-metal effective valence is convention-dependent; no hidden Cu/Al/Ni assignment is used.",
            )
            e_a_cu = st.number_input("Effective valence: Cu", value=0.0, step=0.1)
            e_a_al = st.number_input("Effective valence: Al", value=0.0, step=0.1)
            e_a_ni = st.number_input("Effective valence: Ni", value=0.0, step=0.1)
            e_a_convention = st.text_input("Convention name", value="User-supplied effective-valence convention")
            e_a_citation = st.text_input("Convention citation / DOI / URL")

        with st.expander("Processing and microstructure", expanded=False):
            solution_c = st.number_input("Solution treatment (°C)", value=900.0, step=10.0)
            solution_min = st.number_input("Solution treatment time (min)", min_value=0.0, value=30.0, step=5.0)
            quench = st.selectbox("Quench medium", ["water", "oil", "air", "ice_brine", "other"])
            aging_c = st.number_input("Aging temperature (°C)", value=200.0, step=10.0)
            aging_min = st.number_input("Aging time (min)", min_value=0.0, value=0.0, step=10.0)
            grain_size = st.number_input("Grain size (µm, 0 = unknown)", min_value=0.0, value=0.0, step=5.0)

        with st.expander("Temperature range", expanded=False):
            t_min_c = st.number_input("Minimum temperature (°C)", value=0.0, step=10.0)
            t_max_c = st.number_input("Maximum temperature (°C)", value=1200.0, step=10.0)
            t_steps = st.number_input("Temperature points", min_value=20, max_value=1000, value=121, step=10)

        st.markdown("### Scientific inputs")
        tdb = st.file_uploader(
            "Thermodynamic database (.tdb)",
            type=["tdb"],
            help="No proprietary or unverified TDB is bundled. Upload a Cu-Al-Ni database you are licensed to use.",
        )
        tdb_citation = st.text_input(
            "Thermodynamic database citation",
            help="Recommended: assessment authors, year, DOI/version, and licensing identifier.",
        )
        with st.expander("CALPHAD and solidification options", expanded=False):
            run_scheil = st.checkbox(
                "Run Scheil-Gulliver solidification",
                help="Requires the optional scheil package and a database with a compatible LIQUID phase.",
            )
            scheil_start_c = st.number_input("Scheil start temperature (°C)", value=1200.0, step=10.0)
            scheil_step_k = st.number_input("Scheil temperature step (K)", min_value=0.1, value=2.0, step=0.5)
            run_phase_map = st.checkbox(
                "Calculate isothermal ternary phase grid",
                help="Potentially expensive: each grid point is an independent equilibrium calculation.",
            )
            phase_map_c = st.number_input("Phase-grid temperature (°C)", value=800.0, step=10.0)
            phase_map_divisions = st.slider("Phase-grid divisions", 4, 30, 12)

        cif_files = st.file_uploader(
            "Crystal structures (.cif; one or more)",
            type=["cif"],
            accept_multiple_files=True,
            help="Complete, cited CIFs enable 3D structure and structure-factor XRD. No atomic sites are guessed.",
        )
        cif_metadata_file = st.file_uploader(
            "CIF citation metadata (.json, optional)",
            type=["json"],
            key="cif_metadata",
            help="Map each CIF filename or stem to phase, citation, and source_url.",
        )
        ml_csv = st.file_uploader(
            "ML training data (.csv)",
            type=["csv"],
            help="Requires composition, applicable process fields, at least one target, and separate provenance metadata.",
        )
        ml_metadata_file = st.file_uploader("ML provenance metadata (.json)", type=["json"], key="ml_metadata")

        with st.expander("Experimental characterization", expanded=False):
            eds = st.file_uploader("EDS table", type=["csv", "txt"], key="eds_file")
            xrd = st.file_uploader("XRD pattern", type=["csv", "txt", "xy"], key="xrd_file")
            dsc = st.file_uploader("DSC curve", type=["csv", "txt"], key="dsc_file")
            sem = st.file_uploader("SEM micrograph", type=["png", "jpg", "jpeg", "tif", "tiff"], key="sem_file")
            eds_voltage = st.number_input("EDS accelerating voltage (kV, 0 = omit)", min_value=0.0, value=0.0, step=1.0)
            eds_density = st.number_input(
                "Specimen density for EDS range model (g/cm³, 0 = omit)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                help="Must be independently justified; the composition mixture estimate is not inserted automatically.",
            )
            xrd_wavelength = st.number_input("XRD wavelength (Å)", min_value=0.1, value=1.5406, format="%.5f")

        with st.expander("AM screening profile", expanded=False):
            am_profile_file = st.file_uploader(
                "Documented normalization profile (.json)",
                type=["json"],
                key="am_profile",
                help="A score is not emitted without explicit metric definitions, directions, bounds, weights, and provenance.",
            )

        submitted = st.form_submit_button("RUN COMPLETE ANALYSIS", type="primary", width="stretch")

    if not submitted:
        return False, None

    total = cu + al + ni
    if not np.isclose(total, 100.0, atol=1e-6, rtol=0.0):
        st.sidebar.error(f"Composition totals {total:.6f} wt%. Adjust Cu, Al, and Ni to exactly 100 wt%.")
        return True, None
    if t_max_c <= t_min_c:
        st.sidebar.error("Maximum temperature must exceed minimum temperature.")
        return True, None
    if t_min_c <= -273.15 or (run_scheil and scheil_start_c <= -273.15) or (run_phase_map and phase_map_c <= -273.15):
        st.sidebar.error("Calculation temperatures must be above absolute zero (−273.15 °C).")
        return True, None
    if use_e_a and not e_a_citation.strip():
        st.sidebar.error("A citation is required when the optional Hume-Rothery e/a convention is enabled.")
        return True, None
    metadata, metadata_error = _parse_json_upload(ml_metadata_file, "ML provenance metadata")
    profile, profile_error = _parse_json_upload(am_profile_file, "AM screening profile")
    cif_metadata, cif_metadata_error = _parse_json_upload(cif_metadata_file, "CIF citation metadata")
    if metadata_error or profile_error or cif_metadata_error:
        st.sidebar.error(metadata_error or profile_error or cif_metadata_error)
        return True, None

    experimental_files = {
        key: upload.getvalue()
        for key, upload in {"eds": eds, "xrd": xrd, "dsc": dsc, "sem": sem}.items()
        if upload is not None
    }
    experimental_options: dict[str, dict[str, Any]] = {
        "xrd": {"wavelength_angstrom": xrd_wavelength},
    }
    if eds_voltage > 0 and eds_density > 0:
        experimental_options["eds"] = {
            "accelerating_voltage_kV": eds_voltage,
            "density_g_cm3": eds_density,
        }

    structure_files: dict[str, bytes] = {}
    structure_metadata: dict[str, dict[str, Any]] = {}
    citation_map = cif_metadata or {}
    for upload in cif_files or []:
        raw_meta = citation_map.get(upload.name, citation_map.get(Path(upload.name).stem, {}))
        raw_meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        phase_label = str(raw_meta.get("phase") or Path(upload.name).stem)
        if phase_label in structure_files:
            phase_label = f"{phase_label} ({upload.name})"
        structure_files[phase_label] = upload.getvalue()
        structure_metadata[phase_label] = {
            "filename": upload.name,
            "citation": raw_meta.get("citation"),
            "source_url": raw_meta.get("source_url"),
        }

    processing: dict[str, Any] = {
        "solution_treatment_C": solution_c,
        "solution_treatment_min": solution_min,
        "quench_medium": quench,
        "aging_C": aging_c,
        "aging_min": aging_min,
    }
    if grain_size > 0:
        processing["grain_size_um"] = grain_size

    return True, AnalysisInputs(
        composition_wt_percent={"Cu": cu, "Al": al, "Ni": ni},
        e_a_valence_map=(
            {"Cu": e_a_cu, "Al": e_a_al, "Ni": e_a_ni} if use_e_a else None
        ),
        e_a_convention_name=e_a_convention,
        e_a_citation=e_a_citation.strip() or None,
        temperature_min_K=t_min_c + 273.15,
        temperature_max_K=t_max_c + 273.15,
        temperature_steps=int(t_steps),
        tdb_bytes=tdb.getvalue() if tdb is not None else None,
        tdb_filename=tdb.name if tdb is not None else None,
        tdb_citation=tdb_citation.strip() or None,
        run_scheil=run_scheil,
        scheil_start_temperature_K=scheil_start_c + 273.15,
        scheil_step_temperature_K=float(scheil_step_k),
        run_phase_map=run_phase_map,
        phase_map_temperature_K=phase_map_c + 273.15,
        phase_map_divisions=int(phase_map_divisions),
        structure_files=structure_files,
        structure_metadata=structure_metadata,
        xrd_wavelength_angstrom=xrd_wavelength,
        ml_dataset_bytes=ml_csv.getvalue() if ml_csv is not None else None,
        ml_metadata=metadata,
        processing=processing,
        experimental_files=experimental_files,
        experimental_options=experimental_options,
        am_normalization_profile=profile,
    )


def _hero() -> None:
    st.markdown(
        """
        <section class="dt-hero">
          <div class="dt-kicker">COMPUTATIONAL ALLOY DESIGN · CHARACTERIZATION · LPBF SCREENING</div>
          <h1>CuAlNi-DigitalTwin Pro</h1>
          <p>A traceable workspace for Cu-Al-Ni composition conversion, CALPHAD, crystallography,
          diffraction, empirical property models, experimental comparison, and additive-manufacturing assessment.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _empty_landing() -> None:
    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col1:
        st.markdown("### Start with composition")
        st.write(
            "The primary alloy **Cu-14.3Al-4.3Ni wt%** is preloaded. Adjust Cu, Al, and Ni in the sidebar, "
            "add any databases or experimental files you are authorized to use, then run the complete analysis."
        )
        st.info(
            "A thermodynamic database is intentionally not bundled. Without a compatible TDB, CALPHAD results "
            "remain unavailable; the app does not substitute an empirical phase diagram.",
            icon="🔒",
        )
    with col2:
        st.markdown("### Evidence classes")
        for key in ("physics_calculated", "machine_learning_prediction", "engineering_screening", "experimental_measurement"):
            st.markdown(classification_badge(key), unsafe_allow_html=True)
        st.caption("Every result retains classification, confidence, provenance, and limitations in screen and exports.")
    with col3:
        st.markdown("### Analysis route")
        st.markdown("Composition → structure → phases → transformation → properties → LPBF screen")
        st.caption("Unavailable inputs stop only the dependent module; other validated calculations can continue.")


def _composition_tab(analysis: Mapping[str, Any], composition_input: Mapping[str, float]) -> tuple[Any, Any]:
    section = _mapping(analysis.get("composition"))
    _section_intro(section)
    weight = _mapping(_first(section, "weight_percent", "input_wt_percent", default=composition_input))
    atomic = _mapping(section.get("atomic_percent"))
    mole = _mapping(section.get("mole_fraction"))
    descriptors = _mapping(section.get("descriptors"))
    metric_cols = st.columns(4)
    metric_cols[0].metric("Cu", f"{float(weight.get('Cu', composition_input['Cu'])):.3f} wt%")
    metric_cols[1].metric("Al", f"{float(weight.get('Al', composition_input['Al'])):.3f} wt%")
    metric_cols[2].metric("Ni", f"{float(weight.get('Ni', composition_input['Ni'])):.3f} wt%")
    metric_cols[3].metric("Basis", "Weight percent")
    ternary = ternary_composition_plot(weight or composition_input, basis="wt%")
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(ternary, width="stretch", config={"displaylogo": False})
    with right:
        st.markdown("#### Composition bases")
        table_rows = []
        for element in ("Cu", "Al", "Ni"):
            table_rows.append(
                {
                    "Element": element,
                    "wt%": weight.get(element),
                    "at%": atomic.get(element),
                    "mole fraction": mole.get(element),
                }
            )
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, width="stretch")
        st.markdown("#### Calculated descriptors")
        if descriptors:
            _display_mapping_table(descriptors)
        else:
            st.info("Descriptor output is unavailable.")
    _limitations(section)
    return ternary, section


def _phase_arrays(section: Mapping[str, Any]) -> tuple[Any, Any]:
    temperatures = _first(section, "temperature_K", "temperatures_K", "temperature", "equilibrium.temperature_K")
    fractions = _first(section, "phase_fractions", "fractions", "equilibrium.phase_fractions")
    if isinstance(fractions, list) and fractions and isinstance(fractions[0], Mapping):
        frame = pd.DataFrame(fractions)
        phase_column = next((column for column in ("phase", "Phase") if column in frame), None)
        fraction_column = next(
            (column for column in ("molar_phase_fraction", "phase_fraction", "fraction", "NP") if column in frame),
            None,
        )
        temp_column = next((column for column in ("temperature_K", "T_K", "temperature") if column in frame), None)
        if temp_column and phase_column and fraction_column:
            pivot = frame.pivot_table(
                index=temp_column,
                columns=phase_column,
                values=fraction_column,
                aggfunc="sum",
                fill_value=0.0,
            ).sort_index()
            temperatures = pivot.index.to_list()
            fractions = {str(column): pivot[column].to_list() for column in pivot}
        elif temp_column:
            temperatures = frame.pop(temp_column).tolist()
            numeric_columns = frame.select_dtypes(include="number").columns
            fractions = {column: frame[column].tolist() for column in numeric_columns}
    return temperatures, fractions


def _phases_tab(analysis: Mapping[str, Any], composition_input: Mapping[str, float]) -> tuple[Any, Any, Any]:
    thermo = _mapping(analysis.get("thermodynamics"))
    phases = _mapping(analysis.get("phases"))
    _section_intro(thermo)
    temperatures, fractions = _phase_arrays(thermo)
    phase_figure = phase_fraction_temperature(temperatures, fractions)
    ternary = ternary_composition_plot(composition_input, basis="wt%", title="Current composition in Cu-Al-Ni space")
    map_payload = _mapping(thermo.get("composition_phase_map"))
    map_points = _first(map_payload, "points", default=None)
    if map_points:
        map_temperature = map_payload.get("temperature_K")
        suffix = f" at {float(map_temperature):.1f} K" if map_temperature is not None else ""
        stability = ternary_phase_assemblage_map(
            map_points,
            title=f"Calculated isothermal phase-assemblage map{suffix}",
        )
    else:
        stability = phase_stability_map(
            _first(thermo, "stability_map.x", "phase_stability.x"),
            _first(thermo, "stability_map.y", "phase_stability.y"),
            _first(thermo, "stability_map.z", "phase_stability.z"),
            value_label=str(_first(thermo, "stability_map.label", default="Stability metric")),
        )
    st.plotly_chart(phase_figure, width="stretch", config={"displaylogo": False})
    c1, c2 = st.columns(2)
    c1.plotly_chart(stability, width="stretch", config={"displaylogo": False})
    c2.plotly_chart(ternary, width="stretch", config={"displaylogo": False})
    st.markdown("#### Phase and processing risks")
    risks = _mapping(_first(phases, "risks", "risk_indicators", default={}))
    risk_cols = st.columns(3)
    for column, (key, label) in zip(
        risk_cols,
        (("gamma2", "γ₂ formation"), ("brittleness", "Brittleness"), ("segregation", "Segregation")),
        strict=True,
    ):
        value = risks.get(key, risks.get(f"{key}_risk", "Unavailable"))
        column.markdown(
            result_card_html(
                label,
                value,
                classification=_classification(phases) if value != "Unavailable" else "unavailable",
                confidence=phases.get("confidence") if value != "Unavailable" else None,
                provenance=phases.get("provenance"),
                limitations=[] if value != "Unavailable" else ["The active phase engine did not return this risk indicator."],
            ),
            unsafe_allow_html=True,
        )
    catalog = _mapping(phases.get("phase_reference_catalog"))
    if catalog:
        with st.expander("Candidate Cu-Al-Ni phase reference catalog", expanded=False):
            st.caption(
                "Reference descriptions document candidate phases only. They are not phase assignments, "
                "stability fields, or calculated fractions."
            )
            st.json(catalog.get("phases", catalog))
    scheil = _mapping(thermo.get("scheil"))
    if _status(scheil):
        st.markdown("#### Scheil-Gulliver solidification result")
        st.dataframe(
            _safe_dataframe(scheil.get("phase_fractions")),
            hide_index=True,
            width="stretch",
        )
        _limitations(scheil, title="Scheil assumptions and limitations")
    _limitations(thermo)
    return phase_figure, stability, ternary


def _structure_payload(record: Mapping[str, Any]) -> tuple[Any, Any, str | None]:
    structure = _mapping(record.get("structure")) or dict(record)
    lattice = _first(structure, "lattice.matrix", "lattice_matrix", "matrix")
    sites = structure.get("sites") or record.get("sites")
    normalized_sites = []
    for site in sites or []:
        item = _mapping(site)
        species = item.get("species")
        element = item.get("element", item.get("label"))
        occupancy = item.get("occupancy", 1.0)
        if isinstance(species, list) and species:
            species_item = _mapping(species[0])
            element = species_item.get("element", species_item.get("name", element))
            occupancy = species_item.get("occu", occupancy)
        normalized_sites.append(
            {
                "element": element,
                "occupancy": occupancy,
                "fractional_coordinates": item.get("fractional_coordinates", item.get("abc", item.get("frac_coords"))),
            }
        )
    space_group = _first(record, "space_group", "symmetry.space_group", "spacegroup")
    return lattice, normalized_sites, str(space_group) if space_group else None


def _pattern_payload(pattern: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    two_theta = _first(pattern, "two_theta", "two_theta_deg", "x", "pattern.two_theta_deg")
    intensity = _first(pattern, "intensity", "relative_intensity", "y", "pattern.intensity")
    peaks = _first(pattern, "peaks", "peak_table", default=[])
    return two_theta, intensity, peaks


def _crystal_xrd_tab(analysis: Mapping[str, Any]) -> tuple[Any, Any]:
    crystal = _mapping(analysis.get("crystallography"))
    xrd = _mapping(analysis.get("xrd"))
    _section_intro(crystal)
    structures = _mapping(crystal.get("structures"))
    patterns = _mapping(xrd.get("patterns"))
    phase_names = sorted(set(structures) | set(patterns))
    selected = st.selectbox("Crystal / phase model", phase_names or ["No validated structure available"])
    record = _mapping(structures.get(selected))
    lattice, sites, space_group = _structure_payload(record)
    crystal_figure = crystal_structure_figure(lattice, sites, phase_name=selected, space_group=space_group)
    pattern = _mapping(patterns.get(selected))
    two_theta, intensity, peaks = _pattern_payload(pattern)
    xrd_figure = xrd_pattern_plot(
        two_theta,
        intensity,
        peaks=peaks,
        wavelength_angstrom=_first(xrd, "wavelength_angstrom"),
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(crystal_figure, width="stretch", config={"displaylogo": False})
    c2.plotly_chart(xrd_figure, width="stretch", config={"displaylogo": False})
    if record:
        metadata = {key: value for key, value in record.items() if key not in {"structure", "sites", "lattice", "lattice_matrix"}}
        if metadata:
            with st.expander("Structure metadata and citation", expanded=True):
                st.json(metadata)
    literature = _mapping(crystal.get("literature_references"))
    if literature:
        with st.expander("Curated phase references (metadata only)", expanded=not bool(record)):
            st.caption("These records do not contain enough atomic data for XRD and are never promoted to CIFs.")
            st.json(literature)
    if crystal.get("load_errors"):
        st.warning("Some CIFs could not be loaded.")
        st.json(crystal["load_errors"])
    if peaks:
        peak_frame = _safe_dataframe(peaks)
        preferred = [column for column in ("two_theta", "two_theta_deg", "hkl", "d_spacing", "d_spacing_angstrom", "phase", "intensity") if column in peak_frame]
        st.markdown("#### Indexed peak table")
        st.dataframe(peak_frame[preferred] if preferred else peak_frame, hide_index=True, width="stretch")
    _limitations(crystal, title="Crystallography limitations")
    _limitations(xrd, title="Diffraction limitations")
    return crystal_figure, xrd_figure


def _ml_values(section: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    predictions: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    display_names = {
        "Ms_C": "Ms",
        "Mf_C": "Mf",
        "As_C": "As",
        "Af_C": "Af",
        "hardness_HV": "Hardness",
        "yield_strength_MPa": "Yield strength",
        "uts_MPa": "UTS",
    }
    for target, result in _mapping(section.get("targets")).items():
        mapping = _mapping(result)
        label = display_names.get(target, target)
        for output, candidates in (
            (predictions, ("prediction", "value", "values")),
            (lower, ("lower", "lower_bound")),
            (upper, ("upper", "upper_bound")),
        ):
            value = next((mapping.get(key) for key in candidates if mapping.get(key) is not None), None)
            if isinstance(value, list) and value:
                value = value[0]
            try:
                output[label] = float(value)
            except (TypeError, ValueError):
                pass
    return predictions, lower, upper


def _ml_tab(analysis: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    section = _mapping(analysis.get("ml"))
    _section_intro(section)
    predictions, lower, upper = _ml_values(section)
    transformation_section = {
        "targets": _mapping(section.get("transformation_targets"))
        or {
            target: value
            for target, value in _mapping(section.get("targets")).items()
            if target in {"Ms_C", "Mf_C", "As_C", "Af_C"}
        }
    }
    transformation_predictions, transformation_lower, transformation_upper = _ml_values(
        transformation_section
    )
    uncertainties = {
        key: max(0.0, transformation_upper[key] - value)
        for key, value in transformation_predictions.items()
        if key in transformation_upper and key in transformation_lower
    }
    transform = transformation_temperature_chart(transformation_predictions, uncertainties)
    uncertainty = uncertainty_plot(predictions, lower, upper)
    target_rows: list[dict[str, Any]] = []
    for target, result in _mapping(section.get("targets")).items():
        item = _mapping(result)
        prediction = item.get("prediction")
        low = item.get("lower")
        high = item.get("upper")
        if isinstance(prediction, list):
            prediction = prediction[0] if prediction else None
        if isinstance(low, list):
            low = low[0] if low else None
        if isinstance(high, list):
            high = high[0] if high else None
        applicability = item.get("applicability") or []
        first_applicability = _mapping(applicability[0]) if applicability else {}
        confidence = _mapping(item.get("confidence"))
        target_rows.append(
            {
                "Target": target,
                "Prediction": prediction,
                "Lower": low,
                "Upper": high,
                "Unit": item.get("unit"),
                "Nominal coverage": confidence.get("nominal_coverage"),
                "Within training ranges": first_applicability.get("within_training_ranges"),
            }
        )
    if target_rows:
        st.markdown("#### Current-composition predictions")
        st.dataframe(pd.DataFrame(target_rows), hide_index=True, width="stretch")
        if not bool(_first(section, "applicability_summary.all_predictions_within_univariate_training_ranges", default=True)):
            st.warning(
                "At least one prediction is outside a one-dimensional training range. It is retained for audit, "
                "but should be treated as extrapolation."
            )
        consistency = _mapping(section.get("transformation_consistency"))
        if consistency.get("all_available_checks_pass") is False:
            st.error(
                "Independent target models violate Mf ≤ Ms and/or As ≤ Af. Values were not silently reordered; "
                "review the dataset and model validation."
            )

    property_maps = _mapping(section.get("property_maps"))
    if property_maps:
        selected_property = st.selectbox(
            "Composition-property map target",
            list(property_maps),
            key="property_map_target",
        )
        property_map_payload = _mapping(property_maps[selected_property])
    else:
        property_map_payload = _mapping(
            _first(section, "property_map", "composition_map", default={})
        )
    property_map = composition_property_map(
        property_map_payload.get("compositions"),
        property_map_payload.get("values"),
        property_name=str(property_map_payload.get("property_name", "Predicted property")),
        property_unit=str(property_map_payload.get("unit", "")),
    )
    c1, c2 = st.columns(2)
    c1.plotly_chart(transform, width="stretch", config={"displaylogo": False})
    c2.plotly_chart(uncertainty, width="stretch", config={"displaylogo": False})
    st.plotly_chart(property_map, width="stretch", config={"displaylogo": False})
    validation = _mapping(section.get("validation"))
    if validation:
        st.markdown("#### Cross-validation")
        selected = st.selectbox("Validation target", list(validation), key="validation_target")
        st.dataframe(_safe_dataframe(validation[selected]), hide_index=True, width="stretch")
    explanations = _mapping(section.get("explainability"))
    if explanations:
        st.markdown("#### Explainability")
        selected_explanation = st.selectbox(
            "Explanation target",
            list(explanations),
            key="explanation_target",
        )
        explanation = _mapping(explanations[selected_explanation])
        st.caption(
            f"Method: {explanation.get('method', 'unavailable')} · representative ensemble member: "
            f"{explanation.get('representative_model', 'unavailable')}"
        )
        st.dataframe(
            _safe_dataframe(explanation.get("importance", [])),
            hide_index=True,
            width="stretch",
        )
        if explanation.get("shap_unavailable_reason"):
            st.info(
                "SHAP was unavailable or incompatible for this estimator; the displayed result is explicitly "
                "model-agnostic permutation importance, not SHAP."
            )
        st.caption(
            explanation.get("limitation")
            or "Feature importance explains this fitted model only; it does not establish causality."
        )
    else:
        importance = _first(section, "feature_importance", "shap.feature_importance")
        if importance:
            st.markdown("#### Explainability")
            st.dataframe(_safe_dataframe(importance), hide_index=True, width="stretch")
            st.caption("Feature importance explains this fitted model only; it does not establish causality.")
        else:
            st.info("Feature-importance output is unavailable for the current fitted model.")
    if property_map_payload.get("limitations"):
        with st.expander("Composition-property map limitations", expanded=False):
            for limitation in property_map_payload["limitations"]:
                st.markdown(f"- {limitation}")
    _limitations(section)
    return transform, uncertainty, property_map


def _characterization_tab(analysis: Mapping[str, Any]) -> None:
    section = _mapping(analysis.get("characterization"))
    _section_intro(section)
    analyses = _mapping(section.get("analyses"))
    if not analyses:
        st.markdown("Upload instrument exports in the sidebar. Accepted data are parsed and validated before analysis.")
        _limitations(section)
        return
    tabs = st.tabs([key.upper() for key in analyses])
    for tab, (kind, result) in zip(tabs, analyses.items(), strict=True):
        with tab:
            result_mapping = _mapping(result)
            _section_intro(result_mapping)
            if kind == "eds":
                statistics = _first(result_mapping, "statistics", default=[])
                st.dataframe(_safe_dataframe(statistics), hide_index=True, width="stretch")
                segregation = _mapping(result_mapping.get("segregation_metrics"))
                if segregation:
                    st.markdown("#### Segregation diagnostics")
                    st.json(segregation)
            elif kind == "xrd":
                peaks = _first(result_mapping, "peaks", default=[])
                matches = _first(result_mapping, "matches", default=[])
                c1, c2 = st.columns(2)
                c1.markdown("#### Detected peaks")
                c1.dataframe(_safe_dataframe(peaks), hide_index=True, width="stretch")
                c2.markdown("#### Reference matches")
                c2.dataframe(_safe_dataframe(matches), hide_index=True, width="stretch")
                scores = _mapping(result_mapping.get("phase_matching_scores"))
                if scores:
                    st.bar_chart(pd.Series(scores, name="Uncalibrated score"))
                    st.caption("Scores are uncalibrated reference-match scores, not probabilities or phase fractions.")
            elif kind == "dsc":
                _display_mapping_table(result_mapping)
                transitions = _first(result_mapping, "temperatures_C", "transformations", "temperatures", default={})
                if isinstance(transitions, Mapping):
                    st.plotly_chart(transformation_temperature_chart(transitions), width="stretch")
            elif kind == "sem":
                preview = _mapping(result_mapping.get("preview"))
                processed = np.asarray(preview.get("processed_image", []), dtype=float)
                label_map = np.asarray(preview.get("label_map", []))
                if processed.ndim == 2 and processed.size:
                    c1, c2 = st.columns(2)
                    c1.markdown("#### Preprocessed SEM preview")
                    c1.image(
                        processed,
                        clamp=True,
                        width="stretch",
                        caption=(
                            f"Normalized, denoised image · preview stride "
                            f"{preview.get('downsample_stride', 1)}"
                        ),
                    )
                    c2.markdown("#### Segmentation preview")
                    if label_map.shape == processed.shape and label_map.size:
                        palette = np.asarray(
                            [
                                [55, 182, 255],
                                [84, 214, 200],
                                [245, 158, 11],
                                [169, 159, 255],
                                [239, 125, 50],
                                [148, 163, 184],
                                [244, 114, 182],
                                [74, 222, 128],
                            ],
                            dtype=np.uint8,
                        )
                        safe_labels = np.mod(label_map.astype(int), len(palette))
                        c2.image(
                            palette[safe_labels],
                            width="stretch",
                            caption="Colours denote segmentation labels; unsupervised regions are not phases.",
                        )
                else:
                    st.info("A bounded SEM preview was not included in this analysis result.")
                fractions = _mapping(result_mapping.get("area_fractions_2d"))
                if fractions:
                    st.markdown("#### Two-dimensional area fractions")
                    st.bar_chart(pd.Series(fractions, name="Area fraction"))
                    label_names = preview.get("label_names", list(fractions))
                    st.caption("Labels: " + ", ".join(str(item) for item in label_names))
                _display_mapping_table(result_mapping)
            _limitations(result_mapping)


def _am_tab(analysis: Mapping[str, Any]) -> Any:
    section = _mapping(analysis.get("am"))
    _section_intro(section)
    components = _mapping(section.get("components"))
    criteria = {
        name: float(item["normalized_score_0_to_10"])
        for name, raw in components.items()
        if isinstance((item := _mapping(raw)).get("normalized_score_0_to_10"), (int, float))
    }
    if not criteria:
        criteria = _mapping(_first(section, "criteria", "criterion_scores", "scores", default={}))
    radar = am_suitability_radar(criteria)
    left, right = st.columns([1.2, 1])
    left.plotly_chart(radar, width="stretch", config={"displaylogo": False})
    with right:
        score = _first(section, "score_0_to_10", "score", "overall_score")
        unit = "/10" if score is not None else ""
        st.markdown(
            result_card_html(
                "LPBF suitability",
                score if score is not None else "Unavailable",
                unit=unit,
                classification=_classification(section),
                confidence=section.get("confidence"),
                provenance=section.get("provenance"),
                limitations=section.get("limitations"),
            ),
            unsafe_allow_html=True,
        )
        risks = section.get("risks") or section.get("risk_flags") or []
        recommendations = section.get("recommendations") or []
        st.markdown("#### Risks")
        if isinstance(risks, Mapping):
            st.json(risks)
        elif risks:
            for risk in risks:
                st.markdown(f"- {risk}")
        else:
            st.caption("No validated risk list is available.")
        st.markdown("#### Recommendations")
        if recommendations:
            for recommendation in recommendations:
                st.markdown(f"- {recommendation}")
        else:
            st.caption("No process recommendation is emitted without a completed screen.")
        if section.get("grade"):
            st.caption(f"Profile grade: {section['grade']} (defined by the uploaded profile)")
        if section.get("missing_metrics"):
            st.caption("Missing profile metrics: " + ", ".join(section["missing_metrics"]))
    _limitations(section)
    return radar


def _report_tab(analysis: Mapping[str, Any], figures: Mapping[str, list[Any]]) -> None:
    st.markdown("### Export a traceable result bundle")
    st.write(
        "The HTML report retains interactive Plotly figures. The PDF is a stable paginated record. "
        "The JSON export preserves machine-readable values and audit metadata."
    )
    # Keep one export bundle per analysis, private to this user's session. A
    # rerun or download must not create a different report ID or regenerate MBs
    # of embedded Plotly content. A new analysis invalidates this bundle below.
    if "report_bundle" not in st.session_state:
        enriched = deepcopy(dict(analysis))
        for section, section_figures in figures.items():
            if section in enriched and isinstance(enriched[section], dict):
                enriched[section]["figures"] = [figure for figure in section_figures if figure is not None]
        generator = AnalysisReportGenerator(
            enriched,
            generated_at=datetime.now(timezone.utc),
            title="Cu-Al-Ni Digital Twin Analysis",
        )
        try:
            st.session_state["report_bundle"] = {
                "html": generator.to_html_bytes(),
                "pdf": generator.to_pdf_bytes(),
                "json": generator.to_json_bytes(),
                "id": generator.report_id,
                "generated_at": generator.generated_at,
            }
        except ReportGenerationError as exc:
            st.error(str(exc))
            return
    bundle = st.session_state["report_bundle"]
    stem = "CuAlNi_DigitalTwin_analysis"
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download scientific report (PDF)", bundle["pdf"], f"{stem}.pdf", "application/pdf", width="stretch", on_click="ignore")
    c2.download_button("Download interactive report (HTML)", bundle["html"], f"{stem}.html", "text/html", width="stretch", on_click="ignore")
    c3.download_button("Download result bundle (JSON)", bundle["json"], f"{stem}.json", "application/json", width="stretch", on_click="ignore")
    st.caption(f"Report ID: {bundle['id']} · generated {bundle['generated_at']:%Y-%m-%d %H:%M UTC}")
    with st.expander("Audit preview"):
        st.json({key: {"status": _mapping(value).get("status"), "classification": _classification(_mapping(value)), "provenance": _mapping(value).get("provenance")} for key, value in analysis.items() if isinstance(value, Mapping)})


def main() -> None:
    _hero()
    submitted, inputs = _build_sidebar()
    if submitted and inputs is not None:
        progress_bar = st.progress(0.0, text="Preparing analysis")

        def update_progress(message: str, value: float) -> None:
            progress_bar.progress(value, text=message)

        with st.spinner("Running independent scientific engines…"):
            st.session_state["analysis"] = run_complete_analysis(inputs, progress=update_progress)
            st.session_state["analysis_inputs"] = inputs
            st.session_state.pop("report_bundle", None)
        progress_bar.empty()
        st.toast("Analysis completed. Review provenance and limitations in each tab.", icon="✅")

    analysis = st.session_state.get("analysis")
    active_inputs = st.session_state.get("analysis_inputs")
    if submitted and inputs is None and analysis:
        st.warning("The new inputs were rejected. The results below are from your last completed analysis.")
    if not analysis or active_inputs is None:
        _empty_landing()
        return

    available_count = sum(
        _status(_mapping(value))
        for key, value in analysis.items()
        if key != "schema_version" and isinstance(value, Mapping)
    )
    st.caption(f"Analysis modules available: {available_count}/8 · unavailable modules remain explicit in screen and exports")

    tab_names = ["Overview", "Phases", "Crystal & XRD", "Properties & ML", "Characterization", "AM suitability", "Report"]
    overview, phases_tab, crystal_tab, ml_tab, characterization_tab, am_tab, report_tab = st.tabs(tab_names)
    figures: dict[str, list[Any]] = {}
    with overview:
        ternary, _ = _composition_tab(analysis, active_inputs.composition_wt_percent)
        figures["composition"] = [ternary]
    with phases_tab:
        phase_figure, stability, phase_ternary = _phases_tab(analysis, active_inputs.composition_wt_percent)
        figures["thermodynamics"] = [phase_figure, stability]
        figures["phases"] = [phase_ternary]
    with crystal_tab:
        crystal_figure, xrd_figure = _crystal_xrd_tab(analysis)
        figures["crystallography"] = [crystal_figure]
        figures["xrd"] = [xrd_figure]
    with ml_tab:
        transform, uncertainty, property_map = _ml_tab(analysis)
        figures["ml"] = [transform, uncertainty, property_map]
    with characterization_tab:
        _characterization_tab(analysis)
    with am_tab:
        radar = _am_tab(analysis)
        figures["am"] = [radar]
    with report_tab:
        _report_tab(analysis, figures)


if __name__ == "__main__":
    main()
