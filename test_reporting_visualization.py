import json
from datetime import datetime, timezone

import plotly.graph_objects as go

from reporting.report_generator import AnalysisReportGenerator
from visualization.phase_plots import ternary_composition_plot
from visualization.xrd_plots import xrd_pattern_plot


def _minimal_analysis():
    unavailable = {
        "status": "unavailable",
        "classification": "unavailable",
        "confidence": "Not applicable",
        "provenance": "Not executed",
        "limitations": ["Required scientific input was not supplied."],
    }
    composition = {
        "status": "available",
        "classification": "physics_calculated",
        "confidence": "deterministic",
        "provenance": "software test",
        "limitations": ["test payload"],
        "weight_percent": {"Cu": 81.4, "Al": 14.3, "Ni": 4.3},
    }
    return {
        "composition": composition,
        **{key: dict(unavailable) for key in ("thermodynamics", "phases", "crystallography", "xrd", "ml", "characterization", "am")},
    }


def test_report_exports_html_pdf_json():
    generator = AnalysisReportGenerator(
        _minimal_analysis(),
        report_id="TEST-001",
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    html = generator.to_html_bytes()
    pdf = generator.to_pdf_bytes()
    payload = json.loads(generator.to_json_bytes())
    assert b"TEST-001" in html
    assert pdf.startswith(b"%PDF")
    assert payload["report_id"] == "TEST-001"


def test_ternary_figure_contains_current_point():
    figure = ternary_composition_plot({"Cu": 81.4, "Al": 14.3, "Ni": 4.3})
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1


def test_xrd_stick_pattern_accepts_peak_table_without_profile_arrays():
    figure = xrd_pattern_plot(
        None,
        None,
        peaks=[{"two_theta_deg": 42.0, "scaled_intensity": 100.0, "hkl": "(1, 1, 0)", "phase": "test"}],
    )
    assert isinstance(figure, go.Figure)
    assert len(figure.data) == 1

