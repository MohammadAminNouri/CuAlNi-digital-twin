"""HTML and PDF exports for CuAlNi-DigitalTwin Pro.

The exporter reports the values it receives; it does not calculate, infer, or fill
scientific results.  Every section carries a mandatory classification, confidence,
provenance, and limitations block.  If any of those are absent, the section is
marked unavailable instead of silently implying scientific validity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any
from uuid import uuid4

CLASSIFICATION_LABELS = {
    "physics_calculated": "Physics calculated",
    "machine_learning_prediction": "Machine learning prediction",
    "engineering_screening": "Engineering screening",
    "experimental_measurement": "Experimental measurement",
    "unavailable": "Unavailable",
}

SECTION_SPECS = (
    ("composition", "Composition and descriptors"),
    ("thermodynamics", "Thermodynamics and phase stability"),
    ("phases", "Phase assessment"),
    ("crystallography", "Crystallography"),
    ("xrd", "Diffraction / XRD"),
    ("ml", "Machine-learning property predictions"),
    ("characterization", "Experimental characterization"),
    ("am", "Additive-manufacturing screening"),
)

REPORT_CSS = """
@page { size: A4; margin: 18mm 16mm 18mm; }
:root { --ink:#172235; --muted:#65758b; --line:#d7e0eb; --teal:#138f84; --violet:#6253cf; --amber:#bb7100; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); font-family:Inter,Segoe UI,Arial,sans-serif; font-size:14px; line-height:1.48; }
.cover { padding:34px; color:white; min-height:250px; background:linear-gradient(125deg,#0c273d,#26356f); border-radius:18px; }
.kicker { font-size:11px; letter-spacing:.15em; color:#83e8db; font-weight:750; }
h1 { margin:8px 0 3px; font-size:34px; letter-spacing:-.035em; }
h2 { font-size:20px; margin:32px 0 12px; border-bottom:2px solid #dfe7f0; padding-bottom:7px; }
.subtitle { color:#c6d2e2; max-width:720px; }
.meta { display:flex; flex-wrap:wrap; gap:10px; margin-top:26px; }
.meta span { border:1px solid rgba(255,255,255,.24); border-radius:999px; padding:5px 10px; font-size:11px; }
.provenance-banner { margin:22px 0 5px; padding:14px 16px; background:#f2f6fb; border-left:4px solid #66758a; border-radius:5px; }
.section { break-inside:avoid-page; }
.badges { margin:3px 0 10px; }
.badge { display:inline-block; border-radius:999px; padding:3px 8px; margin-right:5px; font-size:11px; font-weight:700; border:1px solid currentColor; }
.physics_calculated { color:var(--teal); background:#edf9f7; }
.machine_learning_prediction { color:var(--violet); background:#f2f0ff; }
.engineering_screening { color:var(--amber); background:#fff7e6; }
.experimental_measurement { color:#0879ac; background:#edf8fc; }
.unavailable { color:#687488; background:#f1f3f6; }
table { width:100%; border-collapse:collapse; margin:8px 0 13px; font-size:12px; }
th { text-align:left; color:#536176; background:#f1f5fa; font-size:10px; text-transform:uppercase; letter-spacing:.06em; }
th,td { padding:7px 8px; border:1px solid var(--line); vertical-align:top; }
.limitations { color:#5e6878; background:#f8fafc; border-left:3px solid #8794a6; padding:9px 11px; font-size:12px; }
.figure { margin:14px 0; border:1px solid var(--line); border-radius:9px; padding:7px; break-inside:avoid-page; }
.empty { color:#738095; font-style:italic; padding:8px 0; }
.footer-note { margin-top:36px; padding-top:10px; border-top:1px solid var(--line); color:#6b788c; font-size:10px; }
code { background:#eff3f8; padding:1px 4px; border-radius:3px; }
"""


class ReportGenerationError(RuntimeError):
    """Raised when an export dependency or payload is invalid."""


@dataclass(frozen=True)
class ReportSection:
    key: str
    title: str
    classification: str
    confidence: str
    provenance: str
    limitations: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]
    figure_html: tuple[str, ...] = ()


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime,)):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return vars(value)
    return str(value)


def _as_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    elif hasattr(value, "__dict__") and not isinstance(value, dict):
        value = vars(value)
    return dict(value) if isinstance(value, dict) else {"value": value}


def _classification(mapping: dict[str, Any]) -> str:
    candidate = mapping.get("result_classification", mapping.get("classification", "unavailable"))
    return str(candidate) if str(candidate) in CLASSIFICATION_LABELS else "unavailable"


def _confidence(value: Any) -> str:
    if value is None:
        return "Not reported"
    if isinstance(value, (int, float)):
        return f"{float(value):.0%}" if 0 <= float(value) <= 1 else f"{float(value):.3g}"
    return str(value)


def _limitations(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("No limitations metadata was provided; treat this result as unavailable.",)
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value) or ("No limitations declared.",)
    except TypeError:
        return (str(value),)


def _display(value: Any, depth: int = 0) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (int, str)):
        return str(value)
    if depth > 2:
        return json.dumps(value, default=_json_default, ensure_ascii=False)
    if isinstance(value, dict):
        return "; ".join(f"{key}: {_display(item, depth + 1)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        preview = list(value)
        if len(preview) > 12:
            return ", ".join(_display(item, depth + 1) for item in preview[:12]) + f" … ({len(preview)} values)"
        return ", ".join(_display(item, depth + 1) for item in preview)
    return str(value)


def _pdf_safe_text(value: Any) -> str:
    """Transliterate common scientific glyphs for ReportLab's core fonts."""

    replacements = str.maketrans(
        {
            "α": "alpha",
            "β": "beta",
            "γ": "gamma",
            "δ": "delta",
            "θ": "theta",
            "λ": "lambda",
            "μ": "micro",
            "µ": "micro",
            "Å": "angstrom",
            "°": " deg",
            "₀": "0",
            "₁": "1",
            "₂": "2",
            "₃": "3",
            "₄": "4",
            "₅": "5",
            "₆": "6",
            "₇": "7",
            "₈": "8",
            "₉": "9",
            "⁰": "0",
            "¹": "1",
            "²": "2",
            "³": "3",
            "⁴": "4",
            "⁵": "5",
            "⁶": "6",
            "⁷": "7",
            "⁸": "8",
            "⁹": "9",
            "‐": "-",
            "‑": "-",
            "‒": "-",
            "–": "-",
            "—": "-",
            "−": "-",
            "·": " | ",
            "…": "...",
            "×": "x",
            "≤": "<=",
            "≥": ">=",
            "′": "'",
            "″": '"',
            " ": " ",
        }
    )
    return str(value).translate(replacements)


def _rows(mapping: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    metadata = {
        "result_classification",
        "classification",
        "confidence",
        "limitations",
        "provenance",
        "status",
        "figures",
        "raw",
    }
    output: list[tuple[str, str]] = []
    for key, value in mapping.items():
        if key in metadata or value is None:
            continue
        if key == "descriptors" and isinstance(value, dict):
            for descriptor_key, descriptor_value in value.items():
                descriptor = _as_mapping(descriptor_value)
                measured_value = descriptor.get("value")
                unit = str(descriptor.get("unit") or "").strip()
                summary = _display(measured_value)
                if measured_value is not None and unit:
                    summary = f"{summary} {unit}"
                descriptor_metadata = _as_mapping(descriptor.get("metadata"))
                descriptor_class = descriptor_metadata.get("classification")
                if descriptor_class:
                    summary += f" [{str(descriptor_class).replace('_', ' ')}]"
                output.append((str(descriptor_key).replace("_", " ").title(), summary))
            continue
        # Dense solver/model arrays belong in figures or downloadable data, not report tables.
        if isinstance(value, (list, tuple)) and len(value) > 60:
            output.append((str(key).replace("_", " ").title(), f"{len(value)} values (see digital export)"))
        else:
            rendered = _display(value)
            if isinstance(value, dict) and len(rendered) > 900:
                records = value.get("phases") if isinstance(value.get("phases"), dict) else value
                record_names = list(records) if isinstance(records, dict) else []
                preview = ", ".join(str(item) for item in record_names[:12])
                suffix = f": {preview}" if preview else ""
                rendered = f"{len(record_names)} structured records{suffix} (see JSON export for full metadata)"
            output.append((str(key).replace("_", " ").title(), rendered))
    return tuple(output)


def _figure_fragments(figures: Any) -> tuple[str, ...]:
    fragments: list[str] = []
    if figures is None:
        return ()
    if isinstance(figures, dict):
        iterable = figures.values()
    elif isinstance(figures, (list, tuple)):
        iterable = figures
    else:
        iterable = (figures,)
    for figure in iterable:
        if figure is None or not hasattr(figure, "to_html"):
            continue
        fragments.append(
            figure.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={"displaylogo": False, "responsive": True},
            )
        )
    return tuple(fragments)


def _embedded_plotly_javascript(sections: tuple[ReportSection, ...]) -> str:
    """Return one embedded Plotly runtime when the report contains figures.

    A report is an archival output, so it must not depend on a CDN that may be
    unavailable in a laboratory network or after the URL changes.  Figure
    fragments deliberately omit their own runtime and share this single copy.
    """

    if not any(section.figure_html for section in sections):
        return ""
    try:
        from plotly.offline.offline import get_plotlyjs
    except ImportError as exc:
        raise ReportGenerationError(
            "Offline interactive figures require Plotly. Install `plotly`."
        ) from exc
    return get_plotlyjs()


class AnalysisReportGenerator:
    """Create self-contained HTML and paginated PDF reports from one analysis payload."""

    def __init__(
        self,
        analysis: Any,
        *,
        title: str = "Cu-Al-Ni Digital Twin Analysis",
        report_id: str | None = None,
        generated_at: datetime | None = None,
    ) -> None:
        self.analysis = _as_mapping(analysis)
        self.title = title
        self.report_id = report_id or f"CAN-{uuid4().hex[:10].upper()}"
        self.generated_at = generated_at or datetime.now(timezone.utc)

    def _sections(self, *, include_figures: bool) -> tuple[ReportSection, ...]:
        sections: list[ReportSection] = []
        for key, title in SECTION_SPECS:
            mapping = _as_mapping(self.analysis.get(key))
            classification = _classification(mapping)
            provenance = str(mapping.get("provenance") or "Not reported")
            limitations = _limitations(mapping.get("limitations"))
            # Missing required scientific metadata cannot inherit a positive classification.
            if not mapping or provenance == "Not reported" or "limitations" not in mapping:
                classification = "unavailable"
            sections.append(
                ReportSection(
                    key=key,
                    title=title,
                    classification=classification,
                    confidence=_confidence(mapping.get("confidence")),
                    provenance=provenance,
                    limitations=limitations,
                    rows=_rows(mapping),
                    figure_html=_figure_fragments(mapping.get("figures")) if include_figures else (),
                )
            )
        return tuple(sections)

    def to_json_bytes(self) -> bytes:
        """Return the complete machine-readable result bundle."""

        envelope = {
            "schema_version": "1.0",
            "report_id": self.report_id,
            "generated_at_utc": self.generated_at.astimezone(timezone.utc).isoformat(),
            "analysis": self.analysis,
        }
        return json.dumps(envelope, indent=2, ensure_ascii=False, default=_json_default).encode("utf-8")

    def to_html(self) -> str:
        """Return a self-contained scientific HTML report with interactive figures."""

        sections = self._sections(include_figures=True)
        plotly_javascript = _embedded_plotly_javascript(sections)
        try:
            from jinja2 import BaseLoader, Environment, select_autoescape
        except ImportError as exc:
            raise ReportGenerationError("HTML export requires Jinja2. Install `jinja2`.") from exc
        template = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(default=True),
        ).from_string(_HTML_TEMPLATE)
        return template.render(
            css=REPORT_CSS,
            title=self.title,
            report_id=self.report_id,
            generated_at=self.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            sections=sections,
            labels=CLASSIFICATION_LABELS,
            plotly_javascript=plotly_javascript,
        )

    def to_html_bytes(self) -> bytes:
        return self.to_html().encode("utf-8")

    def to_pdf_bytes(self) -> bytes:
        """Return a paginated PDF. Plotly figures remain available in the HTML export."""

        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_LEFT
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import mm
            from reportlab.platypus import (
                KeepTogether,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise ReportGenerationError("PDF export requires ReportLab. Install `reportlab`.") from exc

        buffer = BytesIO()
        document = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=16 * mm,
            leftMargin=16 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title=self.title,
            author="CuAlNi-DigitalTwin Pro",
            subject="Provenance-aware Cu-Al-Ni analysis",
        )
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=23,
                leading=27,
                textColor=colors.HexColor("#FFFFFF"),
                alignment=TA_LEFT,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="CoverKicker",
                parent=styles["BodyText"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#83E8DB"),
                spaceAfter=5,
            )
        )
        styles.add(
            ParagraphStyle(
                name="CoverSubtitle",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=9.5,
                leading=12,
                textColor=colors.HexColor("#C6D2E2"),
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="CoverMeta",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#C6D2E2"),
            )
        )
        styles.add(
            ParagraphStyle(
                name="SectionTitle",
                parent=styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=14,
                leading=17,
                textColor=colors.HexColor("#14233A"),
                spaceBefore=14,
                spaceAfter=8,
            )
        )
        styles.add(
            ParagraphStyle(
                name="Small",
                parent=styles["BodyText"],
                fontSize=8.5,
                leading=11,
                textColor=colors.HexColor("#526178"),
            )
        )
        styles.add(
            ParagraphStyle(
                name="Cell",
                parent=styles["BodyText"],
                fontSize=8.5,
                leading=10.5,
                textColor=colors.HexColor("#1D2A3D"),
                wordWrap="CJK",
            )
        )
        styles.add(
            ParagraphStyle(
                name="Badge",
                parent=styles["Small"],
                fontName="Helvetica-Bold",
                fontSize=8,
                textColor=colors.HexColor("#147D75"),
            )
        )
        story: list[Any] = []
        cover_content = [
            Paragraph("CUALNI-DIGITALTWIN PRO", styles["CoverKicker"]),
            Paragraph(escape(_pdf_safe_text(self.title)), styles["ReportTitle"]),
            Paragraph(
                "Composition to crystal structure, phase stability, properties, and "
                "additive-manufacturing screening",
                styles["CoverSubtitle"],
            ),
            Paragraph(
                f"Report {escape(_pdf_safe_text(self.report_id))} | "
                f'{self.generated_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}',
                styles["CoverMeta"],
            ),
        ]
        cover = Table(
            [[cover_content]],
            colWidths=[178 * mm],
            rowHeights=[59 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#17344F")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#345A78")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 15 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 15 * mm),
                ]
            ),
        )
        story.extend(
            [
                cover,
                Spacer(1, 7 * mm),
                Paragraph(
                    "<b>Scientific interpretation rule.</b> Physics calculations, ML predictions, engineering "
                    "screens, and experimental measurements are distinct evidence classes. Unavailable results "
                    "are not replaced with assumed values. Review each section's provenance, confidence, and "
                    "limitations before making experimental or manufacturing decisions.",
                    styles["Small"],
                ),
                Spacer(1, 4 * mm),
            ]
        )
        for section in self._sections(include_figures=False):
            metadata = Paragraph(
                f'<b>{escape(_pdf_safe_text(CLASSIFICATION_LABELS[section.classification]))}</b> | '
                f'Confidence: {escape(_pdf_safe_text(section.confidence))} | '
                f'Provenance: {escape(_pdf_safe_text(section.provenance))}',
                styles["Badge"],
            )
            block: list[Any] = [
                Paragraph(escape(_pdf_safe_text(section.title)), styles["SectionTitle"]),
                metadata,
                Spacer(1, 2 * mm),
            ]
            if section.rows:
                table_data = [[Paragraph("Result", styles["Small"]), Paragraph("Value", styles["Small"])]]
                table_data.extend(
                    [
                        Paragraph(escape(_pdf_safe_text(key)), styles["Cell"]),
                        Paragraph(escape(_pdf_safe_text(value)), styles["Cell"]),
                    ]
                    for key, value in section.rows
                )
                table = Table(table_data, colWidths=[54 * mm, 124 * mm], repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF2F7")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#526178")),
                            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD7E4")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 5),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                block.append(table)
            else:
                block.append(Paragraph("No validated result values are available for this section.", styles["Small"]))
            block.extend(
                [
                    Spacer(1, 2 * mm),
                    Paragraph(
                        "<b>Limitations:</b> "
                        + " | ".join(escape(_pdf_safe_text(item)) for item in section.limitations),
                        styles["Small"],
                    ),
                ]
            )
            # Keep short sections together; long tables can paginate naturally.
            story.extend(block if len(section.rows) > 8 else [KeepTogether(block)])
        story.extend(
            [
                Spacer(1, 8 * mm),
                Paragraph(
                    "This decision-support report does not replace thermodynamic database licensing, instrument "
                    "calibration, metallographic validation, qualified process development, or expert review.",
                    styles["Small"],
                ),
            ]
        )

        report_id = self.report_id

        def draw_page(canvas: Any, doc: Any) -> None:
            canvas.saveState()
            canvas.setStrokeColor(colors.HexColor("#D8E1EC"))
            canvas.line(16 * mm, 13 * mm, 194 * mm, 13 * mm)
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor("#718096"))
            canvas.drawString(16 * mm, 9 * mm, f"CuAlNi-DigitalTwin Pro - {report_id}")
            canvas.drawRightString(194 * mm, 9 * mm, f"Page {doc.page}")
            canvas.restoreState()

        document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
        return buffer.getvalue()

    def write(self, output_directory: str | Path, *, stem: str = "CuAlNi_analysis") -> dict[str, Path]:
        """Write HTML, PDF, and JSON atomically enough for a local desktop workflow."""

        directory = Path(output_directory)
        directory.mkdir(parents=True, exist_ok=True)
        outputs = {
            "html": directory / f"{stem}.html",
            "pdf": directory / f"{stem}.pdf",
            "json": directory / f"{stem}.json",
        }
        content = {
            "html": self.to_html_bytes(),
            "pdf": self.to_pdf_bytes(),
            "json": self.to_json_bytes(),
        }
        for key, path in outputs.items():
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(content[key])
            temporary.replace(path)
        return outputs


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{{ title }}</title>
  <style>{{ css }}</style>
  {% if plotly_javascript %}<script>{{ plotly_javascript|safe }}</script>{% endif %}
</head>
<body>
  <main>
    <header class="cover">
      <div class="kicker">CUALNI-DIGITALTWIN PRO</div>
      <h1>{{ title }}</h1>
      <p class="subtitle">Composition to crystal structure, phase stability, properties, and additive-manufacturing screening.</p>
      <div class="meta"><span>Report {{ report_id }}</span><span>{{ generated_at }}</span><span>Schema 1.0</span></div>
    </header>
    <div class="provenance-banner"><b>Scientific interpretation rule.</b> Physics calculations, ML predictions,
      engineering screens, and experimental measurements are distinct evidence classes. Unavailable results are
      never replaced with assumed values.</div>
    {% for section in sections %}
    <section class="section" id="{{ section.key }}">
      <h2>{{ section.title }}</h2>
      <div class="badges">
        <span class="badge {{ section.classification }}">{{ labels[section.classification] }}</span>
        <span class="badge unavailable">Confidence: {{ section.confidence }}</span>
      </div>
      <p><b>Provenance:</b> {{ section.provenance }}</p>
      {% if section.rows %}
      <table><thead><tr><th>Result</th><th>Value</th></tr></thead><tbody>
        {% for key, value in section.rows %}<tr><td>{{ key }}</td><td>{{ value }}</td></tr>{% endfor %}
      </tbody></table>
      {% else %}<div class="empty">No validated result values are available for this section.</div>{% endif %}
      {% for figure in section.figure_html %}<div class="figure">{{ figure|safe }}</div>{% endfor %}
      <div class="limitations"><b>Limitations:</b> {{ section.limitations|join(' · ') }}</div>
    </section>
    {% endfor %}
    <p class="footer-note">This decision-support report does not replace thermodynamic database licensing,
      instrument calibration, metallographic validation, qualified process development, or expert review.</p>
  </main>
</body>
</html>"""
