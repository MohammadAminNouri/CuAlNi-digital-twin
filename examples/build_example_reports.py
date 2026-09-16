"""Build deterministic HTML/PDF reports from the committed composition-only example."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reporting.report_generator import AnalysisReportGenerator  # noqa: E402


def main() -> None:
    example_directory = Path(__file__).resolve().parent
    analysis = json.loads(
        (example_directory / "primary_alloy_composition_only.json").read_text(encoding="utf-8")
    )
    generator = AnalysisReportGenerator(
        analysis,
        title="Cu-14.3Al-4.3Ni wt% - composition-only audit",
        report_id="CAN-PRIMARY-COMPOSITION-ONLY",
        generated_at=datetime(2026, 9, 9, tzinfo=timezone.utc),
    )
    (example_directory / "primary_alloy_composition_only.html").write_bytes(
        generator.to_html_bytes()
    )
    (example_directory / "primary_alloy_composition_only.pdf").write_bytes(
        generator.to_pdf_bytes()
    )


if __name__ == "__main__":
    main()
