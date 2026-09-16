"""Generate the composition-only audit example for Cu-14.3Al-4.3Ni wt%."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from application.analysis_service import AnalysisInputs, run_complete_analysis  # noqa: E402


def main() -> Path:
    result = run_complete_analysis(
        AnalysisInputs(composition_wt_percent={"Cu": 81.4, "Al": 14.3, "Ni": 4.3})
    )
    output = Path(__file__).with_name("primary_alloy_composition_only.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)
    return output


if __name__ == "__main__":
    main()
