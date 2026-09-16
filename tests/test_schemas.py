import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_json_templates_parse_and_contain_no_am_calibration_values():
    cif = json.loads((ROOT / "data/templates/cif_metadata_schema.json").read_text(encoding="utf-8"))
    am = json.loads((ROOT / "data/templates/am_screening_profile_schema.json").read_text(encoding="utf-8"))
    assert cif["type"] == "object"
    assert am["type"] == "object"
    assert "preferred_value" not in am.get("examples", {})

