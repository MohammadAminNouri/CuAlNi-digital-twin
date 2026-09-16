from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.streamlit
def test_streamlit_landing_page_starts_without_scientific_inputs():
    app = Path(__file__).resolve().parents[1] / "app.py"
    test = AppTest.from_file(str(app), default_timeout=30).run()
    assert not test.exception
    assert any("CuAlNi-DigitalTwin Pro" in item.value for item in test.markdown)
    test.button[0].click().run(timeout=60)
    assert not test.exception
    assert len(test.tabs) == 7
    assert len(test.get("download_button")) == 3
    bundle = test.session_state["report_bundle"]
    assert bundle["pdf"].startswith(b"%PDF")
    test.run(timeout=60)
    assert not test.exception
    assert test.session_state["report_bundle"]["id"] == bundle["id"]
    assert test.session_state["report_bundle"]["json"] == bundle["json"]
    test.button[0].click().run(timeout=60)
    assert not test.exception
    assert test.session_state["report_bundle"]["id"] != bundle["id"]
    next(item for item in test.number_input if item.label == "Cu").set_value(80.0)
    test.button[0].click().run(timeout=60)
    assert not test.exception
    assert any("last completed analysis" in item.value for item in test.warning)


@pytest.mark.streamlit
def test_invalid_composition_is_rejected_and_can_be_corrected():
    app = Path(__file__).resolve().parents[1] / "app.py"
    test = AppTest.from_file(str(app), default_timeout=30).run()
    copper = next(item for item in test.number_input if item.label == "Cu")
    copper.set_value(80.0)
    test.button[0].click().run(timeout=60)
    assert not test.exception
    assert any("Composition totals" in item.value for item in test.error)
    assert len(test.tabs) == 0
    copper = next(item for item in test.number_input if item.label == "Cu")
    copper.set_value(81.4)
    test.button[0].click().run(timeout=60)
    assert not test.exception
    assert len(test.tabs) == 7
    assert test.session_state["analysis"]["composition"]["status"] == "available"


@pytest.mark.streamlit
def test_invalid_temperature_range_is_rejected():
    app = Path(__file__).resolve().parents[1] / "app.py"
    test = AppTest.from_file(str(app), default_timeout=30).run()
    maximum = next(item for item in test.number_input if item.label == "Maximum temperature (°C)")
    maximum.set_value(0.0)
    test.button[0].click().run(timeout=60)
    assert not test.exception
    assert any("Maximum temperature must exceed" in item.value for item in test.error)
    assert len(test.tabs) == 0
