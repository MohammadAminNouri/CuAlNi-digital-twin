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
