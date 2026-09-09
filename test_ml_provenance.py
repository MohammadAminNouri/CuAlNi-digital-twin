"""Synthetic numerical fixtures here test software behavior, not alloy science."""

import json

import numpy as np
import pandas as pd
import pytest

from machine_learning.dataset import DatasetValidationError, load_dataset
from machine_learning.ensemble import PropertyModelSet, TransformationEnsemble


def _synthetic_bundle(tmp_path):
    rows = 15
    al = np.linspace(12.0, 15.0, rows)
    ni = np.linspace(3.0, 5.0, rows)
    cu = 100.0 - al - ni
    frame = pd.DataFrame(
        {
            "Cu_wt_pct": cu,
            "Al_wt_pct": al,
            "Ni_wt_pct": ni,
            "Ms_C": 300.0 - 8.0 * al + 2.0 * ni,
            "hardness_HV": 80.0 + 4.0 * al + ni,
        }
    )
    csv_path = tmp_path / "synthetic.csv"
    frame.to_csv(csv_path, index=False)
    metadata = {
        "title": "Synthetic software-test fixture",
        "source": "Generated inside test_ml_provenance.py",
        "license": "CC0-1.0",
        "measurement_method": "None; deterministic software test",
        "units": {
            "Cu_wt_pct": "wt%",
            "Al_wt_pct": "wt%",
            "Ni_wt_pct": "wt%",
            "Ms_C": "degC",
            "hardness_HV": "HV",
        },
        "retrieved_at": "2026-09-05",
        "synthetic": True,
        "allow_training": True,
    }
    metadata_path = tmp_path / "synthetic.metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return load_dataset(csv_path, metadata_path)


def test_synthetic_training_is_disabled_by_default(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    with pytest.raises(DatasetValidationError, match="Synthetic/example data"):
        bundle.training_frame("Ms_C")


def test_ensemble_prediction_has_interval_and_provenance(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    model = TransformationEnsemble(
        "Ms_C", model_names=("linear", "ridge"), n_splits=3
    ).fit(bundle, allow_synthetic=True)
    query = pd.DataFrame(
        {"Cu_wt_pct": [82.0], "Al_wt_pct": [14.0], "Ni_wt_pct": [4.0]}
    )
    result = model.predict(query)
    assert result.classification == "machine_learning_prediction"
    assert result.lower[0] <= result.values[0] <= result.upper[0]
    assert model.model_card()["dataset_sha256"] == bundle.sha256


def test_property_model_set_trains_hardness_target(tmp_path):
    bundle = _synthetic_bundle(tmp_path)
    models = PropertyModelSet(model_names=("linear",), random_state=7).fit(
        bundle, allow_synthetic=True
    )
    assert "hardness_HV" in models.models_
    assert models.models_["hardness_HV"].target_unit_ == "HV"
    assert "yield_strength_MPa" in models.unavailable_
