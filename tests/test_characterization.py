"""Algorithm tests use generated signals and make no scientific alloy claims."""

import numpy as np
import pandas as pd

from characterization.dsc_analysis import analyze_dsc
from characterization.eds_analysis import analyze_eds
from characterization.sem_analysis import analyze_sem
from characterization.xrd_analysis import analyze_xrd


def test_xrd_peak_detection_and_matching():
    angle = np.linspace(20.0, 80.0, 3001)
    intensity = 5.0 + 100.0 * np.exp(-0.5 * ((angle - 40.0) / 0.08) ** 2)
    intensity += 60.0 * np.exp(-0.5 * ((angle - 60.0) / 0.12) ** 2)
    references = pd.DataFrame(
        {"two_theta_deg": [40.0, 60.0], "phase": ["test_A", "test_B"], "hkl": ["100", "110"]}
    )
    result = analyze_xrd(
        pd.DataFrame({"two_theta_deg": angle, "intensity": intensity}),
        references,
        minimum_prominence_pct=5.0,
    )
    assert len(result.peaks) == 2
    assert len(result.matches) == 2
    assert result.to_dict()["phase_matching_probability"] is None


def test_eds_normalization_and_target_deviation():
    data = pd.DataFrame(
        {
            "Cu_wt_pct": [81.0, 82.0, 83.0],
            "Al_wt_pct": [15.0, 14.0, 13.0],
            "Ni_wt_pct": [4.0, 4.0, 4.0],
        }
    )
    result = analyze_eds(data, {"Cu": 82.0, "Al": 14.0, "Ni": 4.0})
    assert np.allclose(result.normalized_points_wt_pct.sum(axis=1), 100.0)
    assert result.confidence["n_points"] == 3


def test_dsc_detects_both_scan_directions():
    heating_t = np.linspace(0.0, 200.0, 201)
    cooling_t = np.linspace(200.0, 0.0, 201)
    heating_signal = np.exp(-0.5 * ((heating_t - 110.0) / 8.0) ** 2)
    cooling_signal = -np.exp(-0.5 * ((cooling_t - 80.0) / 7.0) ** 2)
    frame = pd.DataFrame(
        {
            "temperature_C": np.r_[heating_t, cooling_t],
            "heat_flow": np.r_[heating_signal, cooling_signal],
        }
    )
    result = analyze_dsc(frame)
    assert all(result.temperatures_C[name] is not None for name in ("Ms", "Mf", "As", "Af"))
    assert result.temperatures_C["Mf"] < result.temperatures_C["Ms"]
    assert result.temperatures_C["As"] < result.temperatures_C["Af"]


def test_sem_unsupervised_regions_are_not_called_phases():
    image = np.tile(np.linspace(0.0, 1.0, 64), (64, 1))
    result = analyze_sem(image, n_regions=3)
    assert result.confidence["phase_classification_available"] is False
    assert set(result.area_fractions_2d) == {"region_1", "region_2", "region_3"}
    assert np.isclose(sum(result.area_fractions_2d.values()), 1.0)
