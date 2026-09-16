"""Tests verify transparent calculations, not Cu-Al-Ni acceptance limits."""

import numpy as np

from additive_manufacturing.cracking import kou_cracking_index
from additive_manufacturing.printability import (
    MetricRule,
    ScreeningMetric,
    ScreeningProfile,
    assess_printability,
)
from additive_manufacturing.solidification import analyze_solidification_path


def test_solidification_range_uses_declared_fraction_cutoffs():
    fraction = np.linspace(0.0, 1.0, 101)
    temperature = 1100.0 - 200.0 * fraction
    result = analyze_solidification_path(
        temperature,
        fraction,
        source={"method": "generated linear test path; not alloy data"},
    )
    assert np.isclose(result.freezing_range_C, 196.0)
    indicator = kou_cracking_index(temperature, fraction)
    assert indicator.value > 0


def test_printability_is_unavailable_without_profile():
    result = assess_printability({"freezing_range": 100.0})
    assert result.score_0_to_10 is None
    assert result.classification == "engineering_screening"


def test_explicit_profile_produces_auditable_score():
    profile = ScreeningProfile(
        name="software-test profile",
        version="1",
        rules=(
            MetricRule(
                name="metric",
                preferred_value=0.0,
                adverse_value=10.0,
                weight=1.0,
                unit="1",
                rationale="Tests monotonic normalization only",
                source="test_additive_manufacturing.py; not scientific calibration",
            ),
        ),
        provenance="Synthetic software-test profile",
        applicability="Unit tests only",
    )
    result = assess_printability(
        {
            "metric": ScreeningMetric(
                5.0,
                "1",
                "synthetic_test_input",
                "test_additive_manufacturing.py",
            )
        },
        profile,
    )
    assert np.isclose(result.score_0_to_10, 5.0)
    assert result.coverage == 1.0
