from application.analysis_service import AnalysisInputs, run_complete_analysis


def test_composition_only_analysis_fails_closed_for_dependent_science():
    result = run_complete_analysis(
        AnalysisInputs({"Cu": 81.4, "Al": 14.3, "Ni": 4.3})
    )
    assert result["composition"]["status"] == "available"
    for section in ("thermodynamics", "crystallography", "xrd", "ml", "characterization", "am"):
        assert result[section]["status"] == "unavailable"
        assert result[section]["classification"] == "unavailable"
        assert result[section]["limitations"]
    assert result["crystallography"]["literature_references"]
    assert result["phases"]["phase_reference_catalog"]["status"] == "reference_only"


def test_cited_ea_convention_reaches_result_envelope():
    result = run_complete_analysis(
        AnalysisInputs(
            {"Cu": 81.4, "Al": 14.3, "Ni": 4.3},
            e_a_valence_map={"Cu": 1, "Al": 3, "Ni": 0},
            e_a_convention_name="software test only",
            e_a_citation="test_application.py; not a scientific convention",
        )
    )
    descriptor = result["composition"]["descriptors"]["hume_rothery_e_a"]
    assert descriptor["value"] is not None
    assert descriptor["details"]["convention"] == "software test only"


def test_sourced_am_profile_is_parsed_and_method_status_is_retained():
    profile = {
        "metrics": {
            "test_metric": {
                "value": 5.0,
                "unit": "1",
                "classification": "synthetic_test_input",
                "provenance": "test_application.py; not scientific calibration",
            }
        },
        "profile": {
            "name": "software test only",
            "version": "1",
            "provenance": "test_application.py",
            "applicability": "unit test only",
            "rules": [
                {
                    "name": "test_metric",
                    "preferred_value": 0.0,
                    "adverse_value": 10.0,
                    "weight": 1.0,
                    "unit": "1",
                    "rationale": "tests decreasing normalization",
                    "source": "test_application.py",
                }
            ],
        },
    }
    result = run_complete_analysis(
        AnalysisInputs(
            {"Cu": 81.4, "Al": 14.3, "Ni": 4.3},
            am_normalization_profile=profile,
        )
    )["am"]
    assert result["status"] == "available"
    assert result["method_status"] == "complete"
    assert result["score_0_to_10"] == 5.0
