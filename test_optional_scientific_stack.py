"""Integration checks for dependencies installed by the complete requirements file."""

import inspect

import pytest

from application.analysis_service import AnalysisInputs, run_complete_analysis

TEST_CIF = b"""data_software_test
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   3.0
_cell_length_b   3.0
_cell_length_c   3.0
_cell_angle_alpha   90
_cell_angle_beta    90
_cell_angle_gamma   90
loop_
_space_group_symop_operation_xyz
'x, y, z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Cu1 Cu 0.0 0.0 0.0 1.0
Cu2 Cu 0.5 0.5 0.5 1.0
"""


def test_complete_stack_cif_symmetry_and_xrd_pipeline():
    pytest.importorskip("pymatgen")
    pytest.importorskip("spglib")
    result = run_complete_analysis(
        AnalysisInputs(
            {"Cu": 81.4, "Al": 14.3, "Ni": 4.3},
            structure_files={"software-test structure": TEST_CIF},
            structure_metadata={
                "software-test structure": {
                    "filename": "software-test.cif",
                    "citation": "Generated crystallographic integration fixture; not scientific alloy data",
                }
            },
        )
    )
    assert result["crystallography"]["status"] == "available"
    assert result["xrd"]["status"] == "available"
    structure = result["crystallography"]["structures"]["software-test structure"]
    assert structure["structure"]["sites"]
    assert structure["symmetry"]["space_group_number"] == 229
    assert result["xrd"]["patterns"]["software-test structure"]["peaks"]


def test_scheil_adapter_matches_installed_public_api():
    scheil = pytest.importorskip("scheil")
    parameters = inspect.signature(scheil.simulate_scheil_solidification).parameters
    assert {"dbf", "comps", "phases", "composition", "start_temperature", "stop"}.issubset(parameters)


def test_optional_model_packages_are_importable():
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    pytest.importorskip("shap")
