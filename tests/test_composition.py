import numpy as np
import pytest

from composition.converter import AlloyComposition, CompositionError, convert_atomic_to_wt
from composition.descriptors import compute_descriptors, electron_concentration
from scientific import Provenance


def test_primary_alloy_basis_conversion_is_conservative():
    alloy = AlloyComposition.from_wt_percent({"Cu": 81.4, "Al": 14.3, "Ni": 4.3})
    assert np.isclose(sum(alloy.weight_percent.values()), 100.0)
    assert np.isclose(sum(alloy.atomic_percent.values()), 100.0)
    recovered = convert_atomic_to_wt(alloy.atomic_percent)
    assert all(np.isclose(recovered[key], alloy.weight_percent[key]) for key in recovered)


def test_primary_alloy_atomic_percent_regression():
    alloy = AlloyComposition.from_wt_percent({"Cu": 81.4, "Al": 14.3, "Ni": 4.3})
    assert np.isclose(alloy.atomic_percent["Cu"], 67.9838130488)
    assert np.isclose(alloy.atomic_percent["Al"], 28.1279876269)
    assert np.isclose(alloy.atomic_percent["Ni"], 3.8881993243)


def test_bad_compositions_are_rejected():
    with pytest.raises(CompositionError):
        AlloyComposition.from_wt_percent({"Cu": -1, "Al": 50, "Ni": 51})
    with pytest.raises(CompositionError):
        AlloyComposition.from_wt_percent({"Cu": 0, "Al": 0, "Ni": 0})
    with pytest.raises(CompositionError):
        AlloyComposition.from_wt_percent({"Fe": 100})


def test_hume_rothery_ea_is_unavailable_without_convention():
    alloy = AlloyComposition.from_wt_percent({"Cu": 81.4, "Al": 14.3, "Ni": 4.3})
    value = compute_descriptors(alloy)["hume_rothery_e_a"].to_dict()
    assert value["value"] is None
    assert value["metadata"]["classification"] == "not_available"


def test_user_ea_convention_is_explicit_and_reproducible():
    alloy = AlloyComposition.from_wt_percent({"Cu": 81.4, "Al": 14.3, "Ni": 4.3})
    result = electron_concentration(
        alloy,
        {"Cu": 1, "Al": 3, "Ni": 0},
        convention_name="software-test convention",
        source=Provenance("test", "not scientific data"),
    )
    expected = alloy.mole_fraction["Cu"] + 3 * alloy.mole_fraction["Al"]
    assert np.isclose(result.value, expected)
    assert result.details["convention"] == "software-test convention"

