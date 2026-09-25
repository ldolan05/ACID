"""Column alignment and metadata preservation for full line lists."""
import pickle

import numpy as np
import pytest

from ACID_code import Data, DataList, LineList


def test_full_metadata_survives_copies_data_reset_and_save_load(tmp_path):
    lines = LineList([[5002, 5000], [0.2, 0.1], ["Fe 1", "Ti 2"], [1.2, 0.8]], full=True)
    for source in (lines, lines.ll, list(lines)):
        copied = LineList(source)
        assert list(copied.ll) == list(lines.ll)
        for key in lines.ll:
            np.testing.assert_array_equal(copied[key], lines[key])
            assert not np.shares_memory(copied[key], lines[key])

    data = Data()
    data.config.order = 20
    # String lists in metadata must pass Data's runtime type checking too.
    data.linelist = {key: value.tolist() for key, value in lines.ll.items()}
    data.reset()
    filename = tmp_path / "data.pkl"
    data.save(str(filename))
    with filename.open("rb") as stream:
        stored = pickle.load(stream)["_linelist"]
    assert type(stored) is dict
    assert list(stored) == list(lines.ll)
    loaded = Data.load(str(filename))
    for key in lines.ll:
        np.testing.assert_array_equal(loaded.linelist[key], lines[key])

    datalist = DataList.from_datalist([data])
    datalist.save(str(tmp_path))
    with (tmp_path / "datalist.pkl").open("rb") as stream:
        stored = pickle.load(stream)["data_list"][0]["_linelist"]
    assert type(stored) is dict
    assert list(stored) == list(lines.ll)
    loaded_list = DataList.load(str(tmp_path))
    for key in lines.ll:
        np.testing.assert_array_equal(loaded_list[20].linelist[key], lines[key])


def test_data_retains_constructed_linelist_and_loads_legacy_storage(tmp_path):
    import pickle

    data = Data()
    data.linelist = [[5002.0, 5000.0], [0.2, 0.1]]
    lines = data.linelist
    assert data.linelist is lines

    for stored in (lines, lines.ll):
        data._linelist = stored
        filename = str(tmp_path / "data.pkl")
        data.save(filename)
        assert data._linelist is stored
        with open(filename, "rb") as stream:
            payload = pickle.load(stream)
        assert type(payload["_linelist"]) is dict
        assert set(payload["_linelist"]) == {"wavelengths", "depths"}
        assert all(isinstance(values, np.ndarray)
                   for values in payload["_linelist"].values())
        loaded = Data.load(filename)
        restored = loaded.linelist
        assert isinstance(restored, LineList)
        assert loaded.linelist is restored
        np.testing.assert_array_equal(restored[0], [5000.0, 5002.0])
        np.testing.assert_array_equal(restored[1], [0.1, 0.2])


def test_data_velocity_and_linelist_overwrites_clear_derived_state(synthetic_spectrum):
    # Populate derived state, then change each dependency and check it is invalidated.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    data = Data()
    data.set_inputs(wavelengths, flux, errors, sn)
    data.velocities = velocities
    data.linelist = linelist
    data.alpha["derived"] = np.ones((2, 2))
    data.velocities = velocities + 1
    assert data.alpha == {}

    data.alpha["derived"] = np.ones((2, 2))
    data.config.profile_groups = np.array([0, 1])
    data.ll_mask = np.array([0, 1])
    data.profile_groups = np.array([0, 1])
    changed_linelist = {"wavelengths": linelist["wavelengths"],
                        "depths": linelist["depths"] * 0.9}
    data.linelist = changed_linelist
    assert data.alpha == {}
    np.testing.assert_array_equal(data.config.profile_groups, [0, 1])
    assert data.profile_groups is None
    stored_depths = data.linelist["depths"].copy()
    data.linelist = None
    np.testing.assert_array_equal(data.linelist["depths"], stored_depths)

    with pytest.raises(ValueError, match="finite"):
        data.velocities = np.array([0.0, np.nan])


def test_profile_groups_are_validated_when_clipped_groups_are_set(synthetic_spectrum):
    *_, linelist = synthetic_spectrum
    data = Data()
    data.linelist = linelist
    data.config.profile_groups = np.array([0, 1])

    np.testing.assert_array_equal(data.linelist["wavelengths"], linelist["wavelengths"])

    data.config.profile_groups = np.array([0])
    # The full linelist may be accessed before clipping determines the expected length.
    np.testing.assert_array_equal(data.linelist["wavelengths"], linelist["wavelengths"])
    data.ll_mask = np.array([0, 1])
    with pytest.raises(ValueError, match="after S/N and wavelength clipping"):
        data.profile_groups = data.config.profile_groups
    data.ll_mask = np.array([0])
    data.profile_groups = data.config.profile_groups
    np.testing.assert_array_equal(data.profile_groups, [0])

