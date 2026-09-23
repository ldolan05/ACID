"""Column alignment and metadata preservation for full line lists."""
import pickle

import numpy as np
import pytest

from ACID_code import Data, DataList, LineList
from ACID_code.diagnostics.errors import ACIDInputError
from ACID_code.diagnostics.warnings import ACIDDroppedDataWarning


@pytest.fixture
def full_columns():
    return {
        "wavelengths": [5002.0, 5000.0, 5001.0],
        "depths": [0.2, 0.1, 1.0],
        "spec_ion": ["Fe 1", "Ti 2", "Cr 1"],
        "lande_factor": [1.2, 0.8, 1.4],
    }


@pytest.mark.parametrize("input_format", ["dict", "list", "array"])
def test_full_columns_share_sort_and_validity_mask(full_columns, input_format):
    source = full_columns
    if input_format != "dict":
        source = list(source.values())
    if input_format == "array":
        # A mixed-type array promotes all columns to strings; numeric columns
        # must still be converted and sorted numerically.
        source = np.array(source)
    with pytest.warns(ACIDDroppedDataWarning, match="includes 1"):
        lines = LineList(source, full=True)
    expected = ([5000.0, 5002.0], [0.1, 0.2], ["Ti 2", "Fe 1"], [0.8, 1.2])
    assert list(lines.ll) == list(full_columns)
    for index, (key, values) in enumerate(zip(full_columns, expected)):
        np.testing.assert_array_equal(lines[key], values)
        assert lines[index] is lines[key]
        assert lines[np.int64(index)] is lines[key]
    assert lines[2].dtype.kind == "U"
    assert lines[3].dtype.kind == "f"
    assert len(tuple(lines)) == 4
    with pytest.raises(ACIDInputError):
        _ = lines[4]

    with pytest.warns(ACIDDroppedDataWarning):
        *columns, mask = LineList.validate_linelist(source, full=True, return_mask=True)
    for actual, values in zip(columns, expected):
        np.testing.assert_array_equal(actual, values)
    np.testing.assert_array_equal(mask, [True, False, True])


def test_vald_full_read_maps_columns_and_strips_species_quotes(tmp_path):
    filename = tmp_path / "vald.txt"
    filename.write_text(
        "header\n" * 4
        + "'Fe 1',5002,0,0,0,0,0,0,1.2,0.2,'reference'\n"
        + "'Ti 2',5000,0,0,0,0,0,0,0.8,0.1,'reference'\n"
        + "'Cr 1',5001,0,0,0,0,0,0,1.4,1.0,'reference'\n"
        + "'Ni 1',5003,0,0,0,0,0,0,unknown,0.3,'reference'\n"
        + "'reference',footer,0,0,0,0,0,0,text,text,'reference'\n"
    )
    with pytest.warns(ACIDDroppedDataWarning, match="includes 2"):
        lines = LineList(str(filename), full=True)
    np.testing.assert_array_equal(lines[0], [5000, 5002])
    np.testing.assert_array_equal(lines[1], [0.1, 0.2])
    np.testing.assert_array_equal(lines[2], ["Ti 2", "Fe 1"])
    np.testing.assert_array_equal(lines[3], [0.8, 1.2])
    with pytest.warns(ACIDDroppedDataWarning, match="includes 1"):
        basic = LineList(str(filename))
    assert list(basic.ll) == ["wavelengths", "depths"]
    np.testing.assert_array_equal(basic[0], [5000, 5002, 5003])


@pytest.mark.parametrize("extra, message", [
    ({}, "missing spec_ion, lande_factor"),
    ({"spec_ion": ["Fe 1", "Ti 2"], "lande_factor": [1.0]}, "same length"),
    ({"spec_ion": [["Fe 1", "Ti 2"]], "lande_factor": [1.0, 2.0]}, "one-dimensional"),
    ({"spec_ion": ["Fe 1", "Ti 2"], "lande_factor": [1.0, "bad"]}, "Failed to convert"),
])
def test_full_metadata_validation(extra, message):
    with pytest.raises(ACIDInputError, match=message):
        LineList({"wavelengths": [5000, 5001], "depths": [0.1, 0.2], **extra}, full=True)


@pytest.mark.parametrize("species, lande", [(None, 1.0), (" ", 1.0), ("Fe 1", np.inf)])
def test_invalid_metadata_drops_whole_row(species, lande):
    with pytest.warns(ACIDDroppedDataWarning, match="includes 1"):
        lines = LineList([[5000, 5001], [0.1, 0.2], [species, "Ti 2"], [lande, 0.8]], full=True)
    assert [column.tolist() for column in lines] == [[5001], [0.2], ["Ti 2"], [0.8]]


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
