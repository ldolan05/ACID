"""Column alignment and metadata preservation for full line lists."""

import numpy as np
import pytest

from ACID_code import LineList
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


def test_linelist_sorts_and_rejects_invalid_shapes():
    # Line lists are stored in wavelength order regardless of their input ordering.
    linelist = LineList({"wavelengths": np.array([5002.0, 5000.0]),
                         "depths": np.array([0.2, 0.1])})
    wavelengths, depths = linelist

    np.testing.assert_array_equal(wavelengths, [5000.0, 5002.0])
    np.testing.assert_array_equal(depths, [0.1, 0.2])
    with pytest.raises(ValueError, match="same length"):
        LineList([[1, 2], [0.1]])


@pytest.mark.parametrize("linelist", [
    [[5002.0, 5000.0], [0.2, 0.1]],
    np.array([[5002.0, 5000.0], [0.2, 0.1]]),
    {"wavelengths": [5002.0, 5000.0], "depths": [0.2, 0.1], "ignored": True},
    LineList({"wavelengths": np.array([5002.0, 5000.0]),
              "depths": np.array([0.2, 0.1])}),
])
def test_linelist_accepts_each_documented_in_memory_format(linelist):
    # All public in-memory forms should produce the same sorted pair of arrays.
    wavelengths, depths = LineList(linelist)

    np.testing.assert_array_equal(wavelengths, [5000.0, 5002.0])
    np.testing.assert_array_equal(depths, [0.1, 0.2])


def test_linelist_file_indexing_and_invalid_line_removal(linelist_path):
    # A file-backed line list should be readable through the same validation route.
    line_list = LineList(str(linelist_path))

    assert line_list[0].shape == line_list[1].shape
    assert line_list["wavelengths"].ndim == 1
    with pytest.raises(ACIDInputError):
        _ = line_list[2]

    # Invalid depths and wavelengths are removed together, retaining the validity mask.
    kept_wavelengths, kept_depths, mask = LineList.validate_linelist(
        [np.array([5000.0, np.nan, 5002.0, 5003.0]),
         np.array([0.1, 0.2, -0.1, 1.0])], return_mask=True,
    )
    np.testing.assert_array_equal(mask, [True, False, False, False])
    np.testing.assert_array_equal(kept_wavelengths, [5000.0])
    np.testing.assert_array_equal(kept_depths, [0.1])


def test_linelist_constructor_removes_invalid_lines_and_owns_arrays():
    wavelengths = np.array([5002.0, np.nan, 5000.0, -1.0, 5003.0, 5004.0])
    depths = np.array([0.2, 0.3, 0.1, 0.4, 1.0, -0.1])
    with pytest.warns(ACIDDroppedDataWarning, match="4"):
        lines = LineList({"wavelengths": wavelengths, "depths": depths})
    wavelengths[:] = 1
    depths[:] = 0
    np.testing.assert_array_equal(lines[0], [5000.0, 5002.0])
    np.testing.assert_array_equal(lines[1], [0.1, 0.2])


@pytest.mark.parametrize("linelist, message", [
    (None, "must be provided"),
    ({"wavelengths": [5000]}, "must contain keys"),
    ([[[5000]], [[0.1]]], "one-dimensional"),
    ([[], []], "All lines"),
    ([[5000, -1, np.inf], [1, 0.1, 0.2]], "All lines"),
])
def test_linelist_constructor_rejects_invalid_inputs(linelist, message):
    with pytest.raises(ACIDInputError, match=message):
        LineList(linelist)

