#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code.diagnostics.errors import ACIDInputError, ACIDStateError
from ACID_code.diagnostics.warnings import ACIDDroppedDataWarning
from ACID_code import Config, Data, DataList, LineList, MaskingLines
from ACID_code import utils


@pytest.mark.parametrize("saved_order, folder, expected", [
    (None, "order_20", 20), (None, "order_0", 0),
    (None, "order_-2", -2), (0, "order_20", 0),
    (12, "order_20", 12), (None, "results", None),
    (None, "order_invalid", None),
])
def test_data_load_infers_only_unset_orders(tmp_path, saved_order, folder, expected):
    data = Data()
    data.config = Config(order=saved_order, verbose=0)
    filename = tmp_path / folder / "data.pkl"
    data.save(str(filename))
    assert data.config.order == saved_order
    loaded = Data.load(str(filename))
    assert loaded.config.order == expected
    assert Data().from_dict(loaded.to_dict()).config.order == expected


def test_datalist_load_infers_current_directory_after_move(tmp_path):
    data = Data()
    data.config = Config(verbose=0)
    old_folder = tmp_path / "order_1"
    data.save(str(old_folder / "data.pkl"))
    old_folder.rename(tmp_path / "order_20")

    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded.orders.tolist() == [20]
    assert 20 in loaded.order_range
    assert loaded[20].config.save_path == str(tmp_path / "order_20" / "data.pkl")
    assert loaded[20].config.sampler_path == str(tmp_path / "order_20" / "sampler.h5")
    assert loaded[20].sampler is None
    assert not (tmp_path / "order_1").exists()
    assert not (tmp_path / "order_None").exists()
    assert Data.load(loaded[20].config.save_path).config.order == 20


def test_packed_datalist_infers_unset_order_from_stored_path(tmp_path):
    import pickle

    data = Data()
    data.config = Config(verbose=0)
    data.save(str(tmp_path / "order_12" / "data.pkl"))
    with open(tmp_path / "datalist.pkl", "wb") as stream:
        pickle.dump({"data_list": [data.to_dict()], "verbose": 0}, stream)
    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded.orders.tolist() == [12]
    assert 12 in loaded.order_range
    assert DataList.load(str(tmp_path), verbose=0).orders.tolist() == [12]


def test_datalist_rejects_unset_orders_before_building_paths(tmp_path, completed_datalist):
    data = Data()
    assert data.config.order is None
    with pytest.raises(ValueError, match="explicit order"):
        DataList.from_datalist([data], verbose=0)
    with pytest.raises(ValueError, match="explicit order"):
        completed_datalist.append(data, extend=True)
    with pytest.raises(ValueError, match="Cannot determine"):
        DataList._update_paths_for_data(data, str(tmp_path))
    assert not (tmp_path / "order_None").exists()
    completed_datalist.append(data, force_order=23, extend=True)
    assert completed_datalist[23] is data


def test_config_priorities_properties_and_environment(monkeypatch):
    # Low-priority values should fill gaps, while high-priority values win conflicts.
    config = Config(verbose="off", order=1)
    config.update_lowpri(order=2, poly_ord=4)
    config.update_hipri(order=3)

    assert config.verbose == 0
    assert config.order == 3
    assert config.poly_ord == 4
    verbose_config = Config()
    verbose_config.update_lowpri(verbose="low")
    verbose_config.update_hipri(verbose="high")
    verbose_config.update_lowpri(verbose="off")
    assert verbose_config.verbose == 3
    # Invalid configuration and misplaced Data attributes should fail clearly.
    with pytest.raises(ACIDInputError):
        config.update_hipri(not_a_setting=True)
    with pytest.raises(ACIDInputError):
        config.linelist = []
    with pytest.raises(ACIDInputError):
        config.not_a_setting = True
    config.order = 4
    config.order = None
    assert config.order == 4

    # Environment overrides are intentionally evaluated on attribute access.
    monkeypatch.setenv("_ACID_CONFIG", '{"order": 99}')
    assert config.order == 99


def test_config_dictionary_views_repr_and_verbose_validation(capsys):
    # The compact dictionary contains explicit values; the full view also resolves defaults.
    config = Config(order=7, verbose="medium")
    compact = config.to_dict()
    complete = config.to_full_dict()

    assert compact["order"] == complete["order"] == 7
    assert complete["poly_ord"] == Config.defaults["poly_ord"]
    assert config.pix_chunk == complete["pix_chunk"] == 50
    assert "order: 7" in repr(config)

    # Defaults are printable for interactive inspection, and invalid verbose values fail early.
    Config.print_defaults(use_logger=False)
    assert "poly_ord" in capsys.readouterr().out
    for invalid_verbose in (-1, 5):
        with pytest.raises(ValueError, match="between 0 and 4"):
            config.verbose = invalid_verbose
    with pytest.raises(ValueError, match="not recognised"):
        config.verbose = "loud"
    config.verbose = True
    assert config.verbose == 2
    config.verbose = "low"
    config.verbose = None
    assert config.verbose == 1


def test_config_sampler_path_is_non_destructive_and_sampler_type_is_validated(tmp_path):
    sampler_path = tmp_path / "sampler.h5"
    sampler_path.write_bytes(b"existing sampler")

    config = Config(sampler_path=str(sampler_path))

    assert sampler_path.read_bytes() == b"existing sampler"
    assert config.sampler_path == str(sampler_path)
    with pytest.raises(ValueError, match="sampler_type"):
        Config(sampler_type="nested")


@pytest.mark.parametrize("path_name", ["save_path", "sampler_path", "figure_dir"])
@pytest.mark.parametrize("disabled", ["None", "False", "OFF", "no", "n", "0"])
def test_config_path_disabling_strings_survive_round_trip(tmp_path, monkeypatch, path_name, disabled):
    monkeypatch.chdir(tmp_path)
    config = Config(**{path_name: disabled}, verbose=0)
    assert getattr(config, path_name) is None
    assert not (tmp_path / disabled).exists()

    config.dir = str(tmp_path / "output")
    assert getattr(config, path_name) is None
    restored = Config(**config.to_dict())
    assert getattr(restored, path_name) is None
    # Python None still means "leave unchanged", rather than a disabling string.
    setattr(restored, path_name, None)
    assert getattr(restored, path_name) is None
    restored.dir = str(tmp_path / "moved")
    assert getattr(restored, path_name) is None
    if path_name == "figure_dir":
        assert not (tmp_path / "output" / "figures").exists()
        assert not (tmp_path / "moved" / "figures").exists()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("relocate", [False, True])
def test_disabled_figures_survive_datalist_save_load(tmp_path, packed, relocate):
    root = tmp_path / "original"
    root.mkdir()
    data = Data()
    data.config = Config(figure_dir="None", dir=str(root / "order_20"), order=20, verbose=0)
    data.save()
    assert Data.load(data.config.save_path).config.figure_dir is None
    if packed:
        DataList.from_datalist([data], save_dir=str(root), verbose=0).save()
    if relocate:
        root = root.rename(tmp_path / "relocated")

    loaded = DataList.load(str(root), verbose=0)
    assert loaded[20].config.figure_dir is None
    assert loaded[20].config.dir == str(root / "order_20")
    assert not (root / "order_20" / "figures").exists()
    assert DataList.load(str(root), verbose=0)[20].config.figure_dir is None


def test_masking_lines_accepts_compact_inputs_and_masks_grid():
    # A compact dictionary should expand its default width for each line.
    lines = MaskingLines({"telluric": {"default_width": 100, "lines": [5000.0]}})
    grid = np.array([4990.0, 5000.0, 5010.0])

    # Both the combined and named-mask interfaces should describe the same region.
    assert lines.get_1d_mask_on_grid(grid).tolist() == [False, True, False]
    masks = lines.get_masks(grid)
    assert isinstance(masks, list) and len(masks) == 1
    assert list(lines.get_masks(grid, with_names=True)) == ["telluric"]
    with pytest.raises(ValueError):
        MaskingLines({"bad": {"lines": [5000.0]}})


def test_masking_line_plot_has_one_labelled_series_per_group():
    # Give each group a distinct line and width so plot components are unambiguous.
    config = Config(masking_lines={
        "narrow": {"default_width": 100, "lines": [5000.0]},
        "wide": {"default_width": 500, "lines": [5100.0]},
    })
    fig, ax = config.plot_masking_lines(return_fig=True)

    assert len(ax.lines) == 2
    assert [line.get_label() for line in ax.lines] == ["Narrow line", "Wide line"]
    assert len(ax.patches) == 2
    plt.close(fig)


@pytest.mark.parametrize("line_input, expected_widths", [
    ([(5000.0, 100), (5010.0, 200)], [100, 200]),
    (np.array([[5000.0, 5010.0], [100, 200]]), [100, 200]),
    ([(5000.0,), (5010.0, 200)], [150, 200]),
])
def test_masking_lines_normalises_supported_width_formats(line_input, expected_widths):
    # Tuple, two-row array, and default-width forms all describe the same stored model.
    lines = MaskingLines({"test": {"default_width": 150, "lines": line_input}})

    np.testing.assert_array_equal(lines["test"]["widths"], expected_widths)
    named_masks = lines.get_masks(np.array([4990.0, 5000.0, 5010.0]), with_names=True)
    assert set(named_masks) == {"test"}


def test_masking_lines_rejects_empty_and_mismatched_definitions():
    # Invalid definitions should fail during construction, before they can mask spectra.
    with pytest.raises(ValueError, match="empty"):
        MaskingLines({"bad": {"default_width": 100, "lines": []}})
    with pytest.raises(ValueError, match="same"):
        MaskingLines({"bad": {"lines": [[5000.0, 5010.0], [100]]}})


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


def test_data_without_linelist_save_load(tmp_path):
    data = Data()
    assert data.to_dict()["_linelist"] is None
    filename = str(tmp_path / "data.pkl")
    data.save(filename)
    assert Data.load(filename).linelist is None


def test_data_input_reset_and_pickle_round_trip(tmp_path, synthetic_spectrum):
    # Store a small exact spectrum so the pickle round trip can be compared exactly.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    data = Data()
    data.set_inputs(wavelengths, flux, errors, sn)
    data.linelist = linelist
    data.velocities = velocities
    data.wavelengths["combined"] = wavelengths

    # Saving must preserve input arrays and the validated line list.
    path = tmp_path / "data.pkl"
    data.save(str(path))
    loaded = Data.load(str(path))

    np.testing.assert_array_equal(loaded.wavelengths["input"][0], wavelengths)
    np.testing.assert_array_equal(loaded.linelist["depths"], linelist["depths"])
    loaded.reset()
    assert "input" in loaded.flux


def test_data_input_sorting_skips_and_selective_reset():
    # Inputs arrive in descending order and are sub-sampled only after sorting.
    data = Data()
    wavelengths = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    flux = np.array([1.0, 0.9, 0.8, 0.9, 1.0])
    errors = np.full(5, 0.01)
    data.set_inputs(wavelengths, flux, errors, input_sn=100.0, skips=2)
    data.config.profile_groups = np.array([0, 1])
    data.profile_groups = np.array([0, 1])
    data.alpha["derived"] = np.ones((2, 2))

    np.testing.assert_array_equal(data.wavelengths["input"][0], [1.0, 3.0, 5.0])

    # Reset clears calculations while preserving combined data and config inputs.
    combined_before = data.wavelengths["combined"].copy()
    data.reset(preserve_combined=True)
    np.testing.assert_array_equal(data.wavelengths["combined"], combined_before)
    np.testing.assert_array_equal(data.config.profile_groups, [0, 1])
    assert data.profile_groups is None
    assert data.alpha == {}

    with pytest.raises(ValueError, match="more than 1 value"):
        Data().set_inputs([1.0], [1.0], [0.1], 10.0)

    data.reset(preserve_combined=False)
    assert "combined" not in data.wavelengths
    np.testing.assert_array_equal(data.config.profile_groups, [0, 1])


def test_data_set_inputs_reuses_complete_existing_inputs(synthetic_spectrum):
    # Once all inputs exist, a completely empty update keeps them and rebuilds combined state.
    wavelengths, flux, errors, sn, _, _ = synthetic_spectrum
    data = Data()
    data.set_inputs(wavelengths, flux, errors, sn)
    original = data.flux["input"].copy()
    data.set_inputs()

    np.testing.assert_array_equal(data.flux["input"], original)
    assert "combined" in data.flux

    # Partial replacement is also ignored because all three spectrum arrays are required together.
    data.set_inputs(input_flux=flux * 0.9)
    np.testing.assert_array_equal(data.flux["input"], original)


def test_data_estimates_errors_from_per_pixel_harps_sn_and_plots_lines(harps_order_40):
    """Exercise the legacy per-pixel S/N input route with a real extracted order."""
    wavelengths, flux, _, sn, velocities, linelist = harps_order_40
    # This follows the documented route where one S/N value is supplied per pixel.
    data = Data()
    data.set_inputs(wavelengths, flux, input_sn=np.full_like(flux, sn))
    data.linelist = linelist
    data.velocities = velocities

    assert data.errors["input"].shape == flux[None].shape


def test_data_linelist_plot_supports_indices_and_bounds(harps_order_40):
    # Give Data a real line list, then select by both explicit indices and wavelength bounds.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    data = Data()
    data.set_inputs(wavelengths, flux, errors, sn)
    data.linelist = linelist
    data.velocities = velocities
    line_wavelengths = data.linelist["wavelengths"]
    in_order = np.flatnonzero((line_wavelengths >= wavelengths.min()) &
                              (line_wavelengths <= wavelengths.max()))

    # Index and bound routes should display precisely the chosen line-list region.
    indexed_figure, indexed_axis = data.plot_linelist(idx=in_order[:3], return_fig=True)
    bounded_figure, bounded_axis = data.plot_linelist(
        bounds=(wavelengths.min(), wavelengths.max()), return_fig=True,
    )
    assert len(indexed_axis.collections[0].get_segments()) == 3
    assert len(bounded_axis.collections[0].get_segments()) == len(in_order)
    plt.close(indexed_figure)
    plt.close(bounded_figure)


def test_data_properties_and_result_view(harps_result):
    # Rebuild through the dictionary format used by Data.save and DataList packing.
    payload = harps_result.data.to_dict()
    rebuilt = Data().from_dict(payload)

    assert isinstance(rebuilt.config, Config)
    np.testing.assert_array_equal(rebuilt.velocities, harps_result.data.velocities)
    assert rebuilt.result.data is rebuilt
    assert "Number of velocity points" in repr(rebuilt)

    # The sampler setter accepts an emcee backend and can explicitly discard it again.
    rebuilt.sampler = harps_result.sampler.backend
    np.testing.assert_array_equal(rebuilt.sampler.get_chain(), harps_result.sampler.get_chain())
    rebuilt.sampler = None
    assert rebuilt.sampler is None

    # Incomplete and failed data do not expose misleading Result objects.
    incomplete = Data()
    assert incomplete.result is None
    incomplete.exception = RuntimeError("failed")
    assert incomplete.result is None


def test_data_residual_masking_plot_uses_stored_acid_intermediates(harps_result):
    # The completed shared result already holds every residual-mask diagnostic array.
    before = set(plt.get_fignums())
    harps_result.data.plot_residual_masking()
    created = set(plt.get_fignums()) - before

    # Residuals, masked profile, and forward model are drawn as three separate figures.
    assert len(created) == 3
    for figure_number in created:
        plt.close(figure_number)


def test_data_plot_methods_validate_missing_intermediate_state():
    # Plotting before its corresponding processing stage should name the missing prerequisite.
    data = Data()
    with pytest.raises(ACIDStateError, match="No linelist"):
        data.plot_linelist(return_fig=True)
    with pytest.raises(ValueError, match="key"):
        data.plot_continuum_fit("unknown", return_fig=True)
    with pytest.raises(ACIDStateError, match="Residual masking"):
        data.plot_residual_masking()


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


def test_profile_groups_are_revalidated_after_config_changes(synthetic_spectrum):
    *_, linelist = synthetic_spectrum
    data = Data()
    data.linelist = linelist
    data.config.profile_groups = np.array([0, 1])

    np.testing.assert_array_equal(data.linelist["wavelengths"], linelist["wavelengths"])

    data.config.profile_groups = np.array([0])
    with pytest.raises(ValueError, match="same length"):
        _ = data.linelist


def test_datalist_indexes_orders_and_persists_inputs(tmp_path, synthetic_spectrum):
    # Use non-consecutive order labels to test the instrument-order mapping explicitly.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    configs = [Config(poly_ord=2), Config(poly_ord=4)]
    save_dir = tmp_path / "results"
    datalist = DataList(np.array([wavelengths, wavelengths]), np.array([flux, flux]),
                        np.array([errors, errors]), np.array([sn, sn]), velocities, linelist,
                        order_range=[10, 12], config=configs, save_dir=str(save_dir))

    # Initialisation saves a lightweight Data object for every requested order.
    assert len(datalist) == 2
    assert datalist[12].config.order == 12
    assert [datalist[order].config.poly_ord for order in [10, 12]] == [2, 4]
    assert (save_dir / "order_10" / "data.pkl").exists()
    with pytest.raises(KeyError):
        _ = datalist[11]


def test_datalist_chi_squared_plot_uses_completed_orders():
    # Build completed Data objects directly: plotting needs results, not another MCMC run.
    velocities = np.array([-1.0, 0.0, 1.0])
    data_list = []
    for order, scale in enumerate((1.0, 2.0), start=10):
        data = Data()
        data.config = Config(order=order, order_range=[10, 11])
        data.velocities = velocities
        data.profile["final"] = (np.ones(3), np.full(3, 0.01), np.eye(3) * 1e-4)
        data.flux["final"] = np.array([1.0, 0.9, 1.1])
        data.forward_y["final"] = data.flux["final"] - 0.01 * scale
        data.errors["final"] = np.full(3, 0.01)
        data_list.append(data)

    datalist = DataList.from_datalist(data_list)
    fig, ax = datalist.plot_chi2(return_fig=True)

    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [10, 11])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [3.0, 12.0])
    plt.close(fig)


@pytest.fixture
def completed_datalist():
    """A three-order DataList with deterministic, already-completed profiles."""
    velocities = np.linspace(-5, 5, 11)
    data_list = []
    for order, depth in zip([20, 21, 22], [0.02, 0.04, 0.06]):
        # Construct the minimum final state consumed by DataList combination and plots.
        data = Data()
        data.config = Config(order=order, order_range=[20, 21, 22])
        data.velocities = velocities
        profile = 1 - depth * np.exp(-velocities ** 2 / 4)
        errors = np.full_like(profile, 0.01)
        data.profile["final"] = (profile, errors, np.diag(errors ** 2))
        data.profiles = [(profile, errors, np.diag(errors ** 2))]
        data.complete = True
        data.flux["final"] = np.array([1.0, 0.98, 1.01])
        data.forward_y["final"] = np.array([1.0, 0.99, 1.00])
        data.errors["final"] = np.full(3, 0.01)
        data_list.append(data)
    return DataList.from_datalist(data_list)


def test_datalist_order_mapping_append_and_range_management(completed_datalist):
    # The class sorts by physical order and supports indexing through that order label.
    datalist = completed_datalist
    assert datalist.orders.tolist() == [20, 21, 22]
    assert datalist.i2o == {0: 20, 1: 21, 2: 22}

    # Duplicates require an explicit overwrite rather than silently replacing data.
    duplicate = Data()
    duplicate.config = Config(order=21, order_range=[20, 21, 22])
    duplicate.velocities = datalist.velocities
    with pytest.raises(ValueError, match="already exists"):
        datalist.append(duplicate)
    datalist.append(duplicate, overwrite=True)
    assert datalist[21] is duplicate

    # Extending the range allows a newly observed order to be added safely.
    new_order = Data()
    new_order.config = Config(order=23, order_range=[20, 21, 22])
    new_order.velocities = datalist.velocities
    datalist.append(new_order, extend=True)
    assert datalist.orders.tolist() == [20, 21, 22, 23]
    assert datalist.order_range.tolist() == [20, 21, 22, 23]

    # Shrinking away an existing order must be rejected to avoid losing its mapping.
    with pytest.raises(ValueError, match="subset"):
        datalist.set_order_range(np.array([20, 21]))


def test_datalist_callable_and_container_protocol(monkeypatch, completed_datalist):
    # Iteration, length, string output, and order indexing form the basic container API.
    datalist = completed_datalist
    assert len(list(datalist)) == len(datalist) == 3
    assert datalist[20].config.order == 20
    assert "20" in str(datalist)

    # Calling a DataList is documented as a direct forwarding route to run_ACID.
    received = {}

    def fake_run(self, *args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return "forwarded"

    monkeypatch.setattr(DataList, "run_ACID", fake_run)
    assert datalist([20], overwrite=True) == "forwarded"
    assert received == {"args": ([20],), "kwargs": {"overwrite": True}}


def test_datalist_setter_and_from_datalist_validate_members_and_velocities(completed_datalist):
    # The property accepts only a list containing Data objects.
    with pytest.raises(ValueError, match="must be a list"):
        completed_datalist.data_list = completed_datalist[20]
    with pytest.raises(ValueError, match="instances of the Data"):
        completed_datalist.data_list = [object()]

    # Every order must have a unique label and share one velocity grid.
    duplicate = Data().from_dict(completed_datalist[20].to_dict())
    with pytest.raises(ValueError, match="unique"):
        DataList.from_datalist([completed_datalist[20], duplicate])
    changed_velocity = Data().from_dict(completed_datalist[21].to_dict())
    changed_velocity.velocities = changed_velocity.velocities + 0.1
    with pytest.raises(ValueError, match="same velocity grid"):
        DataList.from_datalist([completed_datalist[20], changed_velocity])


def test_datalist_combines_profiles_and_exposes_all_diagnostics(completed_datalist):
    # Combine the fabricated final profiles in the same optical-depth space as production.
    datalist = completed_datalist
    datalist.combine_profiles(exclude=22)
    profile, errors, covariance = datalist.combined_profile

    assert profile.shape == datalist.velocities.shape
    assert errors.shape == profile.shape
    assert covariance.shape == (len(profile), len(profile))
    assert datalist.excluded_orders == [22]

    # Each diagnostic should work from stored results without invoking ACID again.
    figures = [datalist.plot_combined_profile(return_fig=True)[0],
               datalist.plot_all_profiles(return_fig=True)[0],
               datalist.plot_mean_profile_errors(return_fig=True)[0],
               datalist.plot_chi2(return_fig=True)[0],
               datalist.fit_profile(return_fig=True)[0]]
    for figure in figures:
        assert figure.axes
        plt.close(figure)


def test_datalist_combination_validation_and_lazy_property(completed_datalist):
    # Accessing combined_profile should calculate it once when no stored combination exists.
    datalist = completed_datalist
    assert datalist._combined_profile is None
    profile = datalist.combined_profile
    assert datalist._combined_profile is profile

    # Exclusions must refer to real instrument orders, and all orders cannot be removed.
    with pytest.raises(ValueError, match="available orders"):
        datalist.combine_profiles(exclude=[99])
    with pytest.raises(ValueError):
        datalist.combine_profiles(exclude=datalist.orders)


def test_datalist_save_load_and_input_validation(tmp_path, completed_datalist):
    import pickle

    # Saving a packed DataList should permit loading from either its directory or pickle file.
    datalist = completed_datalist
    for data in datalist:
        data.linelist = [[5000.0, 5002.0], [0.1, 0.2]]
    datalist.save(str(tmp_path))
    with (tmp_path / "datalist.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    for saved_data in payload["data_list"]:
        assert type(saved_data["_linelist"]) is dict
    from_directory = DataList.load(str(tmp_path))
    from_file = DataList.load(str(tmp_path / "datalist.pkl"))

    assert from_directory.orders.tolist() == [20, 21, 22]
    assert from_file.orders.tolist() == [20, 21, 22]
    for original, loaded in zip(datalist, from_file):
        np.testing.assert_array_equal(loaded.linelist[0], original.linelist[0])
        np.testing.assert_array_equal(loaded.linelist[1], original.linelist[1])

    # The Results property mirrors every order and caches the constructed Result objects.
    assert len(datalist.results) == len(datalist)
    assert datalist.results is datalist.results

    # Invalid selection and worker arguments are rejected before any ACID call is made.
    with pytest.raises(ValueError, match="available orders"):
        datalist.run_ACID(orders=[99])
    with pytest.raises(ValueError, match="Both worker"):
        datalist.run_ACID(worker=0)

    # Missing paths and invalid load targets should fail without touching any order data.
    with pytest.raises(ValueError, match="No save directory"):
        DataList.from_datalist(datalist.data_list).save()
    with pytest.raises(ValueError, match="not a directory"):
        DataList.load(str(tmp_path / "missing"))


def test_datalist_path_relocation_updates_each_data_file(tmp_path, completed_datalist):
    # Relocation still updates the data path when there is no sampler.
    data = completed_datalist[20]
    changed = DataList._update_paths_for_data(data, str(tmp_path))
    expected_directory = tmp_path / "order_20"

    assert changed is True
    assert data.config.save_path == str(expected_directory / "data.pkl")
    assert data.config.dir == str(expected_directory)
    assert data.config.sampler_path == str(expected_directory / "sampler.h5")
    assert data.sampler is None
    assert not (expected_directory / "sampler.h5").exists()
    assert (expected_directory / "data.pkl").exists()

    # Reapplying the same root is a no-op and should not rewrite the file.
    assert DataList._update_paths_for_data(data, str(tmp_path)) is False


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("stored_sampler", [False, True])
def test_datalist_load_without_sampler_does_not_rewrite(tmp_path, capsys, packed, stored_sampler):
    data = Data()
    # A missing dir now intentionally requires migration. Start with an established
    # root so this tests only the absence of optional sampler/figure files.
    data.config = Config(order=50, verbose=2, dir=str(tmp_path / "order_50"))
    if stored_sampler:
        data.config.sampler_path = str(tmp_path / "missing.h5")
    filename = tmp_path / "order_50" / "data.pkl"
    data.save(str(filename))
    if packed:
        DataList.from_datalist([data], save_dir=str(tmp_path)).save()
    (tmp_path / "order_50" / "figures").rmdir()
    paths = [filename] + ([tmp_path / "datalist.pkl"] if packed else [])
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    capsys.readouterr()

    loaded = DataList.load(str(tmp_path))

    assert loaded[50].sampler is None
    assert loaded[50].config.sampler_path == data.config.sampler_path
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before
    output = capsys.readouterr().out
    assert "Data object saved" not in output
    assert "current location" not in output
    assert not (tmp_path / "order_50" / "figures").exists()


@pytest.mark.parametrize("packed", [False, True])
def test_datalist_missing_dir_is_migrated_once(tmp_path, packed):
    data = Data()
    data.config = Config(order=50, verbose=0)
    filename = tmp_path / "order_50" / "data.pkl"
    data.save(str(filename))
    if packed:
        DataList.from_datalist([data], save_dir=str(tmp_path), verbose=0).save()

    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded[50].config.dir == str(filename.parent)
    assert Data.load(str(filename)).config.dir == str(filename.parent)
    paths = [filename] + ([tmp_path / "datalist.pkl"] if packed else [])
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    DataList.load(str(tmp_path), verbose=0)
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


@pytest.mark.parametrize("packed", [False, True])
def test_datalist_persists_inferred_order_with_matching_paths(tmp_path, packed):
    import pickle

    data = Data()
    data.config = Config(dir=str(tmp_path / "order_0"), verbose=0)
    data.save()
    if packed:
        with open(tmp_path / "datalist.pkl", "wb") as stream:
            pickle.dump({"data_list": [data.to_dict()]}, stream)
    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded.orders.tolist() == [0]
    # Inspect the saved payload: Data.load would infer again and hide a missed save.
    with open(tmp_path / "order_0" / "data.pkl", "rb") as stream:
        assert pickle.load(stream)["config"]["order"] == 0


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("keep_original", [False, True])
def test_datalist_relocation_rebinds_sampler(tmp_path, harps_result, packed, keep_original):
    import shutil
    from emcee.backends import HDFBackend

    original = tmp_path / "original"
    original.mkdir()
    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(order=0, dir=str(original / "order_0"), verbose=0)
    backend = HDFBackend(data.config.sampler_path)
    backend.reset(4, 1)
    data.sampler = backend
    # Persist explicit overrides as well as dir; all must relocate together.
    data.config.figure_dir = str(original / "order_0" / "figures")
    data.save(data.config.save_path)
    if packed:
        DataList.from_datalist([data], save_dir=str(original), verbose=0).save()
    relocated = tmp_path / "relocated"
    if keep_original:
        shutil.copytree(original, relocated)
    else:
        original.rename(relocated)

    loaded = DataList.load(str(relocated), verbose=0)
    order_dir = relocated / "order_0"
    assert loaded[0].config.dir == str(order_dir)
    assert loaded[0].config.save_path == str(order_dir / "data.pkl")
    assert loaded[0].config.figure_dir == str(order_dir / "figures")
    assert loaded[0].config.sampler_path == str(order_dir / "sampler.h5")
    assert loaded[0].sampler.backend.filename == str(order_dir / "sampler.h5")
    assert loaded[0].sampler.backend.shape == (4, 1)
    assert original.exists() == keep_original
    paths = [order_dir / "data.pkl", order_dir / "sampler.h5"]
    if packed:
        paths.append(relocated / "datalist.pkl")
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    DataList.load(str(relocated), verbose=0)
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


@pytest.mark.parametrize("path_kind", ["save_path", "sampler_path", "figure_dir"])
def test_datalist_repairs_explicit_path_with_matching_dir(tmp_path, harps_result, path_kind):
    from emcee.backends import HDFBackend

    order_dir = tmp_path / "order_20"
    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(order=20, dir=str(order_dir), verbose=0)
    if path_kind == "sampler_path":
        HDFBackend(str(order_dir / "sampler.h5")).reset(4, 1)
        data.config.sampler_path = str(tmp_path / "missing.h5")
    elif path_kind == "figure_dir":
        data.config.figure_dir = str(tmp_path / "old_figures")
        (tmp_path / "old_figures").rmdir()
    else:
        data.config.save_path = str(tmp_path / "old.pkl")
    assert DataList._update_paths_for_data(data, str(tmp_path)) is True
    assert data.config.save_path == str(order_dir / "data.pkl")
    assert data.config.sampler_path == str(order_dir / "sampler.h5")
    assert data.config.figure_dir == str(order_dir / "figures")
    assert DataList._update_paths_for_data(data, str(tmp_path)) is False


@pytest.mark.parametrize("relocate", [False, True])
@pytest.mark.parametrize("newer_source", ["order", "packed"])
def test_datalist_load_uses_newer_saved_results(tmp_path, relocate, newer_source):
    import os

    root = tmp_path / "original"
    root.mkdir()
    data = Data()
    data.config = Config(order=20, dir=str(root / "order_20"), verbose=0)
    data.save()
    DataList.from_datalist([data], save_dir=str(root), verbose=0).save()
    data.complete = True
    data.save()
    order_file = root / "order_20" / "data.pkl"
    packed_file = root / "datalist.pkl"
    # Explicit times avoid depending on filesystem timestamp resolution or sleeps.
    older = 1_700_000_000_000_000_000
    newer = older + 2_000_000_000
    order_time, packed_time = (newer, older) if newer_source == "order" else (older, newer)
    os.utime(order_file, ns=(order_time, order_time))
    os.utime(packed_file, ns=(packed_time, packed_time))
    if relocate:
        root = root.rename(tmp_path / "relocated")

    loaded = DataList.load(str(root), verbose=0)
    expected_complete = newer_source == "order"
    assert loaded[20].complete is expected_complete
    # Loading a moved snapshot must never roll a newer per-order result back.
    assert Data.load(str(root / "order_20" / "data.pkl")).complete is (
        expected_complete if relocate else True
    )
    assert DataList.load(str(root), verbose=0)[20].complete is expected_complete


@pytest.mark.parametrize("relocate", [False, True])
def test_packed_datalist_includes_new_orders(tmp_path, relocate):
    root = tmp_path / "original"
    root.mkdir()
    first = Data()
    first.config = Config(order=20, dir=str(root / "order_20"), verbose=0)
    first.save()
    DataList.from_datalist([first], save_dir=str(root), verbose=0).save()
    added = Data()
    added.config = Config(order=21, dir=str(root / "order_21"), verbose=0)
    added.complete = True
    added.save()
    if relocate:
        root = root.rename(tmp_path / "relocated")
    for _ in range(2):
        loaded = DataList.load(str(root), verbose=0)
        assert loaded.orders.tolist() == [20, 21]
        assert loaded[21].complete


def test_exported_sampler_keeps_saving_new_steps(tmp_path, harps_result):
    import emcee

    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(verbose=0)
    data.sampler = emcee.EnsembleSampler(4, 1, lambda x: -0.5 * np.sum(x ** 2))
    data.sampler.run_mcmc(np.array([[-1.], [-0.3], [0.3], [1.]]), 3)
    filename = str(tmp_path / "data.pkl")
    data.save(filename, str(tmp_path / "sampler.h5"))
    data.sampler.run_mcmc(None, 2)
    data.save()
    loaded = Data.load(filename)
    assert loaded.sampler.backend.iteration == 5
    np.testing.assert_array_equal(loaded.sampler.get_chain(), data.sampler.get_chain())


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_save_preserves_checkpoint(tmp_path, monkeypatch, packed, failure):
    import os
    import pickle

    data = Data()
    data.config = Config(order=20, verbose=0)
    filename = tmp_path / ("datalist.pkl" if packed else "data.pkl")
    save = (DataList.from_datalist([data], save_dir=str(tmp_path), verbose=0).save
            if packed else lambda: data.save(str(filename)))
    save()
    before = filename.read_bytes()
    data.complete = True

    def fail(*args, **kwargs):
        if failure == "write":
            args[1].write(b"partial pickle")
        raise OSError("simulated save failure")

    with monkeypatch.context() as patch:
        patch.setattr(pickle if failure == "write" else os,
                      "dump" if failure == "write" else "replace", fail)
        with pytest.raises(OSError, match="simulated save failure"):
            save()
    assert filename.read_bytes() == before
    assert list(tmp_path.iterdir()) == [filename]
    save()
    payload = pickle.loads(filename.read_bytes())
    assert (payload["data_list"][0] if packed else payload)["complete"]


@pytest.mark.parametrize("packed", [False, True])
def test_datalist_copy_without_sampler_detaches_original(tmp_path, harps_result, packed):
    import shutil
    from emcee.backends import HDFBackend

    original = tmp_path / "original"
    original.mkdir()
    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(order=20, dir=str(original / "order_20"), verbose=0)
    backend = HDFBackend(data.config.sampler_path)
    backend.reset(4, 1)
    data.sampler = backend
    data.save()
    if packed:
        DataList.from_datalist([data], save_dir=str(original), verbose=0).save()
    original_sampler = Path(data.config.sampler_path)
    before = (original_sampler.read_bytes(), original_sampler.stat().st_mtime_ns)
    copied = tmp_path / "copied"
    shutil.copytree(original, copied, ignore=shutil.ignore_patterns("sampler.h5"))

    loaded = DataList.load(str(copied), verbose=0)[20]
    assert loaded.sampler is None
    assert loaded.config.sampler_path == str(copied / "order_20" / "sampler.h5")
    assert loaded.complete
    assert DataList.load(str(copied), verbose=0)[20].sampler is None
    assert (original_sampler.read_bytes(), original_sampler.stat().st_mtime_ns) == before
    assert not (copied / "order_20" / "sampler.h5").exists()


@pytest.mark.parametrize("store_sampler", [False, True])
@pytest.mark.parametrize("use_save_dir", [False, True])
@pytest.mark.parametrize("config_source", ["config", "kwargs"])
def test_datalist_save_dir_overrides_paths_and_controls_sampler(
    tmp_path, synthetic_spectrum, store_sampler, use_save_dir, config_source,
):
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    settings = dict(dir=str(custom_dir), save_path=str(custom_dir / "custom.pkl"),
                    sampler_path=str(custom_dir / "custom.h5"),
                    figure_dir=str(custom_dir / "plots"))
    config_args = {"config": Config(**settings)} if config_source == "config" else settings
    save_dir = tmp_path / "results"
    datalist = DataList(wavelengths[None], flux[None], errors[None], [sn],
                        velocities, linelist, order_range=[20], verbose=0,
                        save_dir=str(save_dir) if use_save_dir else None,
                        store_sampler=store_sampler, **config_args)
    cfg = datalist[20].config
    if use_save_dir:
        order_dir = save_dir / "order_20"
        assert cfg.dir == str(order_dir)
        assert cfg.save_path == str(order_dir / "data.pkl")
        assert cfg.figure_dir == str(order_dir / "figures")
        expected_sampler = str(order_dir / "sampler.h5") if store_sampler else None
        assert cfg.sampler_path == expected_sampler
        loaded = DataList.load(str(save_dir), verbose=0)
        assert loaded[20].config.sampler_path == expected_sampler
        assert not (custom_dir / "custom.pkl").exists()
    else:
        for key, value in settings.items():
            assert getattr(cfg, key) == value
    # Wrapping existing instances must not reconfigure their output settings.
    before = cfg.to_dict()
    DataList.from_datalist([datalist[20]], save_dir=str(tmp_path), verbose=0)
    assert cfg.to_dict() == before


def test_datalist_constructor_relocates_reused_data(tmp_path, synthetic_spectrum):
    import shutil

    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    inputs = dict(wavelengths=wavelengths[None], flux=flux[None], errors=errors[None],
                  sn=[sn], velocities=velocities, linelist=linelist,
                  order_range=[20], config=Config(verbose=0), verbose=0)
    original, copied = tmp_path / "original", tmp_path / "copied"
    DataList(**inputs, save_dir=str(original))
    original_file = original / "order_20" / "data.pkl"
    before = original_file.read_bytes()
    shutil.copytree(original, copied)

    reused = DataList(**inputs, save_dir=str(copied), overwrite=False)[20]
    reused.config.poly_ord = 7
    reused.save()
    assert original_file.read_bytes() == before
    assert Data.load(str(copied / "order_20" / "data.pkl")).config.poly_ord == 7
    assert reused.config.dir == str(copied / "order_20")
    assert reused.config.save_path == str(copied / "order_20" / "data.pkl")
    assert reused.config.sampler_path == str(copied / "order_20" / "sampler.h5")
    assert reused.config.figure_dir == str(copied / "order_20" / "figures")


def test_data_combines_multiple_frames_on_the_highest_sn_grid(harps_order_40):
    # Two frames with slightly different wavelength coverage exercise interpolation and weighting.
    wavelengths, flux, errors, sn, _, _ = harps_order_40
    data = Data()
    shifted_wavelengths = wavelengths + 0.001
    data.set_inputs(np.array([wavelengths, shifted_wavelengths]), np.array([flux, flux]),
                    np.array([errors, errors * 2]), np.array([sn, sn / 2]))

    # The highest-S/N frame defines the combined wavelength grid and gains the greatest weight.
    combined = data.combine_spec()
    np.testing.assert_allclose(combined[0], wavelengths)
    assert np.median(combined[2]) < np.median(errors)
    assert combined[2][0] == pytest.approx(errors[0])
    assert data.sn["input"].ndim + 1 == data.wavelengths["input"].ndim


def test_array_helpers_and_optical_depth_round_trip():
    waves = np.array([1.0, np.nan, 3.0])
    flux = np.array([1.0, -1.0, 2.0])
    errors = np.array([0.1, 0.1, np.inf])
    _, _, _, mask = utils.mask_invalid(waves, flux, errors, return_mask=True)
    assert mask.tolist() == [True, False, False]
    assert utils.drop_invalid(waves, flux, errors)[0].tolist() == [1.0]

    original_flux = np.array([0.8, 0.9])
    original_errors = np.array([0.02, 0.03])
    original_lines = np.array([0.2, 0.1])
    od = utils.flux_to_od(original_flux, original_errors, original_lines)
    restored = utils.od_to_flux(*od)
    for actual, expected in zip(restored, (original_flux, original_errors, original_lines)):
        np.testing.assert_allclose(actual, expected)


def test_numerical_utilities_and_validation():
    assert utils.calc_deltav(np.array([5002.0, 5000.0, 5001.0])) > 0
    assert utils.guess_SNR(np.arange(1, 11), np.ones(10), np.full(10, 0.1)) == pytest.approx(10)
    np.testing.assert_allclose(utils.guess_errors(np.ones((2, 3)), [10, 20]), [[0.1] * 3, [0.05] * 3])
    assert utils.next_pow_2(7) == 8
    with pytest.raises(ValueError):
        utils.calc_deltav(np.array([-1.0, 1.0]))
    with pytest.raises(ValueError):
        utils.convert_moves_to_emcee([("NoMove", 1.0)])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--tb=auto"]))
