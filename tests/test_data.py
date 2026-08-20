#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import Config, Data, DataList, LineList, MaskingLines
from ACID_code import utils


def test_config_priorities_properties_and_environment(monkeypatch):
    # Low-priority values should fill gaps, while high-priority values win conflicts.
    config = Config(verbose="off", order=1)
    config.update_lowpri(order=2, poly_ord=4)
    config.update_hipri(order=3)

    assert config.verbose == 0
    assert config.order == 3
    assert config.poly_ord == 4
    # Invalid configuration and misplaced Data attributes should fail clearly.
    with pytest.raises(KeyError):
        config.update_hipri(not_a_setting=True)
    with pytest.raises(AttributeError):
        config.linelist = []

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
    assert "order: 7" in repr(config)

    # Defaults are printable for interactive inspection, and invalid verbose values fail early.
    Config.print_defaults()
    assert "poly_ord" in capsys.readouterr().out
    with pytest.raises(ValueError, match="between 0 and 4"):
        config.verbose = 5
    with pytest.raises(ValueError, match="not recognised"):
        config.verbose = "loud"


def test_masking_lines_accepts_compact_inputs_and_masks_grid():
    # A compact dictionary should expand its default width for each line.
    lines = MaskingLines({"telluric": {"default_width": 100, "lines": [5000.0]}})
    grid = np.array([4990.0, 5000.0, 5010.0])

    # Both the combined and named-mask interfaces should describe the same region.
    assert lines.get_1d_mask_on_grid(grid).tolist() == [False, True, False]
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
    wavelengths, depths = LineList.validate_linelist(linelist)

    np.testing.assert_array_equal(wavelengths, [5000.0, 5002.0])
    np.testing.assert_array_equal(depths, [0.1, 0.2])
    with pytest.raises(ValueError, match="same length"):
        LineList.validate_linelist([[1, 2], [0.1]])


@pytest.mark.parametrize("linelist", [
    [[5002.0, 5000.0], [0.2, 0.1]],
    {"wavelengths": [5002.0, 5000.0], "depths": [0.2, 0.1]},
    LineList({"wavelengths": np.array([5002.0, 5000.0]),
              "depths": np.array([0.2, 0.1])}),
])
def test_linelist_accepts_each_documented_in_memory_format(linelist):
    # All public in-memory forms should produce the same sorted pair of arrays.
    wavelengths, depths = LineList.validate_linelist(linelist)

    np.testing.assert_array_equal(wavelengths, [5000.0, 5002.0])
    np.testing.assert_array_equal(depths, [0.1, 0.2])


def test_linelist_file_indexing_and_invalid_line_removal(linelist_path):
    # A file-backed line list should be readable through the same validation route.
    wavelengths, depths = LineList.validate_linelist(str(linelist_path))
    line_list = LineList({"wavelengths": wavelengths, "depths": depths})

    assert line_list[0].shape == line_list[1].shape
    assert line_list["wavelengths"].ndim == 1
    with pytest.raises(IndexError):
        _ = line_list[2]

    # Invalid depths and wavelengths are removed together, retaining the validity mask.
    kept_wavelengths, kept_depths, mask = LineList.drop_invalid_lines(
        np.array([5000.0, np.nan, 5002.0, 5003.0]),
        np.array([0.1, 0.2, -0.1, 1.0]), return_mask=True, verbose=0,
    )
    np.testing.assert_array_equal(mask, [True, False, False, False])
    np.testing.assert_array_equal(kept_wavelengths, [5000.0])
    np.testing.assert_array_equal(kept_depths, [0.1])


def test_data_input_reset_and_pickle_round_trip(tmp_path, synthetic_spectrum):
    # Store a small exact spectrum so the pickle round trip can be compared exactly.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    data = Data()
    data.config = Config(verbose=0)
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
    data.config = Config(verbose=0)
    wavelengths = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    flux = np.array([1.0, 0.9, 0.8, 0.9, 1.0])
    errors = np.full(5, 0.01)
    data.set_inputs(wavelengths, flux, errors, input_sn=100.0, skips=2)
    data.input_profile_groups = np.array([0, 1])
    data.alpha["derived"] = np.ones((2, 2))

    np.testing.assert_array_equal(data.wavelengths["input"][0], [1.0, 3.0, 5.0])

    # Reset clears calculations while optionally preserving combined data and manual groups.
    combined_before = data.wavelengths["combined"].copy()
    data.reset(preserve_combined=True, preserve_input_profile_groups=True)
    np.testing.assert_array_equal(data.wavelengths["combined"], combined_before)
    np.testing.assert_array_equal(data.input_profile_groups, [0, 1])
    assert data.alpha == {}

    data.reset(preserve_combined=False, preserve_input_profile_groups=False)
    assert "combined" not in data.wavelengths
    assert data.input_profile_groups is None


def test_data_set_inputs_reuses_complete_existing_inputs(synthetic_spectrum):
    # Once all inputs exist, a completely empty update keeps them and rebuilds combined state.
    wavelengths, flux, errors, sn, _, _ = synthetic_spectrum
    data = Data()
    data.config = Config(verbose=0)
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
    data.config = Config(verbose=0)
    data.set_inputs(wavelengths, flux, input_sn=np.full_like(flux, sn))
    data.linelist = linelist
    data.velocities = velocities

    assert data.errors["input"].shape == flux[None].shape


def test_data_linelist_plot_supports_indices_and_bounds(harps_order_40):
    # Give Data a real line list, then select by both explicit indices and wavelength bounds.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    data = Data()
    data.config = Config(verbose=0)
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
    incomplete.config.verbose = 0
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
    data.config.verbose = 0
    with pytest.raises(ValueError, match="No linelist"):
        data.plot_linelist(return_fig=True)
    with pytest.raises(ValueError, match="plot_type"):
        data.plot_continuum_fit("unknown", return_fig=True)
    with pytest.raises(ValueError, match="Residual masking"):
        data.plot_residual_masking()


def test_data_velocity_and_linelist_overwrites_clear_derived_state(synthetic_spectrum):
    # Populate derived state, then change each dependency and check it is invalidated.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    data = Data()
    data.config = Config(verbose=0)
    data.set_inputs(wavelengths, flux, errors, sn)
    data.velocities = velocities
    data.linelist = linelist
    data.alpha["derived"] = np.ones((2, 2))
    data.velocities = velocities + 1
    assert data.alpha == {}

    data.alpha["derived"] = np.ones((2, 2))
    data.input_profile_groups = np.array([0, 1])
    changed_linelist = {"wavelengths": linelist["wavelengths"],
                        "depths": linelist["depths"] * 0.9}
    data.linelist = changed_linelist
    assert data.alpha == {}
    assert data.input_profile_groups is None

    with pytest.raises(ValueError, match="finite"):
        data.velocities = np.array([0.0, np.nan])


def test_datalist_indexes_orders_and_persists_inputs(tmp_path, synthetic_spectrum):
    # Use non-consecutive order labels to test the instrument-order mapping explicitly.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    datalist = DataList(np.array([wavelengths, wavelengths]), np.array([flux, flux]),
                        np.array([errors, errors]), np.array([sn, sn]), velocities, linelist,
                        order_range=[10, 12], save_dir=str(tmp_path), verbose=0)

    # Initialisation saves a lightweight Data object for every requested order.
    assert len(datalist) == 2
    assert datalist[12].config.order == 12
    assert (tmp_path / "order_10" / "data.pkl").exists()
    with pytest.raises(KeyError):
        _ = datalist[11]


def test_datalist_chi_squared_plot_uses_completed_orders():
    # Build completed Data objects directly: plotting needs results, not another MCMC run.
    velocities = np.array([-1.0, 0.0, 1.0])
    data_list = []
    for order, scale in enumerate((1.0, 2.0), start=10):
        data = Data()
        data.config = Config(order=order, order_range=[10, 11], verbose=0)
        data.velocities = velocities
        data.profile["final"] = (np.ones(3), np.full(3, 0.01), np.eye(3) * 1e-4)
        data.flux["final"] = np.array([1.0, 0.9, 1.1])
        data.forward_y["final"] = data.flux["final"] - 0.01 * scale
        data.errors["final"] = np.full(3, 0.01)
        data_list.append(data)

    datalist = DataList.from_datalist(data_list, verbose=0)
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
        data.config = Config(order=order, order_range=[20, 21, 22], verbose=0)
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
    return DataList.from_datalist(data_list, verbose=0)


def test_datalist_order_mapping_append_and_range_management(completed_datalist):
    # The class sorts by physical order and supports indexing through that order label.
    datalist = completed_datalist
    assert datalist.orders.tolist() == [20, 21, 22]
    assert datalist.i2o == {0: 20, 1: 21, 2: 22}

    # Duplicates require an explicit overwrite rather than silently replacing data.
    duplicate = Data()
    duplicate.config = Config(order=21, order_range=[20, 21, 22], verbose=0)
    duplicate.velocities = datalist.velocities
    with pytest.raises(ValueError, match="already exists"):
        datalist.append(duplicate)
    datalist.append(duplicate, overwrite=True)
    assert datalist[21] is duplicate

    # Extending the range allows a newly observed order to be added safely.
    new_order = Data()
    new_order.config = Config(order=23, order_range=[20, 21, 22], verbose=0)
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
        DataList.from_datalist([completed_datalist[20], duplicate], verbose=0)
    changed_velocity = Data().from_dict(completed_datalist[21].to_dict())
    changed_velocity.velocities = changed_velocity.velocities + 0.1
    with pytest.raises(ValueError, match="same velocity grid"):
        DataList.from_datalist([completed_datalist[20], changed_velocity], verbose=0)


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
    # Saving a packed DataList should permit loading from either its directory or pickle file.
    datalist = completed_datalist
    datalist.save(str(tmp_path))
    from_directory = DataList.load(str(tmp_path), verbose=0)
    from_file = DataList.load(str(tmp_path / "datalist.pkl"), verbose=0)

    assert from_directory.orders.tolist() == [20, 21, 22]
    assert from_file.orders.tolist() == [20, 21, 22]

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
        DataList.from_datalist(datalist.data_list, verbose=0).save()
    with pytest.raises(ValueError, match="not a directory"):
        DataList.load(str(tmp_path / "missing"), verbose=0)


def test_datalist_path_relocation_updates_each_data_file(tmp_path, completed_datalist):
    # Relocation rewrites order-specific save and sampler paths, then persists the Data object.
    data = completed_datalist[20]
    changed = DataList._set_paths_for_data(data, str(tmp_path))
    expected_directory = tmp_path / "order_20"

    assert changed is True
    assert data.config.save_path == str(expected_directory / "data.pkl")
    assert data.config.sampler_path == str(expected_directory / "sampler.h5")
    assert (expected_directory / "data.pkl").exists()

    # Reapplying the same root is a no-op and should not rewrite the file.
    assert DataList._set_paths_for_data(data, str(tmp_path)) is False


def test_data_combines_multiple_frames_on_the_highest_sn_grid(harps_order_40):
    # Two frames with slightly different wavelength coverage exercise interpolation and weighting.
    wavelengths, flux, errors, sn, _, _ = harps_order_40
    data = Data()
    data.config = Config(verbose=0)
    shifted_wavelengths = wavelengths + 0.001
    data.set_inputs(np.array([wavelengths, shifted_wavelengths]), np.array([flux, flux]),
                    np.array([errors, errors * 2]), np.array([sn, sn / 2]))

    # The highest-S/N frame defines the combined wavelength grid and gains the greatest weight.
    combined = data.combine_spec()
    np.testing.assert_allclose(combined[0], wavelengths)
    assert np.median(combined[2]) < np.median(errors)
    assert combined[2][0] == pytest.approx(errors[0])


def test_array_helpers_and_optical_depth_round_trip():
    waves = np.array([1.0, np.nan, 3.0])
    flux = np.array([1.0, -1.0, 2.0])
    errors = np.array([0.1, 0.1, np.inf])
    _, _, _, mask = utils.mask_invalid(waves, flux, errors, return_mask=True, verbose=0)
    assert mask.tolist() == [True, False, False]
    assert utils.drop_invalid(waves, flux, errors, verbose=0)[0].tolist() == [1.0]

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
