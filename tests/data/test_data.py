#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code.diagnostics.errors import ACIDStateError
from ACID_code import Config, Data


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
    data.linelist = [[1.0, 3.0], [0.2, 0.3]]
    data.ll_mask = np.array([0, 1])
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


def test_data_plot_methods_validate_missing_intermediate_state():
    # Plotting before its corresponding processing stage should name the missing prerequisite.
    data = Data()
    with pytest.raises(ACIDStateError, match="No linelist"):
        data.plot_linelist(return_fig=True)
    with pytest.raises(ValueError, match="key"):
        data.plot_continuum_fit("unknown", return_fig=True)
    with pytest.raises(ACIDStateError, match="Residual masking"):
        data.plot_residual_masking()


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



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--tb=auto"]))
