"""Joint profile recovery with independent order continua."""
import numpy as np
import pytest
import shutil
import warnings
import matplotlib.pyplot as plt

from ACID_code import Config, Data, DataList, LSD, MCMC, Result
from ACID_code import utils
from ACID_code.diagnostics.errors import ACIDInputError, ACIDStateError
from ACID_code.diagnostics.warnings import ACIDDroppedDataWarning


def make_orders(nframes=1, **kwargs):
    velocities = np.array([-5., 0., 5.])
    lines = {"wavelengths": np.array([5003., 5007., 5103., 5107., 5203., 5207.]),
             "depths": np.tile([0.2, 0.3], 3)}
    orders, coefficients = [], []
    for i, npix in enumerate([501, 441, 381]):
        wave = np.linspace(5000 + 100 * i, 5010 + 100 * i, npix)
        coefs = np.array([1. + 0.1 * i, 0.05 * (i + 1)] + ([0.02] if i == 1 else []))
        continuum = utils.eval_continuum(utils.normalize_wavelengths(wave), coefs)
        alpha = LSD.calc_alpha(wave, lines["wavelengths"], -np.log1p(-lines["depths"]), velocities)
        flux = [continuum * np.exp(-alpha @ np.array([0.07, 0.12 + 0.02 * frame, 0.07]))
                for frame in range(nframes)]
        data = Data()
        data.config = Config(order=10 + i, poly_ord=len(coefs) - 1, verbose=0,
                             parallel=False, seed=2, nsteps=40, n_bins=10,
                             sigma_lower=100., sigma_upper=100., dev_perc=100.,
                             masking_lines={}, **kwargs)
        error = np.full(npix, 0.01 * (i + 1))
        if nframes == 1:
            data.set_inputs(wave, flux[0], error, 100.)
        else:
            data.set_inputs(np.tile(wave, (nframes, 1)), np.array(flux),
                            np.tile(error, (nframes, 1)), np.full(nframes, 100.))
        data.velocities = velocities
        data.linelist = lines
        orders.append(data)
        coefficients.append(coefs)
    return DataList.from_datalist(orders, verbose=0), np.concatenate(coefficients)


@pytest.mark.parametrize("regularization, scaled", [(0., True), (0.3, False), (0.3, True)])
def test_joint_solve_matches_weighted_least_squares(regularization, scaled):
    orders, theta = make_orders(regularization=regularization, scale_regularization=scaled)
    orders.run_multi_order_ACID(run_mcmc=False)
    joint = orders.multi_order_data
    model = MCMC(joint)
    continuum = utils.eval_continuum(model.x, theta, slices=model.continuum_slices)
    target = -np.log(model.y / continuum)
    weighted = model.alpha / (model.yerr / model.y)[:, None]
    lhs = weighted.T @ weighted
    difference = np.diff(np.eye(len(joint.velocities)), axis=0)
    scale = np.trace(lhs) / len(joint.velocities) if scaled else 1.
    lhs += regularization * scale * difference.T @ difference
    expected = np.linalg.solve(lhs, weighted.T @ (target / (model.yerr / model.y)))
    forward, profile = model.model_function(theta)

    np.testing.assert_allclose(profile, expected, atol=1e-12)
    np.testing.assert_allclose(forward, np.exp(-model.alpha @ expected) * continuum)
    if not regularization:
        np.testing.assert_allclose(profile, [0.07, 0.12, 0.07], atol=1e-12)
    assert joint.ndim == len(theta) == 7
    assert joint.initial_state.shape == (24, 7)
    assert all(data.sampler is None and data.config.run_mcmc for data in orders)
    assert model(np.r_[-1., theta[1:]]) == -np.inf


@pytest.mark.parametrize("use_jax", [False, True])
def test_joint_groups_recover_distinct_profiles_with_missing_groups(tmp_path, use_jax):
    if use_jax:
        pytest.importorskip("jax")
    groups = np.array([2, 2, 2, 7, 7, 7])
    orders, theta = make_orders(nframes=2, profile_groups=groups, use_jax=use_jax)
    expected = np.array([[.07, .12, .07], [.03, .2, .09]])
    for data in orders:
        wave = data.wavelengths["input"]
        spectra = []
        for frame, grid in enumerate(wave):
            grid = grid + frame * .003
            wave[frame] = grid
            alpha, _ = LSD.calc_mp_alpha(grid, data.velocities, data.linelist["wavelengths"],
                                        -np.log1p(-data.linelist["depths"]), groups, verbose=0)
            index = data.config.order - 10
            start = [0, 2, 5][index]
            coefs = theta[start:start + data.config.poly_ord + 1]
            continuum = utils.eval_continuum(utils.normalize_wavelengths(grid), coefs)
            spectra.append(continuum * np.exp(-LSD.flatten_alpha(alpha) @ (expected + frame * .01).ravel()))
        data.set_inputs(wave, np.array(spectra), data.errors["input"], data.sn["input"])

    orders.save_dir = str(tmp_path)
    result = orders.run_multi_order_ACID(pack=True)
    joint = result.data
    model = MCMC(joint)
    continuum = utils.eval_continuum(model.x, theta, slices=model.continuum_slices)
    weighted = model.alpha / (model.yerr / model.y)[:, None]
    reference = np.linalg.lstsq(weighted, -np.log(model.y / continuum) / (model.yerr / model.y), rcond=None)[0]
    np.testing.assert_allclose(model.model_function(theta)[1], reference, atol=1e-12)
    assert joint.profile["final"][0].shape == (2, 3)
    assert joint.profile["single_profile"][0].shape == (3,)
    fig, _ = orders.plot_combined_profile(return_fig=True)
    plt.close(fig)
    assert all(profile[0].shape == (2, 3) for profile in joint.profiles)
    assert all(np.all(np.isfinite(profile[0])) and np.all(profile[1] > 0) for profile in joint.profiles)
    np.testing.assert_allclose(result._get_order_alpha(joint.wavelengths["mcmc"], "mcmc"), joint.alpha["mcmc"])
    for frame in range(2):
        alpha = result._get_order_alpha(joint.wavelengths["input"][frame], f"input_{frame}")
        assert alpha.shape == (2, sum(len(data.wavelengths["input"][frame]) for data in orders), 3)
        slices = joint.continuum_slices[f"input_{frame}"]
        continuum = np.concatenate([utils.eval_continuum(utils.normalize_wavelengths(
            joint.wavelengths["input"][frame][i:j]), theta[a:b]) for i, j, a, b in slices])
        recovered = np.linalg.lstsq(LSD.flatten_alpha(alpha),
                                   -np.log(joint.flux["input"][frame] / continuum), rcond=None)[0]
        np.testing.assert_allclose(recovered.reshape(2, 3), expected + frame * .01, atol=1e-12)
    restored = DataList.load(str(tmp_path), verbose=0).multi_order_data.result
    np.testing.assert_array_equal(restored.data.profile_groups, groups)
    restored.continue_sampling(nsteps=2)
    assert restored[0].shape == (2, 3)
    fig, ax = restored.plot_profiles(return_fig=True)
    assert ax.get_legend_handles_labels()[1] == ["Group 2", "Group 7", "Group 2", "Group 7"]
    plt.close(fig)
    fig, _ = restored.plot_forward_model(separate_orders=True, show_linelist=True, return_fig=True)
    plt.close(fig)


def test_joint_groups_reject_conflicting_shared_line_assignments():
    orders = make_overlapping_orders()
    for data in orders:
        data.config.profile_groups = np.array([0, 1, 0, 1, 0, 1])
    orders[41].config.profile_groups = np.array([0, 1, 1, 1, 0, 1])
    with pytest.raises(ACIDInputError, match="same profile group"):
        orders.run_multi_order_ACID(run_mcmc=False)


def test_joint_invalid_pixels_warn_once_across_orders():
    orders, _ = make_orders()
    for data in orders:
        data.flux["input"][0][0] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once", ACIDDroppedDataWarning)
        orders.run_multi_order_ACID(run_mcmc=False)
    assert sum(w.category is ACIDDroppedDataWarning for w in caught) == 1


def test_joint_sampling_frames_continuum_errors_and_persistence(tmp_path):
    orders, _ = make_orders(nframes=2)
    orders.save_dir = str(tmp_path)
    result = orders.run_multi_order_ACID(pack=True)
    assert isinstance(result, Result)
    assert result.sampler.get_chain().shape == (40, 24, 7)
    assert result.data.complete
    assert len(result.data.profiles) == 2
    assert all(np.all(np.isfinite(p[0])) and np.all(p[1] > 0) for p in result.data.profiles)
    assert np.all(result.data.continuum_error["final"] > 0)
    np.testing.assert_allclose(orders.combined_profile[0], result[0])
    orders.combine_profiles()
    with pytest.raises(ACIDInputError, match="exclude orders"):
        orders.combine_profiles(exclude=10)

    # Joint covariance uses all orders rather than combining repeated copies.
    wave = result.data.wavelengths["final"]
    flux = result.data.fitted_flux["final"]
    errors = result.data.fitted_errors["final"]
    alpha = result.data.alpha["final"]
    valid = errors < 1e11
    weighted = alpha[valid] / (errors[valid] / flux[valid])[:, None]
    expected = np.linalg.inv(weighted.T @ weighted)
    profile = result[0]
    np.testing.assert_allclose(result[2], expected * np.outer(profile, profile))
    assert result.data.continuum["final"].shape == wave.shape

    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.copy2(tmp_path / "datalist.pkl", moved / "datalist.pkl")
    shutil.copytree(tmp_path / "multi_order", moved / "multi_order")
    loaded = DataList.load(str(moved), verbose=0)
    np.testing.assert_allclose(loaded.combined_profile[0], result[0])
    restored = loaded.multi_order_data.result
    restored.continue_sampling(nsteps=5)
    assert restored.data.nsteps == 45
    assert restored.data.continuum_slices["mcmc"].shape == (3, 4)
    assert restored.config.save_path == str(moved / "multi_order" / "data.pkl")
    np.testing.assert_allclose(loaded.combined_profile[0], restored[0])
    assert restored.config.order == [10, 11, 12]
    for saved, original in zip(restored.data.order_ll_masks, result.data.order_ll_masks):
        np.testing.assert_array_equal(saved, original)


@pytest.mark.parametrize("discard", [False, True])
def test_joint_selection_discards_failed_and_unselected_orders(discard):
    orders, _ = make_orders()
    orders[11].linelist = {"wavelengths": np.array([6000.]), "depths": np.array([0.3])}
    with pytest.warns(UserWarning, match="line list range error"):
        orders.run_multi_order_ACID(orders=[10, 11], run_mcmc=False,
                                    discard_unused_orders=discard)
    assert orders.orders.tolist() == ([10] if discard else [10, 11, 12])
    assert orders.multi_order_data.continuum_slices["mcmc"].shape == (1, 4)
    assert orders.multi_order_data.ndim == 2
    assert orders.multi_order_data.config.order == [10]


def test_joint_fit_reports_no_successful_orders():
    orders, _ = make_orders()
    with pytest.raises(ACIDStateError, match="No orders"):
        orders.run_multi_order_ACID(orders=[], run_mcmc=False)


@pytest.mark.parametrize("key, value", [("od", False), ("continuum_method", "chebval"),
                                       ("regularization", 0.3)])
def test_joint_fit_rejects_incompatible_order_settings(key, value):
    orders, _ = make_orders()
    setattr(orders[11].config, key, value)
    with pytest.raises(ACIDInputError, match=f"same {key}"):
        orders.run_multi_order_ACID(run_mcmc=False)


def test_single_order_joint_fit_matches_existing_model():
    orders, _ = make_orders()
    orders.run_multi_order_ACID(orders=[10], run_mcmc=False)
    data = orders[10]
    theta = data.poly_coeffs["masked"]
    joint = MCMC(orders.multi_order_data)
    original = MCMC(data)
    np.testing.assert_allclose(joint.model_function(theta)[1], original.model_function(theta)[1])
    assert joint(theta) == pytest.approx(original(theta))


def test_joint_convergence_sampling_uses_all_continuum_parameters():
    orders, _ = make_orders(max_steps=6, check_interval=3)
    result = orders.run_multi_order_ACID()
    assert result.data.nsteps == 6
    assert result.sampler.get_chain().shape == (6, 24, 7)


def test_joint_initialisation_allows_large_walker_counts():
    orders, _ = make_orders(nwalkers=1002)
    orders.run_multi_order_ACID(run_mcmc=False)
    assert orders.multi_order_data.initial_state.shape == (1002, 7)


def test_joint_fit_real_harps_orders(harps_paths, linelist_path):
    from conftest import extract_harps_e2ds
    from ACID_code import calc_deltav

    wave, flux, errors, sn = extract_harps_e2ds(harps_paths["e2ds"], harps_paths["flat"])
    velocities = np.arange(-25., 25., calc_deltav(wave[40]))
    orders = DataList(wave[40:42], flux[40:42], errors[40:42], sn[40:42],
                      velocities, str(linelist_path), order_range=[40, 41],
                      parallel=False, nsteps=30, verbose=0, seed=1)
    result = orders.run_multi_order_ACID()
    assert orders.orders.tolist() == [40, 41]
    assert result.data.ndim == 8
    assert result[0].shape == velocities.shape
    assert np.all(np.isfinite(result[0]))
    assert np.all(result[1] > 0)


@pytest.mark.parametrize("use_jax", [False, True])
def test_joint_sampling_with_parallel_workers(use_jax):
    if use_jax:
        pytest.importorskip("jax")
    orders, _ = make_orders(use_jax=use_jax, cores=2)
    for data in orders:
        data.config.parallel = True
        data.config.nsteps = 10
    result = orders.run_multi_order_ACID()
    assert result.sampler.get_chain().shape == (10, 24, 7)
    assert np.all(np.isfinite(result[0]))


@pytest.mark.parametrize("method", ["polyval", "chebval"])
def test_joint_jax_probability_matches_numpy(method):
    pytest.importorskip("jax")
    orders, _ = make_orders(continuum_method=method)
    orders.run_multi_order_ACID(run_mcmc=False)
    data = orders.multi_order_data
    theta = data.poly_coeffs["masked"]
    numpy_model = MCMC(data)
    data.config.use_jax = True
    jax_model = MCMC(data)
    assert jax_model.jax_enabled
    assert jax_model(theta) == pytest.approx(numpy_model(theta), rel=1e-10)
    assert jax_model.dynesty_logprob(theta) == pytest.approx(numpy_model.dynesty_logprob(theta), rel=1e-10)


def make_overlapping_orders():
    velocities = np.array([-5., 0., 5.])
    lines = {"wavelengths": np.array([5003., 5007., 5011., 5011.3, 5015., 5019.]),
             "depths": np.array([.2, .3, .25, .015, .2, .3])}
    orders = []
    for index, sn in enumerate([10., 100.]):
        wave = np.linspace(5000. + 10 * index, 5012. + 10 * index, 601)
        # Different frame grids exercise alpha recalculation after sampling.
        waves = np.array([wave, wave + .003])
        used = lines["depths"] >= 1 / (3 * sn)
        spectra = []
        for frame_wave in waves:
            alpha = LSD.calc_alpha(frame_wave, lines["wavelengths"][used],
                                   -np.log1p(-lines["depths"][used]), velocities)
            continuum = 1. + .2 * index + .03 * utils.normalize_wavelengths(frame_wave)
            spectra.append(continuum * np.exp(-alpha @ np.array([.07, .12, .07])))
        data = Data()
        data.config = Config(order=40 + index, poly_ord=1, n_bins=10, masking_lines={},
                             sigma_lower=100., sigma_upper=100., dev_perc=100.,
                             parallel=False, nsteps=30, verbose=0, seed=1)
        data.set_inputs(waves, np.array(spectra), np.full_like(waves, .01), [sn, sn])
        data.velocities = velocities
        data.linelist = lines
        orders.append(data)
    return DataList.from_datalist(orders, verbose=0)


@pytest.mark.parametrize("grouped", [False, True])
def test_overlapping_orders_keep_interpolation_and_line_cuts_independent(monkeypatch, grouped):
    orders = make_overlapping_orders()
    if grouped:
        for data in orders:
            data.config.profile_groups = np.array([2, 7, 2, 7, 2, 7])
    original = LSD.calc_alpha
    grids = []

    def record_grid(wave, *args, **kwargs):
        grids.append(np.asarray(wave).copy())
        return original(wave, *args, **kwargs)

    monkeypatch.setattr(LSD, "calc_alpha", staticmethod(record_grid))
    result = orders.run_multi_order_ACID()
    data = result.data
    assert len(result.data.profiles) == 2
    assert np.any(np.diff(data.wavelengths["final"]) < 0) # overlap was not sorted or merged
    assert 3 not in data.order_ll_masks[0] and 3 in data.order_ll_masks[1]
    assert all(len(wave) == 601 for wave in grids) # interpolation only receives individual grids
    assert len(grids) >= 4 # setup and individual frame recalculations
    np.testing.assert_array_equal(data.full_mask, np.concatenate([order.full_mask for order in orders]))

    recalculated = result._get_order_alpha(data.wavelengths["mcmc"], "mcmc")
    np.testing.assert_allclose(recalculated, data.alpha["mcmc"])
    for (i, j, _, _), order in zip(data.continuum_slices["final"], orders):
        alpha = data.alpha["final"][:, i:j] if grouped else data.alpha["final"][i:j]
        np.testing.assert_array_equal(alpha, order.alpha["initial"])


@pytest.mark.parametrize("show_masking", [False, True])
def test_joint_forward_plots_select_orders_and_use_independent_axes(show_masking):
    orders = make_overlapping_orders()
    result = orders.run_multi_order_ACID()
    data = result.data
    original_flux = data.flux["final"].copy()
    fig, axes = result.plot_forward_model(order=41, show_masking=show_masking,
                                         show_linelist=True, return_fig=True)
    observed = axes[0].lines[0]
    i, j, _, _ = data.continuum_slices["final"][1]
    np.testing.assert_array_equal(observed.get_xdata(), data.forward_x["final"][i + 2:j - 2])
    markers = [collection for collection in axes[0].collections if collection.get_label() == "Line List"]
    for segment in markers[0].get_segments():
        expected_top = np.interp(segment[0, 0], data.forward_x["final"][i:j], data.continuum["final"][i:j])
        assert segment[0, 1] == pytest.approx(expected_top)
    plt.close(fig)

    fig, axes = result.plot_forward_model(separate_orders=True, show_masking=show_masking,
                                         show_linelist=True, return_fig=True)
    assert axes.shape == (2, 2)
    assert "Order 40" in axes[0, 0].get_title() and "Order 41" in axes[1, 0].get_title()
    assert not axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[1, 0])
    assert not axes[0, 0].get_shared_y_axes().joined(axes[0, 0], axes[1, 0])
    plt.close(fig)

    fig, axes = result.plot_forward_model(show_masking=show_masking, return_fig=True)
    assert len(axes) == 2
    observed = [line for line in axes[0].lines if line.get_label() == "Observed Spectrum"]
    assert len(observed) == 2
    assert all(np.all(np.diff(line.get_xdata()) > 0) for line in observed)
    plt.close(fig)
    np.testing.assert_array_equal(data.flux["final"], original_flux)
    with pytest.raises(ACIDInputError, match="unavailable"):
        result.plot_forward_model(order=99, return_fig=True)


def test_joint_plot_recovers_missing_masks_in_older_saved_results(monkeypatch):
    orders = make_overlapping_orders()
    result = orders.run_multi_order_ACID()
    data = result.data
    expected = data.full_mask.copy()
    expected[10] = True
    masked_wave = data.wavelengths["final"][10]
    data.wavelengths["mcmc"] = np.delete(data.wavelengths["mcmc"],
                                         np.flatnonzero(data.wavelengths["mcmc"] == masked_wave))
    # This pixel belongs only to the first order; update the saved slice lengths.
    data.continuum_slices["mcmc"][0, 1] -= 1
    data.continuum_slices["mcmc"][1, :2] -= 1
    data.full_mask = None
    seen = []
    original = utils.plot_masked_line

    def capture_mask(axis, wave, values, mask, **kwargs):
        seen.append(mask.copy())
        return original(axis, wave, values, mask, **kwargs)

    monkeypatch.setattr(utils, "plot_masked_line", capture_mask)
    fig, _ = result.plot_forward_model(return_fig=True)
    np.testing.assert_array_equal(seen[0], ~expected[2:599])
    plt.close(fig)
