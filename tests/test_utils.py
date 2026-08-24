#%%
"""Focused tests for the numerical, persistence, plotting, and environment helpers."""
import matplotlib.pyplot as plt
import numpy as np
import os
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import utils


@pytest.mark.parametrize("method", ["polyval", "chebval"])
def test_continuum_fit_and_evaluation_round_trip(method):
    # Both continuum representations should recover a known positive quadratic.
    x = np.linspace(-1, 1, 21)
    y = 1.2 + 0.1 * x + 0.05 * x ** 2
    coefficients = utils.fit_continuum(x, y, 2, method=method)
    fitted = utils.eval_continuum(x, coefficients, method=method)

    np.testing.assert_allclose(fitted, y, atol=2e-3)
    with pytest.raises(ValueError, match="Unknown method"):
        utils.fit_continuum(x, y, 2, method="invalid")
    with pytest.raises(ValueError, match="Unknown method"):
        utils.eval_continuum(x, coefficients, method="invalid")


def test_move_conversion_supports_kwargs_and_validates_specs():
    # Two- and three-field specifications should instantiate weighted emcee moves.
    converted = utils.convert_moves_to_emcee([
        ("StretchMove", 0.6),
        ("DEMove", 0.4, {"gamma0": 1.0}),
    ])
    assert [weight for _, weight in converted] == [0.6, 0.4]
    assert converted[0][0].__class__.__name__ == "StretchMove"

    # Tuple length, kwargs type, and unknown move name are distinct input failures.
    with pytest.raises(ValueError, match="length 2 or 3"):
        utils.convert_moves_to_emcee([("StretchMove",)])
    with pytest.raises(ValueError, match="kwargs must be a dictionary"):
        utils.convert_moves_to_emcee([("StretchMove", 1.0, [])])
    with pytest.raises(ValueError, match="not a valid"):
        utils.convert_moves_to_emcee([("MissingMove", 1.0)])


def test_invalid_pixel_masking_dropping_and_edge_removal():
    # Invalid pixels should either be retained as NaN or removed using one shared mask.
    wavelengths = np.array([1.0, 2.0, np.nan, 4.0])
    flux = np.array([1.0, -1.0, 1.0, 0.8])
    errors = np.array([0.1, 0.1, 0.1, np.inf])
    masked = utils.mask_invalid(wavelengths, flux, errors, return_mask=True, verbose=0)
    dropped = utils.drop_invalid(wavelengths, flux, errors, return_mask=True, verbose=0)

    np.testing.assert_array_equal(masked[-1], [True, False, False, False])
    np.testing.assert_array_equal(dropped[0], [1.0])
    np.testing.assert_array_equal(dropped[-1], masked[-1])
    np.testing.assert_array_equal(utils.drop_edges(np.arange(8), 2), [2, 3, 4, 5])


def test_sn_error_collapse_and_wavelength_normalisation():
    # S/N estimation and its inverse error estimate should agree on clean constant data.
    wavelengths = np.linspace(5000, 5010, 12)
    flux = np.ones(12)
    errors = np.full(12, 0.02)
    sn = utils.guess_SNR(wavelengths, flux, errors)
    estimated_errors = utils.guess_errors(flux, sn)

    assert sn == pytest.approx(50)
    np.testing.assert_allclose(estimated_errors, errors)

    # Per-pixel S/N collapses over the central two-thirds, and wavelengths map to [-1, 1].
    collapsed = utils.collapse_SNR(np.arange(1, 13, dtype=float), wavelengths)
    normalised = utils.normalize_wavelengths(wavelengths)
    assert collapsed == pytest.approx(6.5)
    np.testing.assert_allclose(normalised[[0, -1]], [-1, 1])
    a, b = utils.get_normalisation_coeffs(wavelengths)
    np.testing.assert_allclose(a * wavelengths + b, normalised)

    # Non-positive physical inputs are rejected before the calculations proceed.
    with pytest.raises(ValueError, match="positive"):
        utils.guess_SNR(wavelengths, -flux, errors)
    with pytest.raises(ValueError, match="positive"):
        utils.calc_deltav(np.array([-1.0, 1.0]))


def test_dictionary_defaults_and_file_discovery(tmp_path):
    # Existing user values win, while missing keys are copied from defaults.
    merged = utils.set_dict_defaults({"a": 9}, {"a": 1, "b": 2})
    assert merged == {"a": 9, "b": 2}
    assert utils.set_dict_defaults(None, {"a": 1}) == {"a": 1}

    # The discovery helper searches one nested level and excludes corrected products.
    night = tmp_path / "night"
    night.mkdir()
    kept = night / "HARPS_e2ds_A.fits"
    corrected = night / "HARPS_e2ds_A_corrected.fits"
    kept.touch()
    corrected.touch()
    assert utils.findfiles(str(tmp_path), "e2ds") == [str(kept)]


def test_robust_mean_and_profile_combination_modes():
    # A large outlier should not control the robust location estimate.
    samples = np.array([[1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [100.0, 200.0]])
    np.testing.assert_allclose(utils.robust_mean(samples, nsig=3, axis=0), [1.0, 2.0])

    # Exercise unweighted, diagonal-error, and full-covariance profile combination.
    profiles = np.array([[1.0, 0.8], [1.0, 0.6]])
    errors = np.array([[0.1, 0.1], [0.2, 0.2]])
    covariances = np.array([np.diag(row ** 2) for row in errors])
    unweighted = utils.combine_profiles(profiles)
    weighted = utils.combine_profiles(profiles, errors=errors)
    covariance_weighted = utils.combine_profiles(profiles, errors=errors,
                                                  covariances=covariances)

    np.testing.assert_allclose(unweighted, [1.0, 0.7])
    np.testing.assert_allclose(weighted[0], covariance_weighted[0])
    np.testing.assert_allclose(weighted[1], covariance_weighted[1])
    np.testing.assert_allclose(covariance_weighted[1] ** 2,
                               np.diag(covariance_weighted[2]))


def test_flux_optical_depth_conversion_modes_and_dependencies():
    # Convert every supported quantity together so covariance scaling is also checked.
    flux = np.array([0.8, 0.9])
    errors = np.array([0.02, 0.03])
    depths = np.array([0.2, 0.1])
    covariance = np.diag(errors ** 2)
    converted = utils.flux_to_od(flux, errors, depths, covariance)
    restored = utils.od_to_flux(*converted)

    for actual, expected in zip(restored, (flux, errors, depths, covariance)):
        np.testing.assert_allclose(actual, expected)

    # od=False is a deliberate no-op used by legacy linear-flux LSD.
    passthrough = utils.flux_to_od(flux, errors, depths, covariance, od=False)
    for actual, expected in zip(passthrough, (flux, errors, depths, covariance)):
        assert actual is expected

    # Errors and covariance cannot be transformed without their associated flux.
    with pytest.raises(ValueError, match="flux.*provided"):
        utils.flux_to_od(errors=errors)
    with pytest.raises(ValueError, match="data.*provided"):
        utils.od_to_flux(errors=errors)


def test_masked_line_plot_builds_coloured_segments():
    # The helper adds two legend handles and one segment collection to an existing axis.
    figure, axis = plt.subplots()
    utils.plot_masked_line(axis, np.arange(4), np.arange(4),
                           np.array([True, True, False, False]))

    assert len(axis.lines) == 2
    assert len(axis.collections) == 1
    assert len(axis.collections[0].get_segments()) == 3
    plt.close(figure)


def test_multiprocessing_environment_and_memory_detection(monkeypatch):
    # Outside SLURM the helper applies emcee's single-thread environment recommendation.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    utils.configure_mp_environ(os)
    assert os.environ["OMP_NUM_THREADS"] == os.environ["MKL_NUM_THREADS"] == "1"

    # Inside SLURM incorrect thread settings fail, while allocated memory is reported in bytes.
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "256")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="must be set to 1"):
        utils.configure_mp_environ(os)
    assert utils.get_available_memory() == 256 * 1024 ** 2


def test_autocorrelation_helpers_and_power_of_two_validation():
    # The FFT helper should return a normalised ACF beginning at exactly one.
    signal = np.sin(np.linspace(0, 4 * np.pi, 64))
    acf = utils.autocorr_func_1d(signal)
    walkers = np.vstack([signal, np.roll(signal, 1)])

    assert acf[0] == pytest.approx(1)
    assert np.isfinite(utils.autocorr_gw2010(walkers))
    assert np.isfinite(utils.autocorr_new(walkers))
    assert utils.next_pow_2(0) == 1
    assert utils.next_pow_2(9) == 16
    assert utils.auto_window(np.ones(5), c=0) == 4

    # Dimensional and sign validation should fail before attempting the FFT.
    with pytest.raises(ValueError, match="dimensions"):
        utils.autocorr_func_1d(np.ones((2, 2)))
    with pytest.raises(ValueError, match="non-negative"):
        utils.next_pow_2(-1)


def test_sampler_backend_copy_size_and_reconstruction(tmp_path, harps_result):
    # Copy the shared sampler backend to HDF5 without running another chain.
    backend_path = tmp_path / "sampler.h5"
    copied = utils.save_backend_to_hdf5(harps_result.sampler.backend, str(backend_path))
    reconstructed = utils.backend_to_sampler(copied, lambda theta: -0.5 * np.sum(theta ** 2))

    np.testing.assert_array_equal(copied.get_chain(), harps_result.sampler.get_chain())
    np.testing.assert_array_equal(reconstructed.get_chain(), copied.get_chain())
    assert utils.sampler_nbytes(harps_result.sampler) > 0


def test_show_or_save_writes_the_requested_figure(tmp_path):
    # Supplying a directory should save and close the current figure under the given name.
    plt.figure()
    plt.plot([0, 1], [0, 1])
    utils.show_or_save(plt, str(tmp_path), "diagnostic.png", verbose=0)

    assert (tmp_path / "diagnostic.png").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
