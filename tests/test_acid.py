#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import ACID, ACID_HARPS, Acid, Config, Data, LineList
from ACID_code.acid import _get_init_and_run_kwargs


def test_acid_runs_preprocessing_without_mcmc(harps_order_40):
    # Use the real order-40 extraction, but stop before the sampler for a fast pipeline check.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist)

    result = acid.ACID(wavelengths, flux, errors, sn, run_mcmc=False)

    # Preprocessing stores its intermediate profile and alpha matrix on Data.
    assert result is None
    assert "masked" in acid.data.profile
    assert acid.data.alpha["mcmc"].shape[1] == len(velocities)


def test_acid_accepts_multiple_frames_without_sampling(harps_order_40):
    # Duplicate one observation deliberately: the input shape, not astrophysical variation, is under test.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist)
    acid.ACID(np.array([wavelengths, wavelengths]), np.array([flux, flux]),
              np.array([errors, errors]), np.array([sn, sn]), run_mcmc=False)

    # ACID keeps frames separate while producing a common velocity-grid profile.
    assert len(acid.data.wavelengths["input"]) == 2
    assert acid.data.profile["masked"][0].shape == velocities.shape


def test_legacy_wrapper_maps_positional_arguments(harps_order_40):
    # The legacy top-level function has a different positional argument order.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40

    result = ACID(wavelengths, flux, errors, linelist, sn, velocities, run_mcmc=False)

    assert result is None


def test_output_directory_creates_only_the_requested_leaf(tmp_path):
    output_dir = tmp_path / "acid_run"
    acid = Acid(dir=str(output_dir), verbose=0)

    assert acid.config.dir == str(output_dir)
    assert acid.config.save_path == str(output_dir / "data.pkl")
    assert acid.config.sampler_path == str(output_dir / "sampler.h5")
    assert acid.config.figure_dir == str(output_dir / "figures")
    assert output_dir.is_dir()
    assert (output_dir / "figures").is_dir()

    config_dir = tmp_path / "configured_run"
    configured = Acid(config=Config(dir=str(config_dir)), verbose=0)
    assert configured.config.save_path == str(config_dir / "data.pkl")
    assert configured.config.figure_dir == str(config_dir / "figures")

    mistyped_parent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="Only the final directory is created"):
        Config(dir=str(mistyped_parent / "acid_run"))
    assert not mistyped_parent.exists()


@pytest.mark.parametrize("verbose, expected", [(False, 0), ("high", 3), (4, 4)])
def test_verbosity_inputs_are_preserved_during_harps_preprocessing(harps_order_40,
                                                                    verbose, expected, capsys):
    """The legacy verbosity modes should not change the preprocessing result."""
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    # Each supported verbosity spelling should configure the same science calculation.
    acid = Acid(velocities=velocities, linelist=linelist, verbose=verbose)
    acid.ACID(wavelengths, flux, errors, sn, run_mcmc=False)

    assert acid.config.verbose == expected
    assert "masked" in acid.data.profile
    if expected == 0:
        assert capsys.readouterr().out == ""


def test_acid_requires_complete_input_and_rejects_unknown_keyword(harps_order_40):
    # Unknown options should be rejected at ACID's public boundary.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist)

    with pytest.raises(ValueError, match="not recognised"):
        acid.ACID(wavelengths, flux, errors, sn, made_up_setting=True)
    # Missing required line-list data should also give a domain-specific exception.
    with pytest.raises(ValueError, match="linelist"):
        Acid(velocities=velocities).ACID(wavelengths, flux, errors, sn, run_mcmc=False)


def test_acid_drops_invalid_spectrum_and_linelist_values(harps_order_40):
    """Invalid values from the legacy edge-case test should survive the public workflow."""
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    line_wavelengths, line_depths = LineList.validate_linelist(linelist)
    line_wavelengths, line_depths = line_wavelengths.copy(), line_depths.copy()
    wavelengths, flux, errors = wavelengths.copy(), flux.copy(), errors.copy()
    line_wavelengths[0] = np.nan
    line_depths[1] = np.nan
    wavelengths[10] = np.nan
    flux[20] = np.nan
    errors[30] = np.inf
    acid = Acid(velocities=velocities, linelist={"wavelengths": line_wavelengths, "depths": line_depths})

    assert acid.ACID(wavelengths, flux, errors, sn, run_mcmc=False) is None
    assert np.all(np.isfinite(acid.data.wavelengths["combined"]))
    assert np.all(np.isfinite(acid.data.flux["combined"]))
    assert np.all(np.isfinite(acid.data.errors["combined"]))
    assert len(acid.data.linelist["wavelengths"]) == len(line_wavelengths) - 2


def test_acid_estimates_errors_from_per_pixel_sn(harps_order_40):
    """The legacy per-pixel S/N route should work through Acid, not only Data."""
    wavelengths, flux, _, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist)

    assert acid.ACID(wavelengths, flux, sn=np.full_like(flux, sn), run_mcmc=False) is None
    assert acid.data.errors["input"].shape == flux[None].shape
    assert np.all(np.isfinite(acid.data.errors["input"]))


def test_acid_preserves_explicit_sn_when_errors_are_also_provided(harps_order_40):
    """An explicit S/N must not be replaced by one estimated from supplied errors."""
    wavelengths, flux, errors, _, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist, seed=1)
    result = acid.ACID(wavelengths, flux, errors, 10000, nsteps=12, nwalkers=12)

    np.testing.assert_array_equal(result.data.sn["input"], [10000])
    assert result.data.sn["combined"] == 10000
    assert result.data.sn["final"] == 10000


def test_continuum_plots_are_available_after_preprocessing(harps_order_40):
    # Keep the run sampler-free: only the two continuum plotting states are needed here.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist)
    acid.ACID(wavelengths, flux, errors, sn, run_mcmc=False)

    # Initial and residual-masked continua are distinct diagnostic stages.
    initial, initial_ax = acid.data.plot_continuum_fit("initial", return_fig=True)
    masked, masked_ax = acid.data.plot_continuum_fit("masked", return_fig=True)

    assert initial_ax.get_title() == "Initial Continuum Fit"
    assert masked_ax.get_title() == "Continuum Fit after Residual Masking"
    plt.close(initial)
    plt.close(masked)


def test_debug_result_stores_extra_lsd_and_indexes_multi_profiles(harps_order_40):
    # Build explicit groups from the full line list to exercise multi-profile bookkeeping.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    linelist_wavelengths, linelist_depths = LineList.validate_linelist(linelist)
    linelist = {"wavelengths": linelist_wavelengths, "depths": linelist_depths}
    # Debug verbosity asks Result to retain the additional LSD products.
    acid = Acid(velocities=velocities, linelist=linelist, verbose=4)
    result = acid.ACID(wavelengths, flux, errors, sn,
                       profile_groups=np.arange(len(linelist_wavelengths)) % 2,
                       nsteps=12, nwalkers=12)

    # Indexing must select group, frame, profile/error components consistently.
    assert "lsd_final" in result.data.debug
    assert result[0].shape == (2, len(velocities))
    np.testing.assert_array_equal(result[0, 0], result["profile"])
    np.testing.assert_array_equal(result[0, 0, 1], result["error"])


@pytest.mark.parametrize("method", ["polyval", "chebval"])
def test_scipy_continuum_fit_stores_each_intermediate_product(method):
    # Build a positive curved continuum with enough bins for a quadratic fit.
    wavelengths = np.linspace(5000, 5010, 100)
    norm_wavelengths = np.linspace(-1, 1, 100)
    flux = 1.2 + 0.05 * norm_wavelengths + 0.02 * norm_wavelengths ** 2
    errors = np.full_like(flux, 0.01)
    data = Data()
    data.config = Config(poly_ord=2, continuum_percentile=50, continuum_method=method)
    data.wavelengths["test"] = wavelengths
    data.flux["test"] = flux
    data.errors["test"] = errors
    data.line_mask = np.zeros_like(flux, dtype=bool)

    # Continuum fitting should populate coefficients, continuum, fitted data, and plot inputs.
    Acid.scipy_continuum_fit(data, "test")
    for mapping in (data.norm_wavelengths, data.poly_coeffs, data.continuum,
                    data.fitted_flux, data.fitted_errors, data.plotting_variables):
        assert "test" in mapping
    np.testing.assert_allclose(data.fitted_flux["test"], 1.0, atol=3e-3)


def test_scipy_continuum_fit_rejects_insufficient_unmasked_bins():
    # A cubic requires four good bins, but masked-sized errors deliberately leave only three.
    data = Data()
    data.config = Config(poly_ord=3, n_bins=5, continuum_method="polyval")
    data.wavelengths["test"] = np.linspace(5000, 5010, 20)
    data.flux["test"] = np.ones(20)
    data.errors["test"] = np.r_[np.full(12, 0.01), np.full(8, 1e12)]
    data.line_mask = np.zeros(20, dtype=bool)

    with pytest.raises(ValueError, match="Insufficient good points"):
        Acid.scipy_continuum_fit(data, "test")


def test_initial_state_and_sampler_kwargs_use_preprocessed_result(harps_result):
    # Reconstruct an independent Data instance so walker setup cannot mutate the shared result.
    data = Data().from_dict(harps_result.data.to_dict())
    acid = Acid(data=data)
    acid.config.parallel = False
    acid.config.nwalkers = 12
    state = acid.get_initial_state()

    # Deterministic sampling contains only the continuum coefficients.
    assert state.shape == (12, acid.config.poly_ord + 1)
    assert np.all(np.isfinite(state))

    # Sampler and run kwargs should carry the derived dimensions and requested state unchanged.
    sampler_kwargs, run_kwargs = acid._get_sampler_kwargs(7, state)
    assert sampler_kwargs["nwalkers"] == 12
    assert sampler_kwargs["ndim"] == acid.config.poly_ord + 1
    assert run_kwargs["initial_state"] is state
    assert run_kwargs["nsteps"] == 7


def test_non_deterministic_initial_state_includes_every_profile_parameter(harps_result):
    # The legacy sampler adds one parameter for every column of the flattened alpha matrix.
    data = Data().from_dict(harps_result.data.to_dict())
    acid = Acid(data=data)
    acid.config.deterministic_profile = False
    acid.config.nwalkers = 12
    state = acid.get_initial_state()
    expected_ndim = acid.data.alpha["mcmc"].shape[1] + acid.config.poly_ord + 1

    assert acid.data.ndim == expected_ndim
    assert state.shape == (12, expected_ndim)


def test_sampler_and_result_properties_validate_acid_state(harps_result):
    # The sampler property is a transparent view onto the underlying Data instance.
    acid = Acid(data=harps_result.data)
    assert acid.sampler is harps_result.sampler
    assert acid.result.data is harps_result.data

    # An incomplete Data object cannot create a Result or continue a missing chain.
    incomplete = Acid(data=Data())
    with pytest.raises(ValueError, match="has not been run"):
        _ = incomplete.result
    with pytest.raises(ValueError, match="Either a state or an existing sampler"):
        incomplete.continue_sampling(nsteps=1)


def test_legacy_argument_splitter_routes_and_validates_arguments():
    # Translate one initialisation setting and two run settings using the legacy names.
    legacy_args = ["line", "vgrid", "input_wavelengths"]
    renamed = {"line": "linelist", "vgrid": "velocities",
               "input_wavelengths": "wavelengths"}
    init_kwargs, run_kwargs = _get_init_and_run_kwargs(
        legacy_args, renamed, "lines.txt", np.arange(3.0), np.arange(5.0), nsteps=10,
    )

    assert set(init_kwargs) == {"linelist", "velocities"}
    assert set(run_kwargs) == {"wavelengths", "nsteps"}
    with pytest.raises(TypeError, match="Too many positional"):
        _get_init_and_run_kwargs(["one"], {}, 1, 2)


def test_removed_harps_entry_points_raise_migration_message():
    # Both the class and legacy function deliberately direct users to explicit HARPS inputs.
    with pytest.raises(NotImplementedError, match="no longer supported"):
        Acid().ACID_HARPS()
    with pytest.raises(NotImplementedError, match="no longer supported"):
        ACID_HARPS()

def test_reinitialising_acid_with_existing_sampler_reuses_backend(harps_result):
    # The sampler property is a transparent view onto the underlying Data instance.
    acid = Acid(data=harps_result.data)
    assert acid.sampler is harps_result.sampler
    nstepsacid1 = np.copy(acid.data.nsteps)

    data = harps_result.data
    acid2 = Acid(data=data)
    acid2.ACID(nsteps=10)
    assert acid2.sampler.backend is acid.sampler.backend
    assert acid2.data.nsteps == nstepsacid1 + 10


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
