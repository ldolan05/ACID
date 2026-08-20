#%%
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import Config, LSD, MCMC
from ACID_code import mcmc as mcmc_module


@pytest.fixture
def mcmc(synthetic_spectrum):
    # Build the compact deterministic model directly, avoiding a full ACID run for unit maths.
    wavelengths, flux, errors, _, velocities, linelist = synthetic_spectrum
    alpha = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)
    return MCMC(np.linspace(-1, 1, len(flux)), flux, errors, alpha, velocities,
                LSD.calc_cholesky(alpha, errors), deterministic_profile=True)


def test_deterministic_model_and_probability_are_finite(mcmc):
    # A neutral continuum coefficient vector should be evaluable by the posterior callable.
    forward, profile = mcmc.deterministic_model([1.0, 0.0])

    assert forward.shape == mcmc.y.shape
    assert profile.shape == (mcmc.k_max,)
    assert np.isfinite(mcmc([1.0, 0.0]))


def test_full_model_uses_profile_and_continuum(synthetic_spectrum):
    # The non-deterministic model contains profile parameters followed by continuum coefficients.
    wavelengths, flux, errors, _, velocities, linelist = synthetic_spectrum
    alpha = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)
    model = MCMC(np.linspace(-1, 1, len(flux)), flux, errors, alpha, velocities,
                 deterministic_profile=False)
    forward, profile = model.full_model(np.r_[np.zeros(len(velocities)), [1.0]])

    np.testing.assert_allclose(forward, 1.0)
    np.testing.assert_array_equal(profile, 0.0)


@pytest.mark.parametrize("tau_list, steps, expected", [
    ([np.array([2.0])], 200, False),
    ([np.array([2.0]), np.array([2.01])], 200, True),
    ([np.array([np.nan]), np.array([2.0])], 200, False),
])
def test_stopping_criterion(mcmc, tau_list, steps, expected):
    # Feed fixed autocorrelation histories to isolate stopping logic from sampling noise.
    converged, _, _ = mcmc._get_mcmc_stopping_criterion(tau_list, steps, 1, 50, 0.1)
    assert converged is expected


def test_soft_prior_penalises_out_of_range_profiles(mcmc):
    # The profile prior should leave physical values unchanged and penalise extreme ones.
    assert mcmc.soft_z_prior(np.zeros(mcmc.k_max)) == 0
    assert mcmc.soft_z_prior(np.full(mcmc.k_max, 3.0)) < 0


def test_mcmc_initialises_from_completed_acid_data(harps_result):
    # Worker processes construct MCMC from Data, so every field must be recovered there.
    model = MCMC(harps_result.data)

    assert model.x is harps_result.data.norm_wavelengths["mcmc"]
    assert model.alpha is harps_result.data.alpha["mcmc"]
    assert model.deterministic_profile is True
    assert model.model_inputs.shape[0] == (model.k_max + harps_result.data.config.poly_ord + 1)


def test_deterministic_model_rejects_non_positive_fitted_flux(mcmc):
    # A negative constant continuum makes the fitted flux invalid in optical-depth space.
    fitted_flux, profile = mcmc.deterministic_model([-1.0, 0.0])

    assert np.all(fitted_flux < 0)
    np.testing.assert_array_equal(profile, np.full(mcmc.k_max, -2))


def test_linear_flux_model_uses_shifted_profile_parameters(synthetic_spectrum):
    # In legacy flux mode a flat physical profile is represented by parameters equal to one.
    wavelengths, flux, errors, _, velocities, linelist = synthetic_spectrum
    alpha = LSD.calc_alpha(wavelengths, linelist["wavelengths"], linelist["depths"], velocities)
    model = MCMC(np.linspace(-1, 1, len(flux)), flux, errors, alpha, velocities,
                 deterministic_profile=False, od=False)
    forward, shifted_profile = model.full_model(np.r_[np.ones(len(velocities)), [1.0]])

    np.testing.assert_array_equal(shifted_profile, 0.0)
    np.testing.assert_allclose(forward, 1.0)
    assert model.soft_z_prior(np.full(model.k_max, 1.0)) < 0


def test_continuum_and_combined_priors_cover_both_continuum_methods(mcmc):
    # Ordinary polynomial continua have no extra coefficient prior.
    assert mcmc.continuum_prior(np.array([1.0, 0.1])) == 0
    assert mcmc.log_prior(np.array([1.0, 0.1]), np.zeros(mcmc.k_max)) == 0

    # Chebyshev coefficients receive progressively tighter zero-centred priors.
    mcmc.continuum_method = "chebval"
    prior = mcmc.continuum_prior(np.array([1.0, 0.1, 0.1]))
    assert prior < 0
    assert mcmc.log_prior(np.array([1.0, 0.1, 0.1]), np.zeros(mcmc.k_max)) == prior


def test_dynesty_likelihood_and_prior_transform_use_acid_starting_model(harps_result):
    # The Data initialiser supplies the curve-fit solution needed by dynesty's unit-cube transform.
    model = MCMC(harps_result.data)
    ndim = harps_result.data.config.poly_ord + 1
    lower = model.ptform(np.zeros(ndim))
    centre = model.ptform(np.full(ndim, 0.5))
    upper = model.ptform(np.ones(ndim))

    np.testing.assert_allclose((lower + upper) / 2, centre)
    assert np.all(lower < upper)
    assert np.isfinite(model.dynesty_logprob(centre))


def test_stopping_description_formats_tolerance_and_effective_samples():
    # Progress text should show both values and the correct side of each threshold.
    config = Config(tau_tol=0.1, min_tau_factor=50)
    tolerance, effective = MCMC._get_tqdm_desc(0.05, 75, config)

    assert tolerance == "0.0500<0.1"
    assert effective == "75.00>50"


def test_multiprocessing_wrappers_delegate_to_the_worker_model(harps_result):
    # Initialise the module-level worker model exactly as multiprocessing does.
    mcmc_module._mp_init_worker(harps_result.data)
    model = mcmc_module._MCMC
    theta = np.asarray(model.model_inputs[model.k_max:])
    unit_cube = np.full(len(theta), 0.5)

    assert mcmc_module._mp_log_probability(theta) == pytest.approx(model(theta))
    assert mcmc_module._mp_log_likelihood(theta) == pytest.approx(model.dynesty_logprob(theta))
    np.testing.assert_allclose(mcmc_module._mp_ptform(unit_cube), model.ptform(unit_cube))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
