#%%
import numpy as np
import importlib.util
import os
import subprocess
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import Acid, Config, LSD, MCMC
from ACID_code import mcmc as mcmc_module
from ACID_code import jax as jax_module
from emcee.ensemble import walkers_independent


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
    assert mcmc([-1.0, 0.0]) == -np.inf


def test_requested_jax_falls_back_when_it_cannot_be_imported(mcmc, monkeypatch):
    # A missing optional accelerator must never make the ordinary MCMC path unusable.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(jax_module, "_import_jax", lambda: None)
    with pytest.warns(RuntimeWarning, match="falling back"):
        model = MCMC(mcmc.x, mcmc.y, mcmc.yerr, mcmc.alpha, mcmc.velocities,
                     mcmc.c_factor, deterministic_profile=True, use_jax=True)

    assert model.use_jax is True
    assert model.jax_enabled is False
    assert model([1.0, 0.0]) == pytest.approx(mcmc([1.0, 0.0]))
    np.testing.assert_allclose(model.log_probability_batch([[1.0, 0.0], [-1.0, 0.0]]),
                               [mcmc([1.0, 0.0]), -np.inf])


@pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                    reason="JAX is an optional dependency")
def test_slurm_allows_jax(mcmc, monkeypatch):
    # JAX no longer needs to be disabled: parallel runs use threads instead of fork.
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    model = MCMC(mcmc.x, mcmc.y, mcmc.yerr, mcmc.alpha, mcmc.velocities,
                 mcmc.c_factor, deterministic_profile=True, use_jax=True)

    assert model.jax_enabled is True
    assert model([1.0, 0.0]) == pytest.approx(mcmc([1.0, 0.0]))


@pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                    reason="JAX is an optional dependency")
@pytest.mark.parametrize("deterministic_profile, od, continuum_method", [
    (True, True, "polyval"),
    (True, True, "chebval"),
    (True, False, "polyval"),
    (False, True, "polyval"),
    (False, False, "polyval"),
])
def test_jax_probabilities_match_numpy(mcmc, deterministic_profile, od,
                                       continuum_method):
    # Cover every static branch specialized by the JIT kernel.
    kwargs = dict(
        x_or_data=mcmc.x,
        y=mcmc.y,
        yerr=mcmc.yerr,
        alpha=mcmc.alpha,
        velocities=mcmc.velocities,
        c_factor=mcmc.c_factor if deterministic_profile else None,
        deterministic_profile=deterministic_profile,
        od=od,
        continuum_method=continuum_method,
    )
    numpy_model = MCMC(**kwargs)
    jax_model = MCMC(**kwargs, use_jax=True)

    if deterministic_profile:
        theta = np.array([1.0, 0.0]) if continuum_method == "polyval" else np.zeros(2)
    else:
        profile = np.zeros(mcmc.k_max) if od else np.ones(mcmc.k_max)
        theta = np.r_[profile, [1.0, 0.0]]

    assert jax_model.jax_enabled is True
    assert jax_model(theta) == pytest.approx(numpy_model(theta), rel=1e-10, abs=1e-8)
    assert jax_model.dynesty_logprob(theta) == pytest.approx(
        numpy_model.dynesty_logprob(theta), rel=1e-10, abs=1e-8
    )

    if deterministic_profile and od and continuum_method == "polyval":
        for invalid in ([-1.0, 0.0], [0.0, 0.0], [np.nan, 0.0]):
            assert jax_model(invalid) == -np.inf
            assert jax_model.dynesty_logprob(invalid) == -np.inf
            np.testing.assert_allclose(jax_model.log_probability_batch([theta, invalid]),
                                       [numpy_model(theta), -np.inf], rtol=1e-10, atol=1e-8)

    # emcee sends the full ensemble and differently sized subsets of walkers.
    walkers = np.array([theta, theta + 0.001, theta - 0.001])
    if deterministic_profile and od and continuum_method == "polyval":
        walkers[-1] = [-1.0, 0.0]
    for batch in (walkers, walkers[:1], walkers[1:]):
        np.testing.assert_allclose(jax_model.log_probability_batch(batch),
                                   numpy_model.log_probability_batch(batch), rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("use_jax", [False, True])
def test_vectorized_sampler_runs_and_continues(harps_order_40, use_jax, monkeypatch):
    if use_jax:
        pytest.importorskip("jax")
    from ACID_code import acid as acid_module
    monkeypatch.setattr(acid_module, "ThreadPool", lambda **kw: pytest.fail("Unexpected pool"))
    monkeypatch.setattr(acid_module.mp, "get_context", lambda *a: pytest.fail("Unexpected pool"))
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist, verbose=0, seed=0, check_interval=5)
    acid.ACID(wavelengths, flux, errors, sn, nsteps=20, use_jax=use_jax, vectorize=True, parallel=True)
    acid.continue_sampling(nsteps=5)
    acid.run_mcmc_until_converged(10)
    assert acid.sampler.vectorize is True
    assert acid.sampler.pool is None
    assert acid.sampler.get_chain().shape[0] == 35
    assert np.all(np.isfinite(acid.sampler.get_log_prob()))


@pytest.mark.skipif(importlib.util.find_spec("jax") is None,
                    reason="JAX is an optional dependency")
@pytest.mark.parametrize("warm_jax, use_jax", [(False, False), (True, True), (True, False)])
def test_parallel_sampling_uses_correct_workers(harps_order_40, tmp_path, warm_jax, use_jax):
    # A subprocess timeout turns a worker deadlock into a bounded test failure.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    inputs = tmp_path / "inputs.npz"
    np.savez(inputs, wavelengths=wavelengths, flux=flux, errors=errors, sn=sn,
             velocities=velocities, linelist=linelist)
    script = """
import os, sys
import numpy as np
from ACID_code import Acid, MCMC

inputs = dict(np.load(sys.argv[1]))
inputs['linelist'] = inputs['linelist'].item()
acid = Acid(verbose=0, poly_ord=2, seed=0, cores=2, check_interval=10)
acid.ACID(**inputs, use_jax=sys.argv[3] == 'True', parallel=True, run_mcmc=False)
assert MCMC(acid.data).jax_enabled == (sys.argv[3] == 'True')
acid.run_mcmc(2, state=acid.data.initial_state)
acid.config.use_jax = sys.argv[2] == 'True'
acid.run_mcmc(3)
assert acid.sampler.get_chain().shape[0] == 5
state = acid.data.initial_state
for method in (acid.run_mcmc, acid.run_mcmc_until_converged):
    acid.sampler = None
    method(20, state=state)
    assert acid.sampler.get_chain().shape == (20, acid.data.nwalkers, acid.data.ndim)
    assert np.all(np.isfinite(acid.sampler.get_log_prob()))

# Check both sampler callbacks through the real pool against serial NumPy.
acid.config.use_jax = False
reference = MCMC(acid.data)
acid.config.use_jax = sys.argv[2] == 'True'
for sampler in ('emcee', 'dynesty'):
    acid.config.sampler_type = sampler
    pool, log_prob, ptform = acid._get_sampler_pool()
    expected = reference if sampler == 'emcee' else reference.dynesty_logprob
    with pool:
        assert (pool.apply(os.getpid) == os.getpid()) == acid.config.use_jax
        np.testing.assert_allclose(pool.map(log_prob, state), [expected(t) for t in state], rtol=1e-10)
        units = np.full((2, acid.data.ndim), 0.5)
        np.testing.assert_allclose(pool.map(ptform, units), [reference.ptform(t) for t in units])
"""
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script, str(inputs), str(use_jax), str(warm_jax)],
                            env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.long
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7])
def test_harps_sampler_stays_bounded_across_seeds(harps_order_40, seed):
    # These seeds previously overflowed or sent the polynomial coefficients towards infinity.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist, seed=seed)
    acid.ACID(wavelengths, flux, errors, sn, poly_ord=4, run_mcmc=False, parallel=False)
    acid.run_mcmc_until_converged(1000, state=acid.data.initial_state)

    chain = acid.sampler.get_chain()
    assert np.all(np.isfinite(chain))
    assert np.max(np.abs(chain)) < 10
    assert walkers_independent(chain[-1])


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


# def test_multiprocessing_wrappers_delegate_to_the_worker_model(harps_result):
#     # Initialise the module-level worker model exactly as multiprocessing does.
#     mcmc_module._mp_init_worker(harps_result.data)
#     model = mcmc_module._MCMC
#     theta = np.asarray(model.model_inputs[model.k_max:])
#     unit_cube = np.full(len(theta), 0.5)

#     assert mcmc_module._mp_log_probability(theta) == pytest.approx(expected_model(theta))
#     assert mcmc_module._mp_log_likelihood(theta) == pytest.approx(expected_model.dynesty_logprob(theta))
#     np.testing.assert_allclose(mcmc_module._mp_ptform(unit_cube), expected_model.ptform(unit_cube))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
