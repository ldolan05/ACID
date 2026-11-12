"""Optional NUTS sampling must preserve ACID's posterior and result interface."""
import copy
import importlib.util
import pickle
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from emcee import EnsembleSampler

from ACID_code import Acid, Config, Data, MCMC, Result, Sampler
from ACID_code.diagnostics.errors import ACIDInputError, ACIDInitialStateError
from ACID_code.mcmc.pymc import PyMCBackend, _import_pymc


optional = pytest.mark.skipif(
    any(importlib.util.find_spec(name) is None for name in ("pymc", "numpyro", "jax")),
    reason="PyMC, NumPyro, and JAX are optional dependencies")


def prepare(spectrum, **kwargs):
    wave, flux, errors, sn, velocities, lines = spectrum
    options = dict(sampler_type="pymc", nwalkers=2, nsteps=24, seed=19, verbose=0,
                   parallel=False, poly_ord=1, masking_lines={}, n_bins=10,
                   sigma_lower=100., sigma_upper=100., dev_perc=100.,
                   pymc_kwargs={"tune":40, "compute_convergence_checks":False})
    options.update(kwargs)
    acid = Acid(velocities=velocities, linelist=lines, **options)
    acid.ACID(wave, flux, errors, sn, run_mcmc=False)
    return acid


def test_import_and_preparation_do_not_import_pymc(synthetic_spectrum):
    script = """
import sys
from ACID_code import Acid
Acid(sampler_type='pymc')
assert all(name not in sys.modules for name in ('pymc', 'numpyro', 'pytensor'))
"""
    process = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    acid = prepare(synthetic_spectrum, nwalkers=None)
    assert acid.data.nwalkers == 4
    assert acid.data.initial_state.shape == (acid.data.ndim,)


def test_missing_pymc_dependency_has_installation_hint(monkeypatch):
    _import_pymc.cache_clear()
    monkeypatch.setitem(sys.modules, "pymc", None)
    with pytest.raises(ImportError, match=r"ACID_code\[pymc\]"):
        _import_pymc()


def test_pymc_convergence_requires_multiple_chains(synthetic_spectrum):
    assert Config(sampler_type="pymc", max_steps=100).sampler_type == "pymc"
    acid = prepare(synthetic_spectrum, nwalkers=1)
    with pytest.raises(ACIDInputError, match="at least two"):
        acid.run_mcmc_until_converged(100)


@optional
@pytest.mark.parametrize("deterministic, od, parallel", [
    (True, True, False), (True, False, True), (False, True, True), (False, False, False),
])
def test_nuts_draws_use_shared_posterior_and_emcee_interface(
        synthetic_spectrum, deterministic, od, parallel):
    acid = prepare(synthetic_spectrum, deterministic_profile=deterministic, od=od,
                   parallel=parallel, continuum_method="chebval", regularization=0.2)
    result = acid.ACID(run_mcmc=True)
    sampler = result.sampler
    assert isinstance(sampler, EnsembleSampler)
    assert sampler.get_chain().shape == (24, 2, acid.data.ndim)
    assert np.any(np.ptp(sampler.get_chain(), axis=0) > 0)
    assert sampler.get_chain(discard=2, thin=2, flat=True).shape == (22, acid.data.ndim)
    assert sampler.get_log_prob().shape == (24, 2)
    assert sampler.get_autocorr_time(tol=0).shape == (acid.data.ndim,)
    assert sampler.get_last_sample().coords.shape == (2, acid.data.ndim)
    assert sampler.get_blobs()["diverging"].shape == (24, 2)
    np.testing.assert_allclose(sampler.acceptance_fraction,
                               sampler.get_blobs()["acceptance_rate"].mean(axis=0))
    reference = MCMC(acid.data)
    expected = np.array([[reference(theta) for theta in draw] for draw in sampler.get_chain()])
    np.testing.assert_allclose(sampler.get_log_prob(), expected, rtol=1e-10, atol=1e-8)
    assert result.burnin == 0
    assert result.thin == 1
    assert result.data.complete
    assert np.all(np.isfinite(result["profile"]))


@optional
def test_pymc_bridge_preserves_density_and_cholesky_gradient(synthetic_spectrum):
    import jax
    import jax.numpy as jnp
    from pymc.sampling.jax import get_jaxified_logp
    pm, _, LogProbability = _import_pymc()
    acid = prepare(synthetic_spectrum)
    probability = MCMC(acid.data, numpy_module=jnp, linalg=jax.scipy.linalg)
    theta0 = acid.data.initial_state
    with pm.Model() as model:
        theta = pm.Flat("theta", shape=acid.data.ndim)
        pm.Potential("log_probability", LogProbability(probability.log_probability)(theta))
    posterior = get_jaxified_logp(model)
    assert float(posterior([theta0])) == pytest.approx(probability(theta0))
    gradient = np.asarray(jax.grad(lambda t: posterior([t]))(jnp.asarray(theta0)))
    delta = np.eye(len(theta0)) * 1e-6
    numerical = np.array([(probability(theta0+d) - probability(theta0-d)) / 2e-6 for d in delta])
    np.testing.assert_allclose(gradient, numerical, rtol=1e-4, atol=1e-4)


@optional
def test_nuts_seed_is_private_and_repeatable(synthetic_spectrum):
    acid = prepare(synthetic_spectrum)
    other = Acid(data=copy.deepcopy(acid.data))
    before = pickle.dumps(np.random.get_state())
    acid.run_mcmc(12, acid.data.initial_state)
    assert pickle.dumps(np.random.get_state()) == before
    try:
        np.random.seed(821)
        np.random.normal(size=100)
        other.run_mcmc(12, other.data.initial_state)
    finally:
        np.random.set_state(pickle.loads(before))
    np.testing.assert_array_equal(acid.sampler.get_chain(), other.sampler.get_chain())
    np.testing.assert_array_equal(acid.sampler.get_log_prob(), other.sampler.get_log_prob())


@optional
def test_nuts_results_persist_plot_and_continue(synthetic_spectrum, tmp_path, monkeypatch):
    acid = prepare(synthetic_spectrum, dir=str(tmp_path))
    result = acid.ACID(run_mcmc=True)
    original_chain = result.sampler.get_chain().copy()
    original_stats = result.sampler.get_blobs().copy()
    # Reading converted results must not require any optional sampling imports.
    with monkeypatch.context() as patch:
        for name in ("pymc", "numpyro", "pytensor"):
            patch.setitem(sys.modules, name, None)
        loaded = Result.load(str(tmp_path / "data.pkl"))
        np.testing.assert_array_equal(loaded.sampler.get_chain(), original_chain)
        np.testing.assert_array_equal(loaded.sampler.get_blobs(), original_stats)
        assert loaded.burnin == 0 and loaded.thin == 1
        np.testing.assert_array_equal(loaded["profile"], result["profile"])
        for plot in (loaded.plot_walkers, loaded.plot_corner, loaded.plot_autocorrelation, loaded.plot_acf):
            kwargs = {"min_steps":3} if plot == loaded.plot_autocorrelation else {}
            figure = plot(return_fig=True, **kwargs)
            plt.close(figure[0] if isinstance(figure, tuple) else figure)
    loaded.continue_sampling(nsteps=3)
    assert loaded.data.nsteps == 27
    np.testing.assert_array_equal(loaded.sampler.get_chain()[:24], original_chain)
    np.testing.assert_array_equal(loaded.sampler.get_blobs()[:24], original_stats)


@optional
def test_pymc_validates_initial_state_and_managed_options(synthetic_spectrum):
    acid = prepare(synthetic_spectrum)
    with pytest.raises(ACIDInputError, match="shape"):
        acid.run_mcmc(10, np.zeros(1))
    with pytest.raises(ACIDInitialStateError, match="finite"):
        acid.run_mcmc(10, np.full(acid.data.ndim, np.nan))
    acid.config.pymc_kwargs = {"random_seed":2}
    with pytest.raises(ACIDInputError, match="cannot override"):
        acid.run_mcmc(10)


@optional
@pytest.mark.parametrize("parallel", [False, True])
def test_nuts_convergence_batches_preserve_state_and_limit(synthetic_spectrum, monkeypatch, parallel):
    acid = prepare(synthetic_spectrum, parallel=parallel, check_interval=7, max_steps=19)
    single = Acid(data=copy.deepcopy(acid.data))
    single.config.check_interval = 19
    def not_converged(chain, divergences):
        return dict(converged=False, rhat_max=2., ess_bulk_min=1., ess_tail_min=1., divergences=divergences)
    monkeypatch.setattr(PyMCBackend, "stopping_criterion", staticmethod(not_converged))
    acid.ACID(run_mcmc=True)
    single.ACID(run_mcmc=True)
    assert acid.data.nsteps == 19
    assert acid.data.sampler_diagnostics["checks"] == 3
    assert acid.result.converged is False
    np.testing.assert_array_equal(acid.sampler.get_chain(), single.sampler.get_chain())
    np.testing.assert_array_equal(acid.sampler.get_blobs(), single.sampler.get_blobs())


@optional
def test_nuts_stops_after_consecutive_checks_and_survives_reload(synthetic_spectrum, tmp_path, monkeypatch):
    acid = prepare(synthetic_spectrum, check_interval=10, min_checks=2, max_steps=80, dir=str(tmp_path))
    calls = []
    def converged(chain, divergences):
        calls.append(len(chain))
        return dict(converged=True, rhat_max=1., ess_bulk_min=1000., ess_tail_min=1000., divergences=0)
    monkeypatch.setattr(PyMCBackend, "stopping_criterion", staticmethod(converged))
    result = acid.ACID(run_mcmc=True)
    assert calls == [10, 20]
    assert result.data.nsteps == 20
    assert result.converged is True
    loaded = Result.load(str(tmp_path / "data.pkl"))
    assert loaded.converged is True
    loaded.continue_sampling(max_steps=35)
    assert loaded.data.nsteps == 35
    assert calls[-2:] == [30, 35]


@optional
@pytest.mark.parametrize("kind", ["good", "different", "constant", "diverging"])
def test_nuts_diagnostics_reject_bad_chains(kind):
    chain = np.random.default_rng(81).normal(size=(2000, 4, 2))
    if kind == "different":
        chain[:, 0] += 5
    if kind == "constant":
        chain[:, :, 0] = 0
    diagnostics = PyMCBackend.stopping_criterion(chain, int(kind == "diverging"))
    assert diagnostics["converged"] is (kind == "good")
