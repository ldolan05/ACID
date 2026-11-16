"""Batched emcee likelihoods must preserve scalar probabilities and continuation."""
import copy
import importlib.util
import numpy as np
import pytest

from ACID_code import Acid, MCMC
from ACID_code.mcmc import jax as jax_module
from ACID_code.mcmc.emcee import EmceeBackend


@pytest.fixture
def prepared_acid(harps_order_40):
    wave, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist, verbose=0, seed=0,
                parallel=False, use_jax=False, check_interval=7, min_checks=100)
    acid.ACID(wave, flux, errors, sn, run_mcmc=False)
    return acid


@pytest.mark.skipif(importlib.util.find_spec('jax') is None, reason='JAX is optional')
@pytest.mark.parametrize('use_jax, _use_jax_vmap, expected', [
    (False, None, False), (False, True, False),
    (True, False, False), (True, None, False), (True, True, True),
])
def test_vectorization_flag(prepared_acid, use_jax, _use_jax_vmap, expected):
    acid = prepared_acid
    acid.ACID(run_mcmc=False, use_jax=use_jax, _use_jax_vmap=_use_jax_vmap)
    acid.run_mcmc(3, acid.data.initial_state)
    assert acid.sampler.vectorize is expected


@pytest.mark.skipif(importlib.util.find_spec('jax') is None, reason='JAX is optional')
@pytest.mark.parametrize('deterministic', [True, False])
@pytest.mark.parametrize('continuum_method', ['polyval', 'chebval'])
def test_batched_likelihood_matches_numpy(prepared_acid, deterministic, continuum_method):
    acid = prepared_acid
    acid.config.update_hipri(deterministic_profile=deterministic, continuum_method=continuum_method,
                            regularization=0.5)
    reference = MCMC(acid.data)
    acid.config.update_hipri(use_jax=True, _use_jax_vmap=True, parallel=True, cores=2)
    context, log_prob, vectorize = EmceeBackend(acid.data)._get_emcee_pool()
    coefs = acid.data.poly_coeffs['masked']
    theta = coefs if deterministic else np.r_[acid.data.profile['masked'][0], coefs]
    walkers = np.tile(theta, (15, 1))
    walkers += np.random.default_rng(0).normal(0, 1e-3, walkers.shape)
    # Exercise the full ensemble and unequal sub-ensembles from red/blue moves.
    with context as pool:
        assert pool is None and vectorize is True
        for batch in (walkers, walkers[:7], walkers[7:]):
            actual = log_prob(batch)
            assert isinstance(actual, np.ndarray)
            np.testing.assert_allclose(actual, [reference(t) for t in batch], rtol=1e-10, atol=1e-8)
        if deterministic and continuum_method == 'polyval':
            walkers[0] = 0
            assert log_prob(walkers)[0] == -np.inf


@pytest.mark.skipif(importlib.util.find_spec('jax') is None, reason='JAX is optional')
@pytest.mark.parametrize('parallel', [True, False])
def test_vectorized_sampling_and_chunked_continuation_match_scalar(prepared_acid, parallel):
    scalar = prepared_acid
    scalar.config.update_hipri(use_jax=True, _use_jax_vmap=False, parallel=parallel, cores=2)
    batched = Acid(data=copy.deepcopy(scalar.data))
    batched.config._use_jax_vmap = True
    scalar.run_mcmc(30, scalar.data.initial_state)
    batched.run_mcmc_until_converged(17, batched.data.initial_state)
    batched.run_mcmc(13)
    assert batched.sampler.vectorize is True and batched.sampler.pool is None
    np.testing.assert_allclose(batched.sampler.get_chain(), scalar.sampler.get_chain(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(batched.sampler.get_log_prob(), scalar.sampler.get_log_prob(), rtol=1e-10, atol=1e-8)
    np.testing.assert_equal(batched.sampler.random_state, scalar.sampler.random_state)


def test_vectorization_falls_back_without_jax(prepared_acid, monkeypatch):
    acid = prepared_acid
    acid.config.update_hipri(use_jax=True, _use_jax_vmap=True)
    monkeypatch.setattr(jax_module, '_import_jax', lambda: None)
    acid.run_mcmc(3, acid.data.initial_state)
    assert acid.sampler.vectorize is False
    assert np.all(np.isfinite(acid.sampler.get_log_prob()))


@pytest.mark.skipif(importlib.util.find_spec('jax') is None, reason='JAX is optional')
def test_vmapped_walkers_ignore_parallel_setting(prepared_acid):
    chains = []
    for parallel in (False, True):
        acid = Acid(data=copy.deepcopy(prepared_acid.data))
        acid.config.update_hipri(use_jax=True, _use_jax_vmap=True, parallel=parallel, cores=2)
        acid.run_mcmc(20, acid.data.initial_state)
        assert acid.sampler.vectorize is True and acid.sampler.pool is None
        chains.append(acid.sampler.get_chain())
    np.testing.assert_array_equal(*chains)


@pytest.mark.skipif(importlib.util.find_spec('jax') is None, reason='JAX is optional')
@pytest.mark.parametrize('vectorize', [True, False])
def test_saved_run_can_switch_vectorization(prepared_acid, tmp_path, vectorize):
    acid = prepared_acid
    acid.config.update_hipri(use_jax=True, _use_jax_vmap=not vectorize, dir=str(tmp_path))
    acid.run_mcmc(17, acid.data.initial_state)
    prefix = acid.sampler.get_chain().copy()
    result = Acid(dir=str(tmp_path), verbose=0).ACID(resume=True, run_mcmc=True, nsteps=30,
                                                   _use_jax_vmap=vectorize)
    assert result.sampler.vectorize is vectorize
    assert result.sampler.get_chain().shape[0] == 30
    np.testing.assert_array_equal(result.sampler.get_chain()[:17], prefix)
    reference = MCMC(result.data)
    np.testing.assert_allclose(result.sampler.get_log_prob()[-1],
                               [reference(t) for t in result.sampler.get_chain()[-1]], rtol=1e-10)
