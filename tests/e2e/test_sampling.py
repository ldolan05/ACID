#%%
import numpy as np
import importlib.util
import os
import subprocess
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code import Acid
from emcee.ensemble import walkers_independent


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



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
