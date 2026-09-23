import importlib.util
import numpy as np
import pytest
from ACID_code import Acid, MCMC
from ACID_code.diagnostics.warnings import ACIDRuntimeWarning
from ACID_code import mcmc as mcmc_module
from ACID_code import jax as jax_module


@pytest.mark.skipif(importlib.util.find_spec('jax') is None, reason='JAX is optional')
@pytest.mark.parametrize('deterministic', [True, False])
@pytest.mark.parametrize('scale_regularization', [True, False])
def test_regularization_prior_and_jax_agree(harps_order_40, deterministic, scale_regularization):
    wave, flux, errors, sn, v, lines = harps_order_40
    acid = Acid(velocities=v, linelist=lines, verbose=0, parallel=False,
                regularization=0.5, deterministic_profile=deterministic,
                scale_regularization=scale_regularization)
    acid.ACID(wave, flux, errors, sn, run_mcmc=False)
    model = MCMC(acid.data)
    acid.config.use_jax = True
    jax_model = MCMC(acid.data)
    z = 0.2 + 0.05 * (-1.0)**np.arange(len(v))
    weighted_alpha = model.alpha / (model.yerr / model.y)[:, None]
    scale = np.trace(weighted_alpha.T @ weighted_alpha) / len(v) if scale_regularization else 1.0
    assert model.regularization_prior(z) == pytest.approx(-0.5 * 0.5 * scale * np.sum(np.diff(z)**2))
    coefs = acid.data.poly_coeffs['masked']
    theta = coefs if deterministic else np.r_[z, coefs]
    assert jax_model(theta) == pytest.approx(model(theta), rel=1e-10)
    assert jax_model.dynesty_logprob(theta) == pytest.approx(model.dynesty_logprob(theta), rel=1e-10)


def test_requested_jax_falls_back_when_it_cannot_be_imported(mcmc, monkeypatch):
    # A missing optional accelerator must never make the ordinary MCMC path unusable.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(jax_module, "_import_jax", lambda: None)
    with pytest.warns(ACIDRuntimeWarning, match="falling back"):
        model = MCMC(mcmc.x, mcmc.y, mcmc.yerr, mcmc.alpha, mcmc.velocities,
                     mcmc.c_factor, deterministic_profile=True, use_jax=True)

    assert model.use_jax is True
    assert model.jax_enabled is False
    assert model([1.0, 0.0]) == pytest.approx(mcmc([1.0, 0.0]))


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
@pytest.mark.parametrize("deterministic_profile", [True, False])
@pytest.mark.parametrize("od", [True, False])
@pytest.mark.parametrize("continuum_method", ["polyval", "chebval"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", [0, 1, 3, 8])
def test_jax_probabilities_match_numpy(mcmc, deterministic_profile, od,
                                       continuum_method, dtype, degree):
    # Cover every static branch specialized by the JIT kernel.
    factor = mcmc.c_factor
    if degree % 2 == 0:
        factor = (factor[0].T, not factor[1]) # also cover lower-triangular factors
    kwargs = dict(
        x_or_data=mcmc.x.astype(dtype),
        y=mcmc.y.astype(dtype),
        yerr=mcmc.yerr.astype(dtype),
        alpha=mcmc.alpha,
        velocities=mcmc.velocities,
        c_factor=factor if deterministic_profile else None,
        deterministic_profile=deterministic_profile,
        od=od,
        continuum_method=continuum_method,
    )
    numpy_model = MCMC(**kwargs)
    jax_model = MCMC(**kwargs, use_jax=True)

    assert jax_model.jax_enabled is True
    rng = np.random.default_rng(42)
    for _ in range(10):
        coefs = rng.normal(0, 0.01, degree + 1)
        if continuum_method == "polyval":
            coefs[0] += 1
        profile = rng.uniform(-1.2, 2.0, mcmc.k_max) + (0 if od else 1)
        theta = coefs if deterministic_profile else np.r_[profile, coefs]
        np.testing.assert_allclose(jax_model._jax_backend._eval_continuum(coefs),
                                   mcmc_module.utils.eval_continuum(numpy_model.x, coefs, method=continuum_method),
                                   rtol=1e-13, atol=1e-14)
        assert jax_model(theta) == pytest.approx(numpy_model(theta), rel=1e-10, abs=1e-8)
        assert jax_model.dynesty_logprob(theta) == pytest.approx(
            numpy_model.dynesty_logprob(theta), rel=1e-10, abs=1e-8
        )

    if deterministic_profile and od and continuum_method == "polyval":
        for value in (-1.0, 0.0, np.nan):
            theta = np.zeros(degree + 1)
            theta[0] = value
            assert jax_model(theta) == -np.inf
            assert jax_model.dynesty_logprob(theta) == -np.inf

