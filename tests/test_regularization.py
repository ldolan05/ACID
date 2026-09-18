"""First-difference LSD regularization and its sampler integration."""
import importlib.util
import numpy as np
import pytest
from ACID_code import Acid, Config, LSD, MCMC


@pytest.mark.parametrize('scale_regularization', [True, False])
def test_regularization_matches_augmented_least_squares(scale_regularization):
    rng = np.random.default_rng(7)
    alpha = rng.uniform(size=(40, 8))
    errors = rng.uniform(0.05, 0.1, 40)
    y = alpha @ np.tile([0.1, 0.3, 0.2, 0.1], 2) + rng.normal(0, errors)
    D = np.diff(np.eye(alpha.shape[1]), axis=0)
    strength = 0.5
    information = np.diag(alpha.T @ np.diag(errors**-2) @ alpha)
    scale = information.mean() if scale_regularization else 1.0
    augmented = np.vstack((alpha / errors[:, None],
                           np.sqrt(strength * scale) * D))
    expected = np.linalg.lstsq(augmented, np.r_[y / errors, np.zeros(len(D))], rcond=None)[0]
    factor = LSD.calc_cholesky(alpha, errors, strength, scale_regularization=scale_regularization)
    profile, _, cov = LSD.solve_z(alpha, y, errors, factor, return_cov=True)
    np.testing.assert_allclose(profile, expected, atol=1e-12)
    np.testing.assert_allclose(cov, np.linalg.inv(augmented.T @ augmented), atol=1e-12)
    # A constant profile is in the penalty's nullspace.
    constants = np.full(alpha.shape[1], 0.2)
    recovered = LSD.solve_z(alpha, alpha @ constants, errors, factor, return_error=False)
    np.testing.assert_allclose(recovered, constants, atol=1e-12)


@pytest.mark.parametrize('change', ['errors', 'line_count', 'line_depth', 'flux'])
def test_relative_regularization_is_invariant_to_input_scale(change):
    rng = np.random.default_rng(17)
    alpha = rng.uniform(size=(40, 8))
    errors = rng.uniform(0.01, 0.03, 40)
    y = alpha @ rng.uniform(size=8)

    def solve(a, f, e, strength=0.5):
        return LSD.solve_z(a, f, e, LSD.calc_cholesky(a, e, regularization=strength),
                           return_error=False)

    baseline = solve(alpha, y, errors)
    assert np.linalg.norm(np.diff(baseline)) < np.linalg.norm(np.diff(solve(alpha, y, errors, 0)))
    if change == 'errors':
        actual = solve(alpha, y, 20 * errors)
    elif change == 'line_count':
        actual = solve(np.tile(alpha, (5, 1)), np.tile(y, 5), np.tile(errors, 5))
    elif change == 'line_depth':
        actual = 3 * solve(3 * alpha, y, errors)
    else:
        actual = solve(alpha, 10 * y, 10 * errors) / 10
    np.testing.assert_allclose(actual, baseline, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('od', [True, False])
def test_standalone_regularization_can_be_disabled(synthetic_spectrum, od):
    wave, flux, errors, sn, v, lines = synthetic_spectrum
    lsd = LSD(od=od, verbose=0)
    kwargs = dict(linelist=lines, velocities=v)
    lsd.run_LSD(wave, flux, errors, sn, **kwargs)
    baseline = lsd.profile.copy()
    lsd.run_LSD(wave, flux, errors, sn, regularization=0.5, **kwargs)
    assert np.linalg.norm(np.diff(lsd.profile)) < np.linalg.norm(np.diff(baseline))
    lsd.run_LSD(wave, flux, errors, sn, regularization=0, **kwargs)
    np.testing.assert_array_equal(lsd.profile, baseline)


@pytest.mark.parametrize('strength', [-1.0, np.nan, np.inf])
def test_invalid_regularization(strength):
    with pytest.raises(ValueError, match='finite and non-negative'):
        Config(regularization=strength)
    config = Config()
    with pytest.raises(ValueError, match='finite and non-negative'):
        config.regularization = strength
    with pytest.raises(ValueError, match='finite and non-negative'):
        LSD.calc_cholesky(np.eye(3), np.ones(3), regularization=strength)


@pytest.mark.parametrize('scale_regularization', [True, False])
def test_config_and_call_local_regularization(synthetic_spectrum, scale_regularization):
    assert Config().regularization == 0.0
    assert Config().scale_regularization is True
    config = Config(regularization=0.5, scale_regularization=scale_regularization)
    assert config.regularization == 0.5
    restored = Config(**config.to_dict())
    assert restored.regularization == 0.5
    assert restored.scale_regularization is scale_regularization
    restored.regularization = 0
    assert restored.regularization == 0.0

    # Omitted arguments inherit the config; explicit zero disables it for one call.
    wave, flux, errors, sn, v, lines = synthetic_spectrum
    lsd = LSD(verbose=0)
    lsd.data.config = lsd.config = config
    lsd.run_LSD(wave, flux, errors, sn, velocities=v, linelist=lines)
    factor = LSD.calc_cholesky(lsd.alpha_flat_masked, errors / flux, 0.5,
                               scale_regularization=scale_regularization)
    expected = LSD.solve_z(lsd.alpha_flat_masked, -np.log(flux), errors / flux, factor, return_error=False)
    np.testing.assert_allclose(lsd.profile, expected)
    lsd.run_LSD(wave, flux, errors, sn, velocities=v, linelist=lines,
                scale_regularization=not scale_regularization)
    factor = LSD.calc_cholesky(lsd.alpha_flat_masked, errors / flux, 0.5,
                               scale_regularization=not scale_regularization)
    expected = LSD.solve_z(lsd.alpha_flat_masked, -np.log(flux), errors / flux, factor, return_error=False)
    np.testing.assert_allclose(lsd.profile, expected)
    assert config.scale_regularization is scale_regularization
    lsd.run_LSD(wave, flux, errors, sn, velocities=v, linelist=lines, regularization=0)
    factor = LSD.calc_cholesky(lsd.alpha_flat_masked, errors / flux)
    expected = LSD.solve_z(lsd.alpha_flat_masked, -np.log(flux), errors / flux, factor, return_error=False)
    np.testing.assert_array_equal(lsd.profile, expected)
    assert config.regularization == 0.5


@pytest.mark.parametrize('scale_regularization', [True, False])
def test_acid_regularization_matches_augmented_least_squares(harps_order_40, scale_regularization):
    wave, flux, errors, sn, v, lines = harps_order_40
    acid = Acid(velocities=v, linelist=lines, verbose=0, parallel=False)
    acid.ACID(wave, flux, errors, sn, run_mcmc=False, regularization=0.5,
              scale_regularization=scale_regularization)
    assert acid.config.scale_regularization is scale_regularization
    model = MCMC(acid.data)
    _, z = model.deterministic_model(acid.data.poly_coeffs['masked'])
    weighted_alpha = model.alpha / (model.yerr / model.y)[:, None]
    scale = np.trace(weighted_alpha.T @ weighted_alpha) / len(v) if scale_regularization else 1.0
    expected = np.linalg.lstsq(
        np.vstack((weighted_alpha,
                   np.sqrt(0.5 * scale) * np.diff(np.eye(len(v)), axis=0))),
        np.r_[-np.log(acid.data.fitted_flux['masked'][~acid.data.full_mask]) / (model.yerr / model.y),
              np.zeros(len(v)-1)], rcond=None)[0]
    np.testing.assert_allclose(z, expected, atol=1e-10)


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
