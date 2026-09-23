"""First-difference LSD regularization and its sampler integration."""
import numpy as np
import pytest
from ACID_code import Acid, Config, LSD, MCMC


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

