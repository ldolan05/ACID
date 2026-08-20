#%%
import matplotlib.pyplot as plt
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import Data, Profiles


@pytest.fixture
def gaussian_profile():
    # A known Gaussian profile makes the centre and depth fit assertions interpretable.
    velocities = np.linspace(-12, 12, 61)
    flux = Profiles.gaussian_func(velocities, -0.25, 1.5, 2.2)
    return velocities, flux, np.full_like(flux, 0.01)


@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt"])
def test_profile_fit_recovers_absorption_centre(gaussian_profile, model):
    # Every supported analytic model should identify the same absorption minimum.
    velocities, flux, errors = gaussian_profile
    profiles = Profiles(velocities, flux, errors)
    parameters, covariance = getattr(profiles, f"fit_{model}")()

    assert parameters[0] < 0
    assert parameters[1] == pytest.approx(1.5, abs=0.2)
    assert covariance.shape[0] == len(parameters)


def test_profile_fit_drops_nan_values_and_plots(gaussian_profile):
    # Invalid points are excluded before fitting, then the fit and residual axes are plotted.
    velocities, flux, errors = gaussian_profile
    flux[0] = np.nan
    profiles = Profiles(velocities, flux, errors)
    fig, axes = profiles.plot_fit("gaussian", return_fig=True)

    assert len(profiles.flux) == len(flux) - 1
    assert len(axes) == 2
    plt.close(fig)


def test_profile_validation_errors():
    # Missing data and unknown model choices are user-input errors, not fit failures.
    with pytest.raises(ValueError, match="velocities and flux"):
        Profiles()
    with pytest.raises(ValueError, match="Model must"):
        Profiles(np.array([0, 1, 2]), np.array([1, 0.9, 1])).plot_fit("invalid")


def test_profile_accepts_documented_list_inputs():
    # Array-like inputs are public API inputs and should be normalised before boolean indexing.
    profiles = Profiles([0, 1, 2], [1, 0.9, 1])

    np.testing.assert_array_equal(profiles.velocities, [0, 1, 2])
    np.testing.assert_array_equal(profiles.flux, [1, 0.9, 1])


def test_profile_functions_have_expected_centres_and_wings():
    # Each normalised analytic model reaches 1 + amplitude + offset at its centre.
    x = np.array([-10.0, 2.0, 10.0])
    functions = [
        Profiles.gaussian_func(x, -0.3, 2.0, 1.5, 0.1),
        Profiles.lorentzian_func(x, -0.3, 2.0, 1.5, 0.1),
        Profiles.voigt_func(x, -0.3, 2.0, 1.5, 0.2, 0.1),
    ]

    for profile in functions:
        assert profile[1] == pytest.approx(0.8)
        assert profile[0] > profile[1]
        assert profile[2] > profile[1]


def test_nan_removal_keeps_errors_and_covariance_aligned(gaussian_profile):
    # Mark different rows invalid through flux and errors, then check covariance drops both axes.
    velocities, flux, errors = gaussian_profile
    flux[2] = np.nan
    errors[4] = 0
    covariance = np.diag(np.full(len(flux), 0.01 ** 2))
    profiles = Profiles(velocities, flux, errors, covariance)

    assert len(profiles.velocities) == len(velocities) - 2
    assert profiles.flux_err.shape == profiles.flux.shape
    assert profiles.cov_matrix.shape == (len(profiles.flux), len(profiles.flux))


def test_copy_inputs_does_not_alias_profile_arrays(gaussian_profile):
    # Internal fitting helpers should work on copies so callers' profile arrays remain unchanged.
    velocities, flux, errors = gaussian_profile
    profiles = Profiles(velocities, flux, errors)
    copied = profiles._copy_inputs(None, None, None, None)
    copied[0][0] = 999
    copied[1][0] = 999
    copied[2][0] = 999

    assert profiles.velocities[0] != 999
    assert profiles.flux[0] != 999
    assert profiles.flux_err[0] != 999


def test_explicit_fit_inputs_and_covariance_are_used(gaussian_profile):
    # Fit a subset through the explicit method arguments and provide a full covariance matrix.
    velocities, flux, errors = gaussian_profile
    profiles = Profiles(velocities, flux, errors)
    subset = slice(5, -5)
    covariance = np.diag(errors[subset] ** 2)
    parameters, parameter_covariance = profiles.fit_gaussian(
        x=velocities[subset], y=flux[subset], yerr=errors[subset], cov_matrix=covariance,
        p0=[-0.25, 1.5, 2.2, 0.0],
    )

    assert parameters[1] == pytest.approx(1.5, abs=0.1)
    assert parameter_covariance.shape == (4, 4)
    assert "gaussian" in profiles.fitted_y
    assert "gaussian" in profiles.fitted_yerr


def test_plot_none_reuses_existing_fits(gaussian_profile):
    # A None model should plot existing fits without fitting an additional model.
    velocities, flux, errors = gaussian_profile
    profiles = Profiles(velocities, flux, errors)
    profiles.fit_gaussian()
    figure, axes = profiles.plot_fit(None, return_fig=True)

    assert set(profiles.fitted_y) == {"gaussian"}
    assert len(axes[0].lines) >= 2
    plt.close(figure)


def test_profiles_can_initialise_from_completed_data(harps_result):
    # The Data shortcut should select the first frame profile, errors, and covariance.
    profiles = Profiles(data=harps_result.data)

    np.testing.assert_array_equal(profiles.velocities, harps_result.data.velocities)
    np.testing.assert_array_equal(profiles.flux, harps_result.data.profiles[0][0])

    # Incomplete Data lacks the required final profile products.
    with pytest.raises(ValueError, match="running ACID"):
        Profiles(data=Data())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
