#%%
"""Tests for the public Result interface using one shared HARPS ACID fit."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ACID_code import Acid, Data, Result


def test_result_indexes_profiles_errors_covariance_and_frames(harps_result, harps_order_40):
    """Indexing must provide the documented views of the completed profile."""
    _, _, _, _, velocities, _ = harps_order_40

    # Integer and descriptive string access select the final profile components.
    np.testing.assert_array_equal(harps_result[0], harps_result["profile"])
    np.testing.assert_array_equal(harps_result[1], harps_result["error"])
    np.testing.assert_array_equal(harps_result[2], harps_result["covariance"])

    # Two indices select a frame and component; the legacy three-index form ignores order.
    np.testing.assert_array_equal(harps_result[0, 0], harps_result["profile"])
    np.testing.assert_array_equal(harps_result[0, 1], harps_result["error"])
    np.testing.assert_array_equal(harps_result[99, 0, 0], harps_result["profile"])
    assert len(list(harps_result)) == 1
    assert "profiles=available" in repr(harps_result)
    assert harps_result["profile"].shape == velocities.shape

    # Invalid selectors should fail at the Result boundary with a useful error.
    with pytest.raises(ValueError, match="0, 1, or 2"):
        _ = harps_result[3]
    with pytest.raises(ValueError, match="String index"):
        _ = harps_result["continuum"]


def test_result_save_load_and_object_constructors(tmp_path, harps_result):
    """Result persistence must preserve both Data and the sampler backend."""
    data_path = tmp_path / "result.pkl"
    sampler_path = tmp_path / "sampler.h5"

    # Save the shared fit once, then exercise every supported loading route.
    harps_result.save(str(data_path), str(sampler_path))
    from_path = Result.load(str(data_path))
    from_data = Result.load(from_path.data)
    from_result = Result.load(from_path)

    assert data_path.exists()
    assert sampler_path.exists()
    assert from_path.sampler.get_chain().shape == harps_result.sampler.get_chain().shape
    assert from_data.data is from_path.data
    assert from_result.data is from_path.data


def test_result_continues_and_can_process_lazily(tmp_path, harps_result):
    """Continuation should add steps, while lazy processing defers only final LSD work."""
    data_path = tmp_path / "result.pkl"
    sampler_path = tmp_path / "sampler.h5"
    harps_result.save(str(data_path), str(sampler_path))
    loaded = Result.load(str(data_path))
    before = loaded.sampler.get_chain().shape[0]

    # Skip the final profile refresh first, then request it through the public method.
    loaded.continue_sampling(nsteps=3, process_results=False)
    assert loaded.sampler.get_chain().shape[0] == before + 3
    loaded.process_results()
    assert loaded.data.complete


def test_result_plots_the_completed_fit_without_re_running_acid(harps_result):
    """Result plots should consume the stored sampler and final profile only."""
    # These figures cover posterior, profile, forward-model, and correlation diagnostics.
    figures = [harps_result.plot_corner(return_fig=True),
               harps_result.plot_profiles(return_fig=True)[0],
               harps_result.plot_forward_model(return_fig=True)[0],
               harps_result.plot_autocorrelation(return_fig=True, min_steps=3)[0],
               harps_result.plot_acf(return_fig=True, max_lag=5)[0]]
    for figure in figures:
        assert figure.axes
        plt.close(figure)


def test_result_plot_options_cover_normalised_unmasked_and_custom_axes(harps_result):
    # Supply an existing profile axis and custom labels to exercise the composable plot route.
    profile_figure, profile_axis = plt.subplots()
    returned_figure, returned_axis = harps_result.plot_profiles(
        labels={"title": "Stored profile"}, fig_ax=(profile_figure, profile_axis),
        return_fig=True,
    )
    assert returned_figure is profile_figure
    assert returned_axis.get_title() == "Stored profile"

    # The forward model supports normalised flux and can omit masking and continuum layers.
    forward_figure, forward_axes = harps_result.plot_forward_model(
        normalized=True, show_masking=False, show_continuum=False, return_fig=True,
    )
    assert len(forward_axes) == 2
    assert len(forward_axes[0].lines) >= 2
    plt.close(profile_figure)
    plt.close(forward_figure)


def test_result_continuum_error_supports_both_basis_functions(harps_result):
    # Use fixed coefficient draws so the expected error shape is independent of MCMC noise.
    result = harps_result
    norm_wavelengths = np.linspace(-1, 1, 20)
    coefficients = np.array([[1.0, 0.01], [1.1, -0.01], [0.9, 0.0]])
    original_method = result.data.config.continuum_method

    result.data.config.continuum_method = "polyval"
    polynomial_error = result._get_continuum_error(norm_wavelengths, coefficients)
    result.data.config.continuum_method = "chebval"
    chebyshev_error = result._get_continuum_error(norm_wavelengths, coefficients)
    result.data.config.continuum_method = original_method

    assert polynomial_error.shape == norm_wavelengths.shape
    assert chebyshev_error.shape == norm_wavelengths.shape
    assert np.all(polynomial_error >= 0)
    assert np.all(chebyshev_error >= 0)


def test_result_walkers_plot_uses_valid_matplotlib_arguments(harps_result):
    """Walker plotting is a separate regression check because it has its own plotting path."""
    # This should return a figure like the other Result plot methods.
    figure, _ = harps_result.plot_walkers(return_fig=True)
    assert figure.axes
    plt.close(figure)


def test_result_requires_a_sampler_when_data_is_not_complete():
    """A bare Data object cannot be processed into a Result without a sampler."""
    # This is a user-input validation path, so pytest's native exception assertion is appropriate.
    data = Data()
    data.config.verbose = 0
    with pytest.raises(ValueError, match="without a sampler"):
        Result(data)

    # Acid remains the supported route for attaching a sampler to incomplete Data.
    assert isinstance(Acid(data=data), Acid)


def test_result_sampler_guards_and_emcee_only_plot_validation(harps_result):
    # Trace plots are reserved for dynesty; emcee users are directed to walker plots.
    with pytest.raises(ValueError, match="only available for dynesty"):
        harps_result.plot_traceplot(return_fig=True)

    # Autocorrelation diagnostics enforce their minimum chain length explicitly.
    chain_length = harps_result.sampler.get_chain().shape[0]
    with pytest.raises(ValueError, match="Not enough"):
        harps_result.plot_autocorrelation(min_steps=chain_length + 1, return_fig=True)

    # A complete Data payload without its sampler can still expose profiles, but not sampler tools.
    data = Data().from_dict(harps_result.data.to_dict())
    data.sampler = None
    result = Result(data)
    with pytest.raises(AttributeError, match="without a sampler"):
        result.initiate_sampler(None, _method_name="test")


def test_result_load_rejects_unsupported_input_types():
    # The public loader documents paths, Data, and Result objects as its complete input set.
    with pytest.raises(BeartypeCallHintParamViolation):
        Result.load(123)


def test_result_plot_forward_model_shows_linelist(harps_result):
    """The forward model plot can optionally show the line list."""
    figure, axes1 = harps_result.plot_forward_model(show_linelist=False, return_fig=True)
    figure, axes2 = harps_result.plot_forward_model(show_linelist=True, return_fig=True)
    n_colls_1 = len(axes1[0].collections)
    n_colls_2 = len(axes2[0].collections)
    assert n_colls_2 == n_colls_1 + 1 # The extra collection is the linelist markers.
    plt.close(figure)


def test_result_plot_forward_model_returns_fig_and_axes(harps_result):
    """The forward model plot returns a figure and axes when requested."""
    figure, axes = harps_result.plot_forward_model(return_fig=True)
    assert isinstance(figure, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(figure)


def test_result_plot_forward_model_accepts_different_keys(harps_result):
    """The forward model plot accepts different keys for the data."""
    # The default is "final", so test something that is not already tested.
    figure, axes = harps_result.plot_forward_model(key="initial", return_fig=True)
    # We mainly test that no exceptions are raised and that the return types are correct.
    assert isinstance(figure, plt.Figure)
    assert isinstance(axes, np.ndarray)
    plt.close(figure)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
