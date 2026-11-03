"""Plots can inspect stored stages before MCMC or result processing."""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

from ACID_code import Acid, Data, LSD, Plot, Result, utils
from ACID_code.diagnostics.errors import ACIDResultError, ACIDStateError
from ACID_code.diagnostics.warnings import ACIDStateWarning


@pytest.mark.parametrize("constructor", [Plot, Result])
@pytest.mark.parametrize("source_type", [Data, Acid, Result])
def test_constructors_accept_supported_data_sources(harps_result, constructor, source_type):
    data = harps_result.data
    source = data if source_type is Data else source_type(data)
    assert constructor(source).data is data


@pytest.mark.parametrize("constructor", [Plot, Result, LSD])
@pytest.mark.parametrize("source", [object(), {}, "data", 1.5])
def test_constructors_reject_unsupported_data_sources(constructor, source):
    with pytest.raises(BeartypeCallHintParamViolation):
        constructor(source)


@pytest.fixture(scope="module")
def initial_plots(harps_order_40):
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist, seed=1, verbose=0)
    # Construct the plotter before any stages exist; it must see subsequent updates.
    plots = Plot(acid.data)
    acid.ACID(wavelengths, flux, errors, sn, run_mcmc=False)
    assert plots.data is acid.data
    assert Plot(acid).data is acid.data
    assert not plots.data.complete
    assert plots.data.sampler is None
    return plots


@pytest.mark.parametrize("key", ["initial", "masked"])
def test_plots_use_selected_stage_before_sampling(initial_plots, key):
    data = initial_plots.data
    figure, axis = initial_plots.profiles(key=key, return_fig=True)
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), data.velocities)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), data.profile[key][0])
    plt.close(figure)

    figure, axes = initial_plots.forward_model(
        key=key, normalized=True, show_masking=False, show_continuum=False,
        show_linelist=True, return_fig=True,
    )
    np.testing.assert_array_equal(axes[0].lines[0].get_xdata(), utils.drop_edges(data.forward_x[key]))
    np.testing.assert_allclose(axes[0].lines[0].get_ydata(), utils.drop_edges(data.flux[key] / data.continuum[key]))
    np.testing.assert_allclose(axes[0].lines[1].get_ydata(), utils.drop_edges(data.forward_y[key] / data.continuum[key]))
    plt.close(figure)
    assert not data.complete
    assert data.sampler is None


def test_data_plot_access_and_legacy_continuum_plots(initial_plots):
    data = initial_plots.data
    assert data.plot.data is data
    for plot in (data.plot.continuum_fit, data.plot_continuum_fit):
        figure, axis = plot("initial", return_fig=True)
        np.testing.assert_array_equal(axis.lines[1].get_ydata(), data.continuum["initial"])
        plt.close(figure)


def test_result_legacy_plots_allow_intermediate_keys_without_sampler(initial_plots):
    data = initial_plots.data
    with pytest.warns(ACIDStateWarning, match="Results not processed"):
        result = Result(data, process_results=False)
    figure, _ = result.plot_forward_model("initial", return_fig=True)
    plt.close(figure)
    # Preserve the old positional grid argument while allowing a stage keyword.
    figure, axis = result.plot_profiles(False, key="masked", return_fig=True)
    np.testing.assert_array_equal(axis.lines[0].get_ydata(), data.profile["masked"][0])
    plt.close(figure)
    assert not data.complete
    with pytest.raises(ACIDResultError, match="no sampler"):
        result.plot_forward_model(return_fig=True)


@pytest.mark.parametrize("method", ["profiles", "forward_model"])
def test_plot_reports_unavailable_stage_without_processing(initial_plots, method):
    with pytest.raises(ACIDStateError, match="key='final'"):
        getattr(initial_plots, method)(return_fig=True)
    with pytest.raises(ACIDStateError, match="key='unknown'"):
        getattr(initial_plots, method)(key="unknown", return_fig=True)


def test_plot_sampler_diagnostics_can_be_used_directly(harps_result):
    plots = Plot(harps_result.data)
    figure, _ = plots.walkers(return_fig=True)
    assert figure.axes
    plt.close(figure)
    figure, _ = plots.acf(max_lag=5, return_fig=True)
    assert figure.axes
    plt.close(figure)
    with pytest.raises(ACIDStateError, match="without a sampler"):
        Plot(Data()).walkers(return_fig=True)


@pytest.mark.parametrize("method", ["plot_profiles", "plot_forward_model"])
def test_result_final_plots_still_process_results_lazily(harps_result, method):
    data = Data().from_dict(deepcopy(harps_result.data.to_dict()))
    data.sampler = harps_result.sampler
    data.complete = False
    with pytest.warns(ACIDStateWarning, match="Results not processed"):
        result = Result(data, process_results=False)
    with pytest.warns(ACIDStateWarning, match="process_results"):
        figure, _ = getattr(result, method)(return_fig=True)
    assert data.complete
    plt.close(figure)
