from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from contextlib import nullcontext
from emcee import EnsembleSampler
from ACID_code import Acid, Config
from ACID_code.diagnostics.errors import ACIDInputError
from ACID_code.mcmc.emcee import EmceeBackend


@pytest.mark.parametrize("tau_list, steps, expected", [
    ([np.array([2.0])], 200, False),
    ([np.array([2.0]), np.array([2.01])], 200, True),
    ([np.array([np.nan]), np.array([2.0])], 200, False),
])
def test_stopping_criterion(mcmc, tau_list, steps, expected):
    # Feed fixed autocorrelation histories to isolate stopping logic from sampling noise.
    converged, _, _ = EmceeBackend.stopping_criterion(tau_list, steps, 1, 50, 0.1)
    assert converged is expected



@pytest.mark.parametrize("min_tau_factor, tau_tol, steps, expected", [
    (0, 0.1, 40, True),
    (0, 0.001, 40, False),
    (0, 0.1, 10, True),
    (5, 0, 11, True),
    (5, 0, 10, False),
    (50, 0, 200, True),
    (50, 0, 100, False),
])
def test_stopping_criterion_with_disabled_threshold(min_tau_factor, tau_tol, steps, expected):
    tau_list = [np.array([3.0]), np.array([2.0])]
    if tau_tol != 0:
        tau_list = [np.array([2.0]), np.array([2.01])]
    converged, _, _ = EmceeBackend.stopping_criterion(tau_list, steps, 1, min_tau_factor, tau_tol)
    assert converged is expected



@pytest.mark.parametrize("tau_list, steps, min_tau_factor, tau_tol, expected", [
    ([0.8, 0.81], 100, 0, 0.1, True),
    ([0.99, 1.0], 100, 0, 0.1, True),
    ([1000.0, 1010.0], 5000, 0, 0.1, True),
    ([2.0, 2.01], 200, 0, np.float64(0.1), True),
    ([np.nan, 2.0], 200, 50, 0, True),
    ([1.0, 0.0], 100, 0, 0.1, False),
    ([-1.0, -1.0], 100, 0, 0.1, False),
    ([2.0, 2.01], 40, 50, 0.1, False),
    ([3.0, 2.0], 200, 50, 0.1, False),
])
def test_stopping_requires_only_enabled_criteria(tau_list, steps, min_tau_factor, tau_tol, expected):
    tau_list = [np.array([tau]) for tau in tau_list]
    converged, _, _ = EmceeBackend.stopping_criterion(tau_list, steps, 1, min_tau_factor, tau_tol)
    assert converged is expected



def test_stopping_description_formats_tolerance_and_effective_samples():
    # Progress text should show both values and the correct side of each threshold.
    config = Config(tau_tol=0.1, min_tau_factor=50)
    tolerance, effective = EmceeBackend.progress_description(0.05, 75, config)

    assert tolerance == "0.0500<0.1"
    assert effective == "75.00>50"



@pytest.mark.parametrize("min_tau_factor, tau_tol, expected", [
    (0, 0.1, ("0.0500<0.1", "disabled")),
    (50, 0, ("disabled", "75.00>50")),
])
def test_stopping_description_labels_disabled_threshold(min_tau_factor, tau_tol, expected):
    config = Config(min_tau_factor=min_tau_factor, tau_tol=tau_tol)
    assert EmceeBackend.progress_description(0.05, 75, config) == expected



def test_convergence_requires_an_enabled_criterion():
    acid = Acid(min_tau_factor=0, tau_tol=0)
    with pytest.raises(ACIDInputError, match="cannot both be 0"):
        acid.run_mcmc_until_converged(100)



@pytest.mark.parametrize("full_tau, min_tau_factor, expected_steps", [
    (30.0, 50, 2000),
    (50.0, 50, 3000),
    (np.nan, 50, 3000),
    (50.0, 0, 2000),
])
@pytest.mark.parametrize("tau_tol", [0.1, np.float64(0.1)])
def test_final_convergence_check_only_validates_neff(monkeypatch, full_tau, min_tau_factor, expected_steps, tau_tol):
    acid = Acid(verbose=0, check_interval=1000, min_tau_factor=min_tau_factor, tau_tol=tau_tol)
    calls, tolerances = [], []
    sampler = EnsembleSampler(4, 1, lambda theta: 0.0)

    def run_mcmc(nsteps, **kwargs):
        sampler.backend.iteration += nsteps

    def get_autocorr_time(tol, thin=None):
        calls.append(thin)
        return np.array([full_tau if thin is None else 10.0 + 0.1 * thin])

    sampler.run_mcmc = run_mcmc
    sampler.get_autocorr_time = get_autocorr_time
    monkeypatch.setattr("ACID_code.mcmc.emcee.EnsembleSampler", lambda **kwargs: sampler)
    monkeypatch.setattr(EmceeBackend, "_get_sampler_kwargs", lambda self, **kwargs: ({}, {"progress": False}))
    monkeypatch.setattr(EmceeBackend, "_get_sampler_pool", lambda self: (nullcontext(None), None, None))
    monkeypatch.setattr(EmceeBackend, "progress_description",
                        staticmethod(lambda tolerance, neff, config: (tolerances.append(tolerance) or "", "")))

    acid.run_mcmc_until_converged(3000)

    assert sampler.backend.iteration == expected_steps
    assert tolerances[1] == pytest.approx(0.1 / 10.2)
    assert (None in calls) == (min_tau_factor != 0)

