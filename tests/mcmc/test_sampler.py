from pathlib import Path
import copy
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code import Acid, Data, MCMC
from ACID_code.mcmc.emcee import EmceeBackend


def test_initial_state_and_sampler_kwargs_use_preprocessed_result(harps_result):
    # Reconstruct an independent Data instance so walker setup cannot mutate the shared result.
    data = Data().from_dict(harps_result.data.to_dict())
    acid = Acid(data=data)
    acid.config.parallel = False
    acid.config.nwalkers = 12
    state = acid.get_initial_state()

    # Deterministic sampling contains only the continuum coefficients.
    assert state.shape == (12, acid.config.poly_ord + 1)
    assert np.all(np.isfinite(state))

    # Sampler and run kwargs should carry the derived dimensions and requested state unchanged.
    sampler_kwargs, run_kwargs = EmceeBackend(acid.data)._get_sampler_kwargs(7, state)
    assert sampler_kwargs["nwalkers"] == 12
    assert sampler_kwargs["ndim"] == acid.config.poly_ord + 1
    np.testing.assert_array_equal(run_kwargs["initial_state"].coords, state)
    np.testing.assert_equal(run_kwargs["initial_state"].random_state, acid.data.get_rng().get_state())
    assert run_kwargs["nsteps"] == 7



def test_non_deterministic_initial_state_includes_every_profile_parameter(harps_result):
    # The legacy sampler adds one parameter for every column of the flattened alpha matrix.
    data = Data().from_dict(harps_result.data.to_dict())
    acid = Acid(data=data)
    acid.config.deterministic_profile = False
    acid.config.nwalkers = 12
    state = acid.get_initial_state()
    expected_ndim = acid.data.alpha["mcmc"].shape[1] + acid.config.poly_ord + 1

    assert acid.data.ndim == expected_ndim
    assert state.shape == (12, expected_ndim)


@pytest.mark.parametrize("od", [True, False])
@pytest.mark.parametrize("nprofiles", [1, 2])
def test_full_model_initial_state_recovers_stored_flux_profile(harps_result, monkeypatch, od, nprofiles):
    from types import SimpleNamespace

    data = Data().from_dict(copy.deepcopy(harps_result.data.to_dict()))
    data.config.update_hipri(deterministic_profile=False, od=od, nwalkers=12)
    if nprofiles == 2:
        data.alpha["mcmc"] = np.tile(data.alpha["mcmc"], (1, 2))
    profile = np.exp(-np.linspace(-0.01, 0.3, data.alpha["mcmc"].shape[1]))
    data.profile["masked"][0] = profile.reshape(nprofiles, -1)
    # Remove walker jitter so the fitted starting model can be checked directly.
    monkeypatch.setattr(Data, "get_rng", lambda self: SimpleNamespace(normal=lambda loc, scale: loc.copy()))
    state = Acid(data=data).get_initial_state()
    model = MCMC(data)
    _, z = model.full_model(state[0])

    np.testing.assert_allclose(np.exp(-z) if od else z + 1, profile)
    np.testing.assert_array_equal(state[0, model.k_max:], data.poly_coeffs["masked"])
    np.testing.assert_array_equal(data.profile["masked"][0], profile.reshape(nprofiles, -1))

