"""Sampling and result randomness must be independent of NumPy's global RNG."""
import copy
import pickle

import numpy as np
import pytest

from ACID_code import Acid, Data, Profiles, Result, utils


@pytest.fixture
def prepare_acid(harps_order_40):
    def prepare(seed=19, **kwargs):
        wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
        acid = Acid(seed=seed, verbose=0, parallel=False, poly_ord=2,
                    velocities=velocities, linelist=linelist, **kwargs)
        acid.ACID(wavelengths, flux, errors, sn, run_mcmc=False)
        return acid
    return prepare


def rng_state(rng):
    state = rng.get_state() if isinstance(rng, np.random.RandomState) else rng.bit_generator.state
    return pickle.dumps(state)


def test_seed_controls_walkers_and_chain_without_using_global_rng(prepare_acid):
    global_state = np.random.get_state()
    first = prepare_acid()
    first.run_mcmc(20, first.data.initial_state)
    assert pickle.dumps(np.random.get_state()) == pickle.dumps(global_state)

    try:
        np.random.seed(982)
        np.random.normal(size=10000)
        second = prepare_acid()
        second.run_mcmc(20, second.data.initial_state)
    finally:
        np.random.set_state(global_state)

    np.testing.assert_array_equal(first.data.initial_state, second.data.initial_state)
    np.testing.assert_array_equal(first.sampler.get_chain(), second.sampler.get_chain())
    assert rng_state(first.data.get_rng()) == rng_state(second.data.get_rng())
    assert rng_state(first.data.get_rng()) != rng_state(first.data.get_result_rng())


@pytest.mark.parametrize("storage", ["memory", "hdf", "converted"])
@pytest.mark.parametrize("seed", [19, None])
def test_split_run_matches_uninterrupted_after_results_and_reload(
        prepare_acid, tmp_path, monkeypatch, storage, seed):
    uninterrupted = prepare_acid(seed=seed)
    split = Acid(data=copy.deepcopy(uninterrupted.data))
    uninterrupted.run_mcmc(40, uninterrupted.data.initial_state)
    uninterrupted.data.nsteps = 40
    if storage == "hdf":
        split.config.sampler_path = str(tmp_path / "chain.h5")
    split.run_mcmc(17, split.data.initial_state)
    split.data.nsteps = 17

    # Force the random subsampling path, even for this small spectrum.
    monkeypatch.setattr(utils, "get_available_memory", lambda: 0)
    before_results = rng_state(split.data.get_rng())
    result = Result(split)
    result.process_results()
    assert rng_state(split.data.get_rng()) == before_results

    if storage != "memory":
        result.data.save(str(tmp_path / "data.pkl"),
                         sampler_path=str(tmp_path / "chain.h5"))
        result = Result(Data.load(str(tmp_path / "data.pkl")))
        assert rng_state(result.data.get_rng()) == before_results
        np.testing.assert_equal(result.sampler.random_state, split.sampler.random_state)

    global_state = np.random.get_state()
    try:
        np.random.seed(891)
        np.random.uniform(size=10000)
        result.continue_sampling(nsteps=23)
    finally:
        np.random.set_state(global_state)

    np.testing.assert_array_equal(result.sampler.get_chain(), uninterrupted.sampler.get_chain())
    np.testing.assert_array_equal(result.sampler.get_log_prob(), uninterrupted.sampler.get_log_prob())
    np.testing.assert_array_equal(result.sampler.backend.accepted, uninterrupted.sampler.backend.accepted)
    assert rng_state(result.data.get_rng()) == rng_state(uninterrupted.data.get_rng())
    expected = Result(uninterrupted)
    for actual_array, expected_array in zip(result.data.profile["final"], expected.data.profile["final"]):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_convergence_chunks_preserve_rng_sequence(prepare_acid):
    uninterrupted = prepare_acid()
    split = prepare_acid(check_interval=7, min_checks=100)
    uninterrupted.run_mcmc(40, uninterrupted.data.initial_state)
    split.run_mcmc_until_converged(17, split.data.initial_state)
    split.continue_sampling(max_steps=40)
    np.testing.assert_array_equal(split.sampler.get_chain(), uninterrupted.sampler.get_chain())
    assert rng_state(split.data.get_rng()) == rng_state(uninterrupted.data.get_rng())


def test_result_subsampling_is_repeatable_and_does_not_advance_rng(prepare_acid, monkeypatch):
    acid = prepare_acid()
    result = Result(acid, process_results=False)
    coefficients = np.random.default_rng(4).normal(size=(1500, 2))
    grid = np.linspace(-1, 1, 30)
    monkeypatch.setattr(utils, "get_available_memory", lambda: 0)
    before = rng_state(acid.data.get_rng())
    first = result._get_continuum_error(grid, coefficients)
    acid.data.get_result_rng().normal(size=500)
    second = result._get_continuum_error(grid, coefficients)
    np.testing.assert_array_equal(first, second)
    assert rng_state(acid.data.get_rng()) == before


def test_prepared_rng_survives_save_and_reload(prepare_acid, tmp_path):
    acid = prepare_acid(seed=None)
    acid.data.save(str(tmp_path / "prepared.pkl"))
    loaded = Acid(data=Data.load(str(tmp_path / "prepared.pkl")))
    assert rng_state(loaded.data.get_result_rng()) == rng_state(acid.data.get_result_rng())
    acid.run_mcmc(20, acid.data.initial_state)
    loaded.run_mcmc(20, loaded.data.initial_state)
    np.testing.assert_array_equal(acid.sampler.get_chain(), loaded.sampler.get_chain())


def test_running_prepared_acid_reuses_walkers_without_extra_draws(prepare_acid):
    direct = prepare_acid()
    prepared = Acid(data=copy.deepcopy(direct.data))
    initial_state = prepared.data.initial_state.copy()
    direct.run_mcmc(20, direct.data.initial_state)
    result = prepared.ACID(run_mcmc=True, nsteps=20)
    np.testing.assert_array_equal(prepared.data.initial_state, initial_state)
    np.testing.assert_array_equal(result.sampler.get_chain(), direct.sampler.get_chain())
    assert rng_state(prepared.data.get_rng()) == rng_state(direct.data.get_rng())


def test_legacy_saved_data_resumes_from_backend_rng(prepare_acid, tmp_path):
    uninterrupted = prepare_acid()
    split = Acid(data=copy.deepcopy(uninterrupted.data))
    uninterrupted.run_mcmc(30, uninterrupted.data.initial_state)
    split.run_mcmc(10, split.data.initial_state)
    split.data.save(str(tmp_path / "data.pkl"), sampler_path=str(tmp_path / "chain.h5"))
    payload = split.data.to_dict()
    del payload["_rng"], payload["_initial_rng"]
    loaded = Acid(data=Data().from_dict(payload))
    loaded.continue_sampling(nsteps=20)
    np.testing.assert_array_equal(loaded.sampler.get_chain(), uninterrupted.sampler.get_chain())


def test_rng_snapshots_do_not_alias_and_reset_reseeds(prepare_acid):
    acid = prepare_acid()
    before = rng_state(acid.data.get_rng())
    payload = acid.data.to_dict()
    loaded = Data().from_dict(payload)
    loaded.get_rng().normal(size=100)
    assert rng_state(payload["_rng"]) == before
    assert rng_state(acid.data.get_rng()) == before
    initial = rng_state(acid.data.get_result_rng())
    acid.data.reset()
    assert rng_state(acid.data.get_rng()) == initial


def test_data_backed_profile_fits_use_initial_rng_copy(prepare_acid):
    acid = prepare_acid()
    velocities = np.linspace(-12, 12, 61)
    flux = Profiles.gaussian_func(velocities, -0.25, 1.5, 2.2)
    acid.data._velocities = velocities
    acid.data.profile["final"] = (flux, np.full_like(flux, 0.01), None)
    before = rng_state(acid.data.get_rng())
    first = Profiles(data=acid.data)
    first.fit_gaussian()
    second = Profiles(data=acid.data)
    second.fit_gaussian()
    np.testing.assert_array_equal(first.fitted_yerr["gaussian"], second.fitted_yerr["gaussian"])
    assert rng_state(acid.data.get_rng()) == before


def test_dynesty_uses_private_seeded_generator(prepare_acid):
    pytest.importorskip("dynesty")
    first = prepare_acid(sampler_type="dynesty", nsteps=20)
    second = prepare_acid(sampler_type="dynesty", nsteps=20)
    assert isinstance(first.data.get_rng(), np.random.Generator)
    initial = rng_state(first.data.get_rng())
    first.run_mcmc(20)
    global_state = np.random.get_state()
    try:
        np.random.seed(3)
        np.random.normal(size=10000)
        second.run_mcmc(20)
    finally:
        np.random.set_state(global_state)
    assert first.sampler.rstate is first.data.get_rng()
    assert rng_state(first.data.get_rng()) != initial
    assert rng_state(first.data.get_result_rng()) == initial
    np.testing.assert_array_equal(first.sampler.results.samples, second.sampler.results.samples)
