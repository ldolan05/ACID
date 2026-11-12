"""Restart interrupted HDF5 runs without repeating preparation."""
import shutil

import numpy as np
import pytest
import emcee
from emcee.backends import HDFBackend

from ACID_code import Acid, Config, Data, DataList, Result
from ACID_code.diagnostics.errors import ACIDStateError


@pytest.mark.parametrize("paths", ["none", "dir", "custom"])
@pytest.mark.parametrize("resume_source", ["method", "init", "config"])
def test_resume_starts_new_run_when_no_saved_run(tmp_path, synthetic_spectrum, paths, resume_source):
    wave, flux, errors, sn, velocities, linelist = synthetic_spectrum
    saved_paths = {}
    if paths == "dir":
        saved_paths = {"dir": str(tmp_path)}
    elif paths == "custom":
        saved_paths = {"save_path": str(tmp_path / "state.pkl"),
                       "sampler_path": str(tmp_path / "chain.h5")}
    resume_kwargs = {"resume": True} if resume_source == "init" else {}
    if resume_source == "config":
        resume_kwargs = {"config": Config(resume=True)}
    acid = Acid(velocities=velocities, linelist=linelist, parallel=False, verbose=0,
                seed=2, poly_ord=1, sigma_lower=100., sigma_upper=100.,
                dev_perc=100., masking_lines={}, n_bins=10, **saved_paths, **resume_kwargs)
    # A fresh allowed-resume run must be identical to an ordinary run.
    reference = Acid(config=Config(**acid.config.to_dict()), velocities=velocities, linelist=linelist)
    reference.config.update_hipri(save_path="off", sampler_path="off", resume=False)
    expected = reference.ACID(wave, flux, errors, sn, nsteps=45)
    result = acid.ACID(wave, flux, errors, sn, nsteps=40,
                       **({"resume": True} if resume_source == "method" else {}))
    assert result.data.complete
    assert result.data.nsteps == 40
    np.testing.assert_array_equal(result.sampler.get_chain(), expected.sampler.get_chain()[:40])
    if saved_paths:
        prefix = result.sampler.get_chain().copy()
        result = Acid(**saved_paths).ACID(resume=True, nsteps=45)
        assert result.data.nsteps == 45
        np.testing.assert_array_equal(result.sampler.get_chain()[:40], prefix)
        np.testing.assert_array_equal(result.sampler.get_chain(), expected.sampler.get_chain())


@pytest.mark.parametrize("resume_source", ["method", "init", "config"])
def test_resume_respects_disabled_paths(interrupted_run, synthetic_spectrum, resume_source):
    wave, flux, errors, sn, velocities, linelist = synthetic_spectrum
    original_data = (interrupted_run / "data.pkl").read_bytes()
    original_chain = (interrupted_run / "sampler.h5").read_bytes()
    disabled = dict(save_path="off", sampler_path="off")
    init_kwargs = {}
    if resume_source == "init":
        init_kwargs = disabled
    elif resume_source == "config":
        init_kwargs = {"config": Config(**disabled)}
    acid = Acid(dir=str(interrupted_run), velocities=velocities, linelist=linelist,
                parallel=False, verbose=0, seed=2, poly_ord=1, sigma_lower=100.,
                sigma_upper=100., dev_perc=100., masking_lines={}, n_bins=10,
                **init_kwargs)
    result = acid.ACID(wave, flux, errors, sn, resume=True, nsteps=40,
                       **(disabled if resume_source == "method" else {}))
    assert result.data.complete
    assert result.data.config.save_path is None
    assert result.data.config.sampler_path is None
    np.testing.assert_array_equal(result.data.velocities, velocities)
    assert (interrupted_run / "data.pkl").read_bytes() == original_data
    assert (interrupted_run / "sampler.h5").read_bytes() == original_chain


@pytest.mark.parametrize("custom_paths", [False, True])
def test_resume_fresh_run_uses_call_time_paths(interrupted_run, synthetic_spectrum, custom_paths):
    wave, flux, errors, sn, velocities, linelist = synthetic_spectrum
    original_data = (interrupted_run / "data.pkl").read_bytes()
    original_chain = (interrupted_run / "sampler.h5").read_bytes()
    directory = interrupted_run / "new"
    if custom_paths:
        paths = dict(save_path=str(directory / "state.pkl"), sampler_path=str(directory / "chain.h5"))
    else:
        paths = dict(dir=str(directory))
    acid = Acid(dir=str(interrupted_run), velocities=velocities, linelist=linelist,
                parallel=False, verbose=0, seed=2, poly_ord=1, sigma_lower=100.,
                sigma_upper=100., dev_perc=100., masking_lines={}, n_bins=10)
    result = acid.ACID(wave, flux, errors, sn, resume=True, nsteps=40, **paths)

    assert result.data.nsteps == 40
    np.testing.assert_array_equal(result.data.velocities, velocities)
    assert (directory / ("state.pkl" if custom_paths else "data.pkl")).exists()
    assert (directory / ("chain.h5" if custom_paths else "sampler.h5")).exists()
    assert (interrupted_run / "data.pkl").read_bytes() == original_data
    assert (interrupted_run / "sampler.h5").read_bytes() == original_chain


@pytest.fixture
def interrupted_run(tmp_path, harps_order_40, monkeypatch):
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(dir=str(tmp_path), velocities=velocities, linelist=linelist,
                seed=19, poly_ord=2, parallel=False, verbose=0)
    run_mcmc = acid.run_mcmc

    def interrupt(nsteps, state=None):
        run_mcmc(17, state)
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(acid, "run_mcmc", interrupt)
        with pytest.raises(KeyboardInterrupt):
            acid.ACID(wavelengths, flux, errors, sn, nsteps=40)
    assert (tmp_path / "data.pkl").exists()
    assert HDFBackend(str(tmp_path / "sampler.h5")).iteration == 17
    # The pickle predates sampling; the backend supplies the actual step count.
    assert Data.load(str(tmp_path / "data.pkl")).nsteps == 0
    return tmp_path


def test_resume_matches_uninterrupted_and_skips_preparation(interrupted_run, monkeypatch):
    directory = str(interrupted_run)
    reference = Acid(data=Data.load(str(interrupted_run / "data.pkl")))
    reference.sampler = None
    reference.config.update_hipri(save_path="off", sampler_path="off")
    reference.run_mcmc(40, reference.data.initial_state)
    reference.data.nsteps = 40

    def unexpected(*args, **kwargs):
        raise AssertionError("Resume repeated preparation")

    monkeypatch.setattr(Data, "set_inputs", unexpected)
    monkeypatch.setattr(Acid, "scipy_continuum_fit", unexpected)
    monkeypatch.setattr(Acid, "get_initial_state", unexpected)
    resumed = Acid(dir=directory, resume=True, parallel=False)
    original_data = resumed.data
    # Supplied spectra and model settings must not change the saved likelihood.
    result = resumed.ACID(flux=np.zeros(2), poly_ord=5, profile_groups=np.ones(2))
    assert resumed.data is original_data
    assert resumed.config.poly_ord == 2
    assert resumed.data.nsteps == 40
    assert result.data.complete
    np.testing.assert_array_equal(resumed.sampler.get_chain(), reference.sampler.get_chain())
    np.testing.assert_array_equal(resumed.sampler.get_log_prob(), reference.sampler.get_log_prob())
    np.testing.assert_equal(resumed.data.get_rng().get_state(), reference.data.get_rng().get_state())
    expected = Result(reference)
    for actual, wanted in zip(result.data.profile["final"], expected.data.profile["final"]):
        np.testing.assert_array_equal(actual, wanted)

    # Repeating a resume at the same target does not append samples.
    again = Acid(dir=directory).ACID(resume=True)
    assert again.data.nsteps == 40
    np.testing.assert_array_equal(again.sampler.get_chain(), reference.sampler.get_chain())


def test_resume_max_steps_is_a_total_cap(interrupted_run):
    result = Acid(dir=str(interrupted_run)).ACID(resume=True, max_steps=24,
                                               check_interval=4, min_checks=100)
    assert result.data.nsteps == 24
    prefix = result.sampler.get_chain().copy()
    result.continue_sampling(nsteps=3)
    assert result.data.nsteps == 27
    np.testing.assert_array_equal(result.sampler.get_chain()[:24], prefix)


@pytest.mark.parametrize("custom_paths", [False, True])
def test_resume_can_restore_without_sampling_after_relocation(interrupted_run, tmp_path, custom_paths):
    copied = tmp_path / "copied"
    shutil.copytree(interrupted_run, copied)
    original_chain = (interrupted_run / "sampler.h5").read_bytes()
    if custom_paths:
        (copied / "data.pkl").rename(copied / "state.pkl")
        (copied / "sampler.h5").rename(copied / "chain.h5")
        paths = dict(save_path=str(copied / "state.pkl"), sampler_path=str(copied / "chain.h5"))
    else:
        paths = dict(dir=str(copied))
    acid = Acid()
    assert acid.ACID(**paths, resume=True, run_mcmc=False) is None
    assert acid.data.nsteps == 17
    assert acid.config.save_path == str(copied / ("state.pkl" if custom_paths else "data.pkl"))
    assert acid.config.sampler_path == str(copied / ("chain.h5" if custom_paths else "sampler.h5"))
    acid.continue_sampling(nsteps=3)
    assert acid.data.nsteps == 20
    assert (interrupted_run / "sampler.h5").read_bytes() == original_chain


def test_datalist_can_resume_a_failed_order(interrupted_run):
    data = Data.load(str(interrupted_run / "data.pkl"))
    data.exception = "previous sampling failure"
    data.config.order = 20
    data.save()
    datalist = DataList.from_datalist([data], verbose=0)
    datalist.run_ACID(resume=True, nsteps=23, overwrite_kwargs=True)
    assert datalist[20] is data
    assert data.complete
    assert data.exception is None
    assert data.nsteps == 23


@pytest.mark.parametrize("missing", ["data.pkl", "sampler.h5"])
def test_resume_missing_files_preserves_chain(interrupted_run, missing):
    chain = interrupted_run / "sampler.h5"
    before = chain.read_bytes()
    (interrupted_run / missing).rename(interrupted_run / (missing + ".moved"))
    with pytest.raises(ACIDStateError, match="requires"):
        Acid(dir=str(interrupted_run)).ACID(resume=True)
    if chain.exists():
        assert chain.read_bytes() == before
    else:
        assert (interrupted_run / "sampler.h5.moved").read_bytes() == before


def test_resume_rejects_mismatched_and_empty_backends(interrupted_run):
    backend = HDFBackend(str(interrupted_run / "sampler.h5"))
    backend.reset(4, 1)
    backend.grow(1, None)
    backend.save_step(emcee.State(np.zeros((4, 1)), log_prob=np.zeros(4),
                                 random_state=np.random.RandomState(1).get_state()),
                      np.zeros(4, dtype=bool))
    before = (interrupted_run / "sampler.h5").read_bytes()
    with pytest.raises(ACIDStateError, match="shape"):
        Acid(dir=str(interrupted_run)).ACID(resume=True)
    assert (interrupted_run / "sampler.h5").read_bytes() == before
    backend.reset(12, 3)
    with pytest.raises(ACIDStateError, match="no samples"):
        Acid(dir=str(interrupted_run)).ACID(resume=True)
