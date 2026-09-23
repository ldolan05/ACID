#%%
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ACID_code import Config, Data, DataList


def test_datalist_load_infers_current_directory_after_move(tmp_path):
    data = Data()
    data.config = Config(verbose=0)
    old_folder = tmp_path / "order_1"
    data.save(str(old_folder / "data.pkl"))
    old_folder.rename(tmp_path / "order_20")

    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded.orders.tolist() == [20]
    assert 20 in loaded.order_range
    assert loaded[20].config.save_path == str(tmp_path / "order_20" / "data.pkl")
    assert loaded[20].config.sampler_path == str(tmp_path / "order_20" / "sampler.h5")
    assert loaded[20].sampler is None
    assert not (tmp_path / "order_1").exists()
    assert not (tmp_path / "order_None").exists()
    assert Data.load(loaded[20].config.save_path).config.order == 20


def test_packed_datalist_infers_unset_order_from_stored_path(tmp_path):
    import pickle

    data = Data()
    data.config = Config(verbose=0)
    data.save(str(tmp_path / "order_12" / "data.pkl"))
    with open(tmp_path / "datalist.pkl", "wb") as stream:
        pickle.dump({"data_list": [data.to_dict()], "verbose": 0}, stream)
    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded.orders.tolist() == [12]
    assert 12 in loaded.order_range
    assert DataList.load(str(tmp_path), verbose=0).orders.tolist() == [12]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("relocate", [False, True])
def test_disabled_figures_survive_datalist_save_load(tmp_path, packed, relocate):
    root = tmp_path / "original"
    root.mkdir()
    data = Data()
    data.config = Config(figure_dir="None", dir=str(root / "order_20"), order=20, verbose=0)
    data.save()
    assert Data.load(data.config.save_path).config.figure_dir is None
    if packed:
        DataList.from_datalist([data], save_dir=str(root), verbose=0).save()
    if relocate:
        root = root.rename(tmp_path / "relocated")

    loaded = DataList.load(str(root), verbose=0)
    assert loaded[20].config.figure_dir is None
    assert loaded[20].config.dir == str(root / "order_20")
    assert not (root / "order_20" / "figures").exists()
    assert DataList.load(str(root), verbose=0)[20].config.figure_dir is None


def test_datalist_indexes_orders_and_persists_inputs(tmp_path, synthetic_spectrum):
    # Use non-consecutive order labels to test the instrument-order mapping explicitly.
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    configs = [Config(poly_ord=2), Config(poly_ord=4)]
    save_dir = tmp_path / "results"
    datalist = DataList(np.array([wavelengths, wavelengths]), np.array([flux, flux]),
                        np.array([errors, errors]), np.array([sn, sn]), velocities, linelist,
                        order_range=[10, 12], config=configs, save_dir=str(save_dir))

    # Initialisation saves a lightweight Data object for every requested order.
    assert len(datalist) == 2
    assert datalist[12].config.order == 12
    assert [datalist[order].config.poly_ord for order in [10, 12]] == [2, 4]
    assert (save_dir / "order_10" / "data.pkl").exists()
    with pytest.raises(KeyError):
        _ = datalist[11]


def test_datalist_save_load_and_input_validation(tmp_path, completed_datalist):
    import pickle

    # Saving a packed DataList should permit loading from either its directory or pickle file.
    datalist = completed_datalist
    for data in datalist:
        data.linelist = [[5000.0, 5002.0], [0.1, 0.2]]
    datalist.save(str(tmp_path))
    with (tmp_path / "datalist.pkl").open("rb") as stream:
        payload = pickle.load(stream)
    for saved_data in payload["data_list"]:
        assert type(saved_data["_linelist"]) is dict
    from_directory = DataList.load(str(tmp_path))
    from_file = DataList.load(str(tmp_path / "datalist.pkl"))

    assert from_directory.orders.tolist() == [20, 21, 22]
    assert from_file.orders.tolist() == [20, 21, 22]
    for original, loaded in zip(datalist, from_file):
        np.testing.assert_array_equal(loaded.linelist[0], original.linelist[0])
        np.testing.assert_array_equal(loaded.linelist[1], original.linelist[1])

    # The Results property mirrors every order and caches the constructed Result objects.
    assert len(datalist.results) == len(datalist)
    assert datalist.results is datalist.results

    # Invalid selection and worker arguments are rejected before any ACID call is made.
    with pytest.raises(ValueError, match="available orders"):
        datalist.run_ACID(orders=[99])
    with pytest.raises(ValueError, match="Both worker"):
        datalist.run_ACID(worker=0)

    # Missing paths and invalid load targets should fail without touching any order data.
    with pytest.raises(ValueError, match="No save directory"):
        DataList.from_datalist(datalist.data_list).save()
    with pytest.raises(ValueError, match="not a directory"):
        DataList.load(str(tmp_path / "missing"))


def test_datalist_path_relocation_updates_each_data_file(tmp_path, completed_datalist):
    # Relocation still updates the data path when there is no sampler.
    data = completed_datalist[20]
    changed = DataList._update_paths_for_data(data, str(tmp_path))
    expected_directory = tmp_path / "order_20"

    assert changed is True
    assert data.config.save_path == str(expected_directory / "data.pkl")
    assert data.config.dir == str(expected_directory)
    assert data.config.sampler_path == str(expected_directory / "sampler.h5")
    assert data.sampler is None
    assert not (expected_directory / "sampler.h5").exists()
    assert (expected_directory / "data.pkl").exists()

    # Reapplying the same root is a no-op and should not rewrite the file.
    assert DataList._update_paths_for_data(data, str(tmp_path)) is False


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("stored_sampler", [False, True])
def test_datalist_load_without_sampler_does_not_rewrite(tmp_path, capsys, packed, stored_sampler):
    data = Data()
    # A missing dir now intentionally requires migration. Start with an established
    # root so this tests only the absence of optional sampler/figure files.
    data.config = Config(order=50, verbose=2, dir=str(tmp_path / "order_50"))
    if stored_sampler:
        data.config.sampler_path = str(tmp_path / "missing.h5")
    filename = tmp_path / "order_50" / "data.pkl"
    data.save(str(filename))
    if packed:
        DataList.from_datalist([data], save_dir=str(tmp_path)).save()
    (tmp_path / "order_50" / "figures").rmdir()
    paths = [filename] + ([tmp_path / "datalist.pkl"] if packed else [])
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    capsys.readouterr()

    loaded = DataList.load(str(tmp_path))

    assert loaded[50].sampler is None
    assert loaded[50].config.sampler_path == data.config.sampler_path
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before
    output = capsys.readouterr().out
    assert "Data object saved" not in output
    assert "current location" not in output
    assert not (tmp_path / "order_50" / "figures").exists()


@pytest.mark.parametrize("packed", [False, True])
def test_datalist_missing_dir_is_migrated_once(tmp_path, packed):
    data = Data()
    data.config = Config(order=50, verbose=0)
    filename = tmp_path / "order_50" / "data.pkl"
    data.save(str(filename))
    if packed:
        DataList.from_datalist([data], save_dir=str(tmp_path), verbose=0).save()

    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded[50].config.dir == str(filename.parent)
    assert Data.load(str(filename)).config.dir == str(filename.parent)
    paths = [filename] + ([tmp_path / "datalist.pkl"] if packed else [])
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    DataList.load(str(tmp_path), verbose=0)
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


@pytest.mark.parametrize("packed", [False, True])
def test_datalist_persists_inferred_order_with_matching_paths(tmp_path, packed):
    import pickle

    data = Data()
    data.config = Config(dir=str(tmp_path / "order_0"), verbose=0)
    data.save()
    if packed:
        with open(tmp_path / "datalist.pkl", "wb") as stream:
            pickle.dump({"data_list": [data.to_dict()]}, stream)
    loaded = DataList.load(str(tmp_path), verbose=0)
    assert loaded.orders.tolist() == [0]
    # Inspect the saved payload: Data.load would infer again and hide a missed save.
    with open(tmp_path / "order_0" / "data.pkl", "rb") as stream:
        assert pickle.load(stream)["config"]["order"] == 0


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("keep_original", [False, True])
def test_datalist_relocation_rebinds_sampler(tmp_path, harps_result, packed, keep_original):
    import shutil
    from emcee.backends import HDFBackend

    original = tmp_path / "original"
    original.mkdir()
    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(order=0, dir=str(original / "order_0"), verbose=0)
    backend = HDFBackend(data.config.sampler_path)
    backend.reset(4, 1)
    data.sampler = backend
    # Persist explicit overrides as well as dir; all must relocate together.
    data.config.figure_dir = str(original / "order_0" / "figures")
    data.save(data.config.save_path)
    if packed:
        DataList.from_datalist([data], save_dir=str(original), verbose=0).save()
    relocated = tmp_path / "relocated"
    if keep_original:
        shutil.copytree(original, relocated)
    else:
        original.rename(relocated)

    loaded = DataList.load(str(relocated), verbose=0)
    order_dir = relocated / "order_0"
    assert loaded[0].config.dir == str(order_dir)
    assert loaded[0].config.save_path == str(order_dir / "data.pkl")
    assert loaded[0].config.figure_dir == str(order_dir / "figures")
    assert loaded[0].config.sampler_path == str(order_dir / "sampler.h5")
    assert loaded[0].sampler.backend.filename == str(order_dir / "sampler.h5")
    assert loaded[0].sampler.backend.shape == (4, 1)
    assert original.exists() == keep_original
    paths = [order_dir / "data.pkl", order_dir / "sampler.h5"]
    if packed:
        paths.append(relocated / "datalist.pkl")
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths]
    DataList.load(str(relocated), verbose=0)
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in paths] == before


@pytest.mark.parametrize("path_kind", ["save_path", "sampler_path", "figure_dir"])
def test_datalist_repairs_explicit_path_with_matching_dir(tmp_path, harps_result, path_kind):
    from emcee.backends import HDFBackend

    order_dir = tmp_path / "order_20"
    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(order=20, dir=str(order_dir), verbose=0)
    if path_kind == "sampler_path":
        HDFBackend(str(order_dir / "sampler.h5")).reset(4, 1)
        data.config.sampler_path = str(tmp_path / "missing.h5")
    elif path_kind == "figure_dir":
        data.config.figure_dir = str(tmp_path / "old_figures")
        (tmp_path / "old_figures").rmdir()
    else:
        data.config.save_path = str(tmp_path / "old.pkl")
    assert DataList._update_paths_for_data(data, str(tmp_path)) is True
    assert data.config.save_path == str(order_dir / "data.pkl")
    assert data.config.sampler_path == str(order_dir / "sampler.h5")
    assert data.config.figure_dir == str(order_dir / "figures")
    assert DataList._update_paths_for_data(data, str(tmp_path)) is False


@pytest.mark.parametrize("relocate", [False, True])
@pytest.mark.parametrize("newer_source", ["order", "packed"])
def test_datalist_load_uses_newer_saved_results(tmp_path, relocate, newer_source):
    import os

    root = tmp_path / "original"
    root.mkdir()
    data = Data()
    data.config = Config(order=20, dir=str(root / "order_20"), verbose=0)
    data.save()
    DataList.from_datalist([data], save_dir=str(root), verbose=0).save()
    data.complete = True
    data.save()
    order_file = root / "order_20" / "data.pkl"
    packed_file = root / "datalist.pkl"
    # Explicit times avoid depending on filesystem timestamp resolution or sleeps.
    older = 1_700_000_000_000_000_000
    newer = older + 2_000_000_000
    order_time, packed_time = (newer, older) if newer_source == "order" else (older, newer)
    os.utime(order_file, ns=(order_time, order_time))
    os.utime(packed_file, ns=(packed_time, packed_time))
    if relocate:
        root = root.rename(tmp_path / "relocated")

    loaded = DataList.load(str(root), verbose=0)
    expected_complete = newer_source == "order"
    assert loaded[20].complete is expected_complete
    # Loading a moved snapshot must never roll a newer per-order result back.
    assert Data.load(str(root / "order_20" / "data.pkl")).complete is (
        expected_complete if relocate else True
    )
    assert DataList.load(str(root), verbose=0)[20].complete is expected_complete


@pytest.mark.parametrize("relocate", [False, True])
def test_packed_datalist_includes_new_orders(tmp_path, relocate):
    root = tmp_path / "original"
    root.mkdir()
    first = Data()
    first.config = Config(order=20, dir=str(root / "order_20"), verbose=0)
    first.save()
    DataList.from_datalist([first], save_dir=str(root), verbose=0).save()
    added = Data()
    added.config = Config(order=21, dir=str(root / "order_21"), verbose=0)
    added.complete = True
    added.save()
    if relocate:
        root = root.rename(tmp_path / "relocated")
    for _ in range(2):
        loaded = DataList.load(str(root), verbose=0)
        assert loaded.orders.tolist() == [20, 21]
        assert loaded[21].complete


def test_exported_sampler_keeps_saving_new_steps(tmp_path, harps_result):
    import emcee

    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(verbose=0)
    data.sampler = emcee.EnsembleSampler(4, 1, lambda x: -0.5 * np.sum(x ** 2))
    data.sampler.run_mcmc(np.array([[-1.], [-0.3], [0.3], [1.]]), 3)
    filename = str(tmp_path / "data.pkl")
    data.save(filename, str(tmp_path / "sampler.h5"))
    data.sampler.run_mcmc(None, 2)
    data.save()
    loaded = Data.load(filename)
    assert loaded.sampler.backend.iteration == 5
    np.testing.assert_array_equal(loaded.sampler.get_chain(), data.sampler.get_chain())


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_save_preserves_checkpoint(tmp_path, monkeypatch, packed, failure):
    import os
    import pickle

    data = Data()
    data.config = Config(order=20, verbose=0)
    filename = tmp_path / ("datalist.pkl" if packed else "data.pkl")
    save = (DataList.from_datalist([data], save_dir=str(tmp_path), verbose=0).save
            if packed else lambda: data.save(str(filename)))
    save()
    before = filename.read_bytes()
    data.complete = True

    def fail(*args, **kwargs):
        if failure == "write":
            args[1].write(b"partial pickle")
        raise OSError("simulated save failure")

    with monkeypatch.context() as patch:
        patch.setattr(pickle if failure == "write" else os,
                      "dump" if failure == "write" else "replace", fail)
        with pytest.raises(OSError, match="simulated save failure"):
            save()
    assert filename.read_bytes() == before
    assert list(tmp_path.iterdir()) == [filename]
    save()
    payload = pickle.loads(filename.read_bytes())
    assert (payload["data_list"][0] if packed else payload)["complete"]


@pytest.mark.parametrize("packed", [False, True])
def test_datalist_copy_without_sampler_detaches_original(tmp_path, harps_result, packed):
    import shutil
    from emcee.backends import HDFBackend

    original = tmp_path / "original"
    original.mkdir()
    data = Data().from_dict(harps_result.data.to_dict())
    data.config = Config(order=20, dir=str(original / "order_20"), verbose=0)
    backend = HDFBackend(data.config.sampler_path)
    backend.reset(4, 1)
    data.sampler = backend
    data.save()
    if packed:
        DataList.from_datalist([data], save_dir=str(original), verbose=0).save()
    original_sampler = Path(data.config.sampler_path)
    before = (original_sampler.read_bytes(), original_sampler.stat().st_mtime_ns)
    copied = tmp_path / "copied"
    shutil.copytree(original, copied, ignore=shutil.ignore_patterns("sampler.h5"))

    loaded = DataList.load(str(copied), verbose=0)[20]
    assert loaded.sampler is None
    assert loaded.config.sampler_path == str(copied / "order_20" / "sampler.h5")
    assert loaded.complete
    assert DataList.load(str(copied), verbose=0)[20].sampler is None
    assert (original_sampler.read_bytes(), original_sampler.stat().st_mtime_ns) == before
    assert not (copied / "order_20" / "sampler.h5").exists()


@pytest.mark.parametrize("store_sampler", [False, True])
@pytest.mark.parametrize("use_save_dir", [False, True])
@pytest.mark.parametrize("config_source", ["config", "kwargs"])
def test_datalist_save_dir_overrides_paths_and_controls_sampler(
    tmp_path, synthetic_spectrum, store_sampler, use_save_dir, config_source,
):
    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    settings = dict(dir=str(custom_dir), save_path=str(custom_dir / "custom.pkl"),
                    sampler_path=str(custom_dir / "custom.h5"),
                    figure_dir=str(custom_dir / "plots"))
    config_args = {"config": Config(**settings)} if config_source == "config" else settings
    save_dir = tmp_path / "results"
    datalist = DataList(wavelengths[None], flux[None], errors[None], [sn],
                        velocities, linelist, order_range=[20], verbose=0,
                        save_dir=str(save_dir) if use_save_dir else None,
                        store_sampler=store_sampler, **config_args)
    cfg = datalist[20].config
    if use_save_dir:
        order_dir = save_dir / "order_20"
        assert cfg.dir == str(order_dir)
        assert cfg.save_path == str(order_dir / "data.pkl")
        assert cfg.figure_dir == str(order_dir / "figures")
        expected_sampler = str(order_dir / "sampler.h5") if store_sampler else None
        assert cfg.sampler_path == expected_sampler
        loaded = DataList.load(str(save_dir), verbose=0)
        assert loaded[20].config.sampler_path == expected_sampler
        assert not (custom_dir / "custom.pkl").exists()
    else:
        for key, value in settings.items():
            assert getattr(cfg, key) == value
    # Wrapping existing instances must not reconfigure their output settings.
    before = cfg.to_dict()
    DataList.from_datalist([datalist[20]], save_dir=str(tmp_path), verbose=0)
    assert cfg.to_dict() == before


def test_datalist_constructor_relocates_reused_data(tmp_path, synthetic_spectrum):
    import shutil

    wavelengths, flux, errors, sn, velocities, linelist = synthetic_spectrum
    inputs = dict(wavelengths=wavelengths[None], flux=flux[None], errors=errors[None],
                  sn=[sn], velocities=velocities, linelist=linelist,
                  order_range=[20], config=Config(verbose=0), verbose=0)
    original, copied = tmp_path / "original", tmp_path / "copied"
    DataList(**inputs, save_dir=str(original))
    original_file = original / "order_20" / "data.pkl"
    before = original_file.read_bytes()
    shutil.copytree(original, copied)

    reused = DataList(**inputs, save_dir=str(copied), overwrite=False)[20]
    reused.config.poly_ord = 7
    reused.save()
    assert original_file.read_bytes() == before
    assert Data.load(str(copied / "order_20" / "data.pkl")).config.poly_ord == 7
    assert reused.config.dir == str(copied / "order_20")
    assert reused.config.save_path == str(copied / "order_20" / "data.pkl")
    assert reused.config.sampler_path == str(copied / "order_20" / "sampler.h5")
    assert reused.config.figure_dir == str(copied / "order_20" / "figures")



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--tb=auto"]))
