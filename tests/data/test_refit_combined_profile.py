"""Refitting stored continua must stack independent order observations once."""
import copy
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ACID_code import Config, Data, DataList, LSD
from ACID_code.diagnostics.errors import ACIDInputError, ACIDStateError


def make_orders(od=True, grouped=False):
    velocities = np.array([-5., 0., 5.])
    lines = {"wavelengths": np.array([5003., 5007., 5103., 5107.]),
             "depths": np.array([.2, .3, .4, .5])}
    orders = []
    for i, order in enumerate([10, 20]):
        data = Data()
        data.config = Config(order=order, od=od, verbose=0)
        data.velocities, data.linelist = velocities, lines
        data.ll_mask = np.arange(2*i, 2*i+2)
        data.profile_groups = np.array([2, 2, 7, 7]) if grouped else None
        data.profile_names = {2 if i == 0 else 7: "Iron" if i == 0 else "Titanium"} if grouped else None
        wave = np.linspace(5000+100*i, 5010+100*i, 501-100*i)
        continuum = 1. + .02 * np.linspace(-1, 1, len(wave))
        depths = -np.log1p(-lines["depths"][data.ll_mask]) if od else lines["depths"][data.ll_mask]
        alpha = LSD.calc_alpha(wave, lines["wavelengths"][data.ll_mask], depths, velocities)
        profile = np.array([.05, .15, .07]) * (i+1 if grouped else 1)
        model = np.exp(-alpha @ profile) if od else 1 - alpha @ profile
        for key, scale in [("final", 1.), ("initial", 1.01)]:
            data.wavelengths[key] = wave.copy()
            data.flux[key] = model * continuum
            data.errors[key] = continuum * .01 * (i+1)
            data.continuum[key] = continuum * scale
            data.sn[key] = 100.
            data.alpha[key] = alpha[None] if grouped else alpha
        flux = np.exp(-profile) if od else 1-profile
        data.profile["final"] = (flux[None] if grouped else flux,
                                  np.full((1, 3) if grouped else 3, .01), np.eye(3)*.0001)
        orders.append(data)
    return DataList.from_datalist(orders, verbose=0)


@pytest.mark.parametrize("od", [False, True])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("key", ["final", "initial"])
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("regularization", [0., .2])
def test_refit_matches_joint_normal_equations(monkeypatch, od, grouped, key, cached, regularization):
    orders = make_orders(od, grouped)
    information = np.zeros((6, 6) if grouped else (3, 3))
    rhs = np.zeros(6 if grouped else 3)
    for i, data in enumerate(orders):
        data.config.regularization = regularization
        alpha = data.alpha[key].reshape(-1, 3)
        flux = data.flux[key] / data.continuum[key]
        errors = data.errors[key] / data.continuum[key]
        y = -np.log(flux) if od else flux-1
        sigma = errors/flux if od else errors
        block = slice(3*i, 3*i+3) if grouped else slice(None)
        information[block, block] += alpha.T @ (alpha / sigma[:, None]**2)
        rhs[block] += alpha.T @ (y / sigma**2)
        if not cached:
            data.alpha.pop(key)
    differences = np.diff(np.eye(len(rhs)), axis=0)
    information += regularization * np.diag(information).mean() * differences.T @ differences
    covariance = np.linalg.inv(information)
    profile = covariance @ rhs
    expected = np.exp(-profile) if od else profile+1
    if od:
        covariance *= np.outer(expected, expected)
    original = pickle.dumps([data.to_dict() for data in orders])
    run_lsd = LSD.run_LSD
    calls = []

    def count_lsd(self, *args, **kwargs):
        calls.append(1)
        return run_lsd(self, *args, **kwargs)

    monkeypatch.setattr(LSD, "run_LSD", count_lsd)
    orders.refit_combined_profile(key=key)
    assert calls == [1]
    assert pickle.dumps([data.to_dict() for data in orders]) == original
    np.testing.assert_allclose(orders.combined_profile[0].ravel(), expected, atol=1e-12)
    np.testing.assert_allclose(orders.combined_profile[2], covariance, atol=1e-12)
    np.testing.assert_allclose(orders.combined_profile[1].ravel()**2, np.diag(covariance))
    if grouped:
        assert orders.profile_names == {2: "Iron", 7: "Titanium"}
        np.testing.assert_array_equal(orders.profiles.profile_ids, [2, 7])
    np.testing.assert_array_equal(orders.profiles.flux, orders.combined_profile[0])
    figure, _ = orders.plot_combined_profile(return_fig=True)
    plt.close(figure)


@pytest.mark.parametrize("mode", [None, "completed", "finished", "done", "containing", "includes", "COMPLETED"])
def test_refit_selects_by_key_instead_of_complete(mode):
    orders = make_orders()
    del orders[20].continuum["final"]
    assert not orders[10].complete
    orders.refit_combined_profile(orders=mode)
    assert orders._refit_data.config.order == [10]


def test_refit_all_explicit_orders_and_exclusions():
    orders = make_orders()
    del orders[20].continuum["final"]
    for selection in ["all", [10, 20], 20]:
        with pytest.raises(ACIDStateError, match="Order 20 lacks.*continuum"):
            orders.refit_combined_profile(orders=selection)
    orders.refit_combined_profile(orders="all", exclude=20)
    assert orders.exclude == [20]
    assert orders._refit_data.config.order == [10]
    orders.refit_combined_profile() # existing exclusions apply
    with pytest.raises(ACIDStateError, match="No non-excluded orders"):
        orders.refit_combined_profile(orders=20)
    with pytest.raises(ACIDStateError, match="No non-excluded orders"):
        orders.refit_combined_profile(key="absent")
    for kwargs in [{"orders": "unknown"}, {"orders": [99]}, {"exclude": 99}]:
        with pytest.raises(ACIDInputError):
            orders.refit_combined_profile(**kwargs)


def test_refit_preserves_overlapping_orders_as_independent_measurements():
    orders = make_orders()
    data = copy.deepcopy(orders[10])
    data.config.order = 20
    orders.data_list = [orders[10], data]
    orders.refit_combined_profile(orders=10)
    single = orders.combined_profile
    orders.refit_combined_profile(orders="all")
    np.testing.assert_allclose(orders.combined_profile[0], single[0])
    np.testing.assert_allclose(orders.combined_profile[2], single[2]/2)


def test_refit_masks_and_invalid_continua():
    orders = make_orders()
    data = orders[10]
    data.errors["final"][100] = 1e12
    data.flux["final"][100] = 1e6
    orders.refit_combined_profile()
    np.testing.assert_allclose(orders.combined_profile[0], np.exp(-np.array([.05, .15, .07])), atol=1e-12)
    original = orders.combined_profile
    data.continuum["final"][100] = 0
    with pytest.raises(ACIDInputError, match="positive and finite"):
        orders.refit_combined_profile()
    assert orders.combined_profile is original


def test_refit_save_load_and_switch_back_to_combination(tmp_path):
    orders = make_orders(grouped=True)
    orders.refit_combined_profile()
    expected = orders.combined_profile
    orders.save(str(tmp_path))
    restored = DataList.load(str(tmp_path), verbose=0)
    for actual, wanted in zip(restored.combined_profile, expected):
        np.testing.assert_array_equal(actual, wanted)
    assert restored.profile_names == {2: "Iron", 7: "Titanium"}
    np.testing.assert_array_equal(restored.profiles.profile_ids, [2, 7])
    restored.exclude = [20]
    assert restored._refit_data is None

    orders = make_orders()
    orders.refit_combined_profile()
    orders.combine_profiles()
    assert orders._refit_data is None


def test_refit_aligns_flattened_groups_and_rejects_conflicting_assignments():
    orders = make_orders(grouped=True)
    orders.refit_combined_profile()
    expected = orders.combined_profile
    third = copy.deepcopy(orders[10])
    third.config.order = 30
    third.ll_mask = np.arange(4)
    third.profile_groups = np.array([2, 2, 7, 7])
    third.profile_names = {2: "Iron", 7: "Titanium"}
    for name in ("wavelengths", "flux", "errors", "continuum"):
        getattr(third, name)["final"] = np.concatenate([getattr(data, name)["final"] for data in orders])
    alpha = np.zeros((2, len(third.flux["final"]), 3))
    edge = len(orders[10].flux["final"])
    alpha[0, :edge] = orders[10].alpha["final"][0]
    alpha[1, edge:] = orders[20].alpha["final"][0]
    third.alpha["final"] = LSD.flatten_alpha(alpha)
    orders.append(third, extend=True)
    orders.refit_combined_profile()
    np.testing.assert_allclose(orders.combined_profile[0], expected[0])
    np.testing.assert_allclose(orders.combined_profile[2], expected[2]/2)
    third.profile_groups = np.array([7, 7, 2, 2])
    with pytest.raises(ACIDInputError, match="same profile group"):
        orders.refit_combined_profile()


def test_refit_overrides_joint_display_without_replacing_joint_sampler():
    orders = make_orders(grouped=True)
    joint = copy.deepcopy(orders[10])
    joint.complete = True
    orders.multi_order_data = joint
    orders.refit_combined_profile()
    assert orders.multi_order_data is joint
    assert orders.combined_profile[0].shape == (2, 3)
    np.testing.assert_array_equal(orders.profiles.profile_ids, [2, 7])
    figure, _ = orders.plot_combined_profile(return_fig=True)
    plt.close(figure)
    orders.combine_profiles()
    np.testing.assert_array_equal(orders.combined_profile[0], joint.profile["final"][0])


def test_refit_single_profile_key_preserves_the_data_groups():
    orders = make_orders(grouped=True)
    for data in orders:
        data.profile_groups = np.array([2, 7, 2, 7])
        data.profile_names = {2: "Iron", 7: "Titanium"}
        lines = data.linelist["wavelengths"][data.ll_mask]
        depths = -np.log1p(-data.linelist["depths"][data.ll_mask])
        data.alpha["final"] = np.stack([LSD.calc_alpha(data.wavelengths["final"],
            lines[data.profile_groups == g], depths[data.profile_groups == g], data.velocities) for g in [2, 7]])
        for name in ("wavelengths", "flux", "errors", "sn", "continuum"):
            getattr(data, name)["single_profile"] = getattr(data, name)["final"]
        data.alpha["single_profile"] = data.alpha["final"].sum(axis=0)
    orders.refit_combined_profile()
    expected = orders.combined_profile
    orders.refit_combined_profile(key="single_profile")
    for actual, wanted in zip(orders.combined_profile, expected):
        np.testing.assert_allclose(actual, wanted)
    np.testing.assert_array_equal(orders.profiles.profile_ids, [2, 7])


@pytest.mark.parametrize("setting,value", [("od", False), ("regularization", .2)])
def test_refit_rejects_incompatible_order_settings(setting, value):
    orders = make_orders()
    setattr(orders[20].config, setting, value)
    with pytest.raises(ACIDInputError, match=f"same {setting}"):
        orders.refit_combined_profile()
