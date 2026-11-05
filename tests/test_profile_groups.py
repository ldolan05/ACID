"""Profile labels and automatic grouping must survive clipping and plotting."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ACID_code import Acid, Config, Data, DataList, LSD, Profiles, ACIDProfileGroupsWarning
from ACID_code.diagnostics.errors import ACIDInputError


def run_grouped_lsd(**kwargs):
    wavelength = np.linspace(5000, 5020, 2001)
    linelist = {"wavelengths": np.r_[4990, np.linspace(5001, 5019, 60), 5030],
                "depths": np.r_[0.4, np.linspace(0.1, 0.8, 60), 0.4],
                "spec_ion": np.array(["Ca 1"] + ["Fe 1"] * 20 + ["Ti 1"] * 20 + ["Other"] * 20 + ["Ca 1"]),
                "lande_factor": np.ones(62)}
    lsd = LSD(verbose=0)
    lsd.run_LSD(wavelength, np.ones_like(wavelength), np.full_like(wavelength, 0.01), 100.,
                velocities=np.array([-2., 0., 2.]), linelist=linelist, **kwargs)
    return lsd


@pytest.mark.parametrize("names", ["iron", [], ["iron", 5], {"0": "iron"}, {-1: "iron"}, {True: "iron"}, {0: ""}])
def test_config_rejects_invalid_profile_names(names):
    with pytest.raises(ACIDInputError, match="profile_names"):
        Config(profile_names=names)


@pytest.mark.parametrize("species", ["Fe 1", [1], ["Fe 1", "Fe 1"], [""]])
def test_config_rejects_invalid_species_inputs(species):
    with pytest.raises(ACIDInputError, match="seperate_species"):
        Config(seperate_species=species)


def test_explicit_names_follow_group_ids_and_clipping():
    groups = np.r_[9, np.repeat([2, 7], 30), 9]
    lsd = run_grouped_lsd(profile_groups=groups, profile_names={2: "Iron", 7: "Other", 9: "Outside"})
    assert lsd.data.profile_names == {2: "Iron", 7: "Other"}
    restored = Data().from_dict(lsd.data.to_dict())
    assert restored.profile_names == lsd.data.profile_names
    for names in [["Iron", "Other"], {2: "Iron"}, {2: "Iron", 7: "Other", 8: "Unknown"}]:
        with pytest.raises(ACIDInputError, match="must match"):
            run_grouped_lsd(profile_groups=groups, profile_names=names)


def test_depth_groups_get_names_from_actual_depths():
    lsd = run_grouped_lsd()
    lsd.config.depth_group_rules = {"n_groups": 2, "min_lines": 20}
    # Reuse the same clipped line list while allowing automatic grouping.
    lsd.run_LSD(np.linspace(5000, 5020, 2001), np.ones(2001), np.full(2001, .01), 100.)
    depths = lsd.data.linelist["depths"][lsd.ll_mask]
    for group, name in lsd.data.profile_names.items():
        selected = depths[lsd.profile_groups == group]
        assert name == f"Depth {selected.min():.3g}–{selected.max():.3g}"


def test_species_groups_warn_after_clipping_and_keep_lines_in_remainder():
    with pytest.warns(ACIDProfileGroupsWarning, match="Ca 1.*0 lines"):
        lsd = run_grouped_lsd(seperate_species=["Fe 1", "Ca 1", "Ti 1"])
    assert lsd.data.profile_names == {0: "Everything else", 1: "Fe 1", 2: "Ti 1"}
    np.testing.assert_array_equal(lsd.profile_groups, np.repeat([1, 2, 0], 20))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ACIDProfileGroupsWarning)
        with pytest.raises(ACIDProfileGroupsWarning):
            run_grouped_lsd(seperate_species=["Missing"])


def test_species_remainder_and_minimum_line_count():
    with pytest.warns(ACIDProfileGroupsWarning):
        groups, names = LSD.group_profs_by_species(["Fe 1"] * 20 + ["Ti 1"] * 19, ["Fe 1", "Ti 1"])
    assert names == {0: "Everything else"}
    assert np.count_nonzero(groups == 0) == 39


def test_profiles_and_raw_plots_use_actual_group_names():
    x = np.linspace(-12, 12, 61)
    flux = np.array([Profiles.gaussian_func(x, -.25, 1.5, 2.2), Profiles.gaussian_func(x, -.3, -2., 3.)])
    data = Data()
    data.velocities = x
    data._profile_groups = np.array([2, 7])
    data.profile_names = {2: "Iron", 7: "Titanium"}
    data.profile["final"] = (flux, np.full_like(flux, .01), np.eye(flux.size) * .0001)
    data.profiles = [data.profile["final"]]
    data.complete = True
    fig, ax = data.plot.profiles(return_fig=True)
    assert ax.get_legend_handles_labels()[1] == ["Iron", "Titanium"]
    plt.close(fig)
    fig, axes = Profiles(data=data).plot_fit("gaussian", return_fig=True)
    labels = axes[0].get_legend_handles_labels()[1]
    assert "Iron Gaussian Fit" in labels and "Titanium Gaussian Fit" in labels
    plt.close(fig)
    with pytest.raises(ACIDInputError, match="must match"):
        Profiles(x, flux, profile_names={0: "Iron", 2: "Absent"})


def test_acid_accepts_profile_names_below_profile_groups(harps_order_40):
    wavelength, flux, errors, sn, velocities, linelist = harps_order_40
    runner = Acid(velocities=velocities, linelist=linelist, verbose=0)
    runner.ACID(wavelength, flux, errors, sn, depth_group_rules={"n_groups": 2, "min_lines": 1},
                profile_names=["Deep", "Shallow"], run_mcmc=False)
    assert runner.data.profile_names == {0: "Deep", 1: "Shallow"}


def test_species_grouping_loads_full_vald_metadata(tmp_path):
    path = tmp_path / "lines.txt"
    path.write_text("header\n" * 4 + "'Fe 1',5003,0,0,0,0,0,0,1.2,.2\n")
    data = Data()
    data.config.seperate_species = ["Fe 1"]
    data.linelist = str(path)
    assert data.linelist["spec_ion"].tolist() == ["Fe 1"]
    assert data.linelist["lande_factor"].tolist() == [1.2]


def grouped_orders(**kwargs):
    """Each order is too small for automatic grouping on its own."""
    wave = np.array([np.linspace(5000, 5010, 1001), np.linspace(5100, 5110, 1001)])
    lines = {"wavelengths": np.r_[np.linspace(5001, 5009, 24), 5050, np.linspace(5101, 5109, 24)],
             "depths": np.r_[np.linspace(.1, .5, 24), .99, np.linspace(.4, .8, 24)],
             "spec_ion": np.array(["Fe 1", "Other"] * 12 + ["Gap"] + ["Fe 1", "Other"] * 12),
             "lande_factor": np.ones(49)}
    return DataList(wave, np.ones_like(wave), np.full_like(wave, .01), np.full(2, 100.),
                    velocities=np.array([-2., 0., 2.]), linelist=lines, verbose=0,
                    poly_ord=1, masking_lines={}, parallel=False,
                    sigma_lower=100., sigma_upper=100., dev_perc=100., **kwargs)


@pytest.mark.parametrize("species", [False, True])
@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("source", ["kwargs", "config", "from_datalist", "run"])
def test_datalist_groups_across_orders_before_preparation(monkeypatch, species, joint, source):
    grouping = ({"seperate_species": ["Fe 1"]} if species else
                {"depth_group_rules": {"n_groups": 2, "min_lines": 20}})
    options = {"config": Config(**grouping)} if source == "config" else grouping if source == "kwargs" else {}
    orders = grouped_orders(**options)
    if source == "from_datalist":
        for data in orders:
            data.config.update_hipri(**grouping)
        orders = DataList.from_datalist(list(orders), verbose=0)
    method_name = "group_profs_by_species" if species else "group_profs_by_depth"
    grouper = getattr(LSD, method_name)
    calls = []

    def count_groups(*args, **kwargs):
        calls.append(len(args[0]))
        return grouper(*args, **kwargs)

    monkeypatch.setattr(LSD, method_name, staticmethod(count_groups))
    runner = orders.run_multi_order_ACID if joint else orders.run_ACID
    runner(run_mcmc=False, **(grouping if source == "run" else {}))
    assert calls == [48] # exclude the line in the wavelength gap
    lines = orders[0].linelist
    selected = np.r_[np.arange(24), np.arange(25, 49)]
    expected = (np.where(lines["spec_ion"][selected] == "Fe 1", 1, 0) if species else
                grouper(lines["depths"][selected], prof_group_rules=grouping["depth_group_rules"]))
    expected_full = np.zeros(49, dtype=int)
    expected_full[selected] = expected
    for data in orders:
        assert data.config.profile_groups is None # automatic settings remain editable
        np.testing.assert_array_equal(data.profile_groups, expected_full[data.ll_mask])
        restored = Data().from_dict(data.to_dict())
        np.testing.assert_array_equal(restored._automatic_profile_groups, expected_full)
        assert restored._automatic_profile_names == data._automatic_profile_names
    assert len(orders.profile_names) == 2
    if species:
        assert orders.profile_names == {0: "Everything else", 1: "Fe 1"}
    else:
        depths = lines["depths"][selected]
        assert orders.profile_names == {g: f"Depth {depths[expected == g].min():.3g}–{depths[expected == g].max():.3g}"
                                        for g in [0, 1]}
    if joint:
        np.testing.assert_array_equal(orders.multi_order_data.profile_groups, expected)


def test_datalist_species_counts_unique_overlapping_lines():
    orders = grouped_orders(seperate_species=["Fe 1"])
    lines = {"wavelengths": np.linspace(5001, 5009, 30), "depths": np.full(30, .3),
             "spec_ion": np.array(["Fe 1"] * 15 + ["Other"] * 15), "lande_factor": np.ones(30)}
    for data in orders:
        data.set_inputs(orders[0].wavelengths["input"].ravel(), np.ones(1001), np.full(1001, .01), 100.)
        data.linelist = lines
    with pytest.warns(ACIDProfileGroupsWarning, match="15 lines") as caught:
        orders.run_ACID(run_mcmc=False)
    assert len(caught) == 1
    assert orders.profile_names == {0: "Everything else"}
    assert all(np.all(data.profile_groups == 0) for data in orders)


def test_datalist_groups_before_splitting_workers_and_refreshes_cached_groups():
    orders = grouped_orders(depth_group_rules={"n_groups": 2, "min_lines": 20})
    orders.run_ACID(run_mcmc=False, worker=0, nworkers=2)
    first_groups = orders[0]._automatic_profile_groups.copy()
    assert orders[1]._automatic_profile_groups is None
    orders.run_ACID(run_mcmc=False, worker=1, nworkers=2)
    np.testing.assert_array_equal(orders[1]._automatic_profile_groups, first_groups)
    orders.run_ACID(run_mcmc=False, depth_group_rules={"n_groups": 1, "min_lines": 20}, overwrite_kwargs=True)
    assert all(data.alpha["initial"].shape[0] == 1 for data in orders)
    assert orders.profile_names == {0: "Depth 0.1–0.8"}


def test_datalist_grouping_respects_sn_cuts_and_explicit_groups():
    orders = grouped_orders(depth_group_rules={"n_groups": 2, "min_lines": 20})
    orders[0].sn["input"][:] = orders[0].sn["combined"] = 1.
    with pytest.raises(ACIDInputError, match="from 34 lines"):
        orders.run_ACID(run_mcmc=False)
    explicit = np.arange(49) % 2
    orders.run_ACID(run_mcmc=False, profile_groups=explicit, overwrite_kwargs=True)
    for data in orders:
        np.testing.assert_array_equal(data.profile_groups, explicit[data.ll_mask])
        assert data._automatic_profile_groups is None


def test_datalist_joint_species_names_allow_missing_groups_and_custom_names(tmp_path):
    orders = grouped_orders(seperate_species=["Fe 1"], profile_names=["Rest", "Iron"])
    orders.save_dir = str(tmp_path)
    for data in orders:
        data.linelist["spec_ion"][:] = ["Fe 1"] * 24 + ["Gap"] + ["Other"] * 24
    orders.run_multi_order_ACID(run_mcmc=False)
    assert orders[0].profile_names == {1: "Iron"}
    assert orders[1].profile_names == {0: "Rest"}
    assert orders.profile_names == {0: "Rest", 1: "Iron"}
    orders.save()
    restored = DataList.load(str(tmp_path), verbose=0)
    assert restored.profile_names == orders.profile_names
    assert restored[0].config.profile_groups is None
    np.testing.assert_array_equal(restored[0]._automatic_profile_groups, orders[0]._automatic_profile_groups)


def test_datalist_rejects_conflicting_automatic_grouping_settings():
    orders = grouped_orders(depth_group_rules={"n_groups": 2, "min_lines": 20})
    orders[1].config.depth_group_rules = {"n_groups": 3, "min_lines": 5}
    with pytest.raises(ACIDInputError, match="same automatic profile grouping settings"):
        orders.run_ACID(run_mcmc=False)


def test_datalist_rvs_keep_named_groups_aligned(completed_datalist, monkeypatch):
    datalist = completed_datalist
    names = {2: "Iron", 7: "Titanium"}
    for data, groups in zip(datalist, [[2], [2, 7], [7]]):
        flux, error, covariance = data.profile["final"]
        data.profile["final"] = (np.tile(flux, (len(groups), 1)), np.tile(error, (len(groups), 1)),
                                 np.kron(np.eye(len(groups)), covariance))
        data._profile_groups = np.array(groups)
        data.profile_names = {group: names[group] for group in groups}
    datalist._combined_profile = datalist[21].profile["final"]

    def fit_voigt(self, **kwargs):
        centres = [2.5 if self.profile_names[group] == "Iron" else -1.5 for group in self.profile_ids]
        parameters = np.tile([-.3, 0., 1., .1, 0.], (len(centres), 1))
        parameters[:, 1] = centres
        return parameters, np.tile(np.eye(5) * .04, (len(centres), 1, 1))

    monkeypatch.setattr(Profiles, "fit_voigt", fit_voigt)
    figure, axis = datalist.plot_all_rvs(return_fig=True)
    np.testing.assert_array_equal(axis.lines[0].get_xdata(), [20, 21])
    np.testing.assert_array_equal(axis.lines[1].get_xdata(), [21, 22])
    np.testing.assert_allclose(axis.lines[0].get_ydata(), 2.5)
    np.testing.assert_allclose(axis.lines[1].get_ydata(), -1.5)
    labels = axis.get_legend_handles_labels()[1]
    assert any("Iron RV" in label for label in labels)
    assert any("Titanium RV" in label for label in labels)
    plt.close(figure)
