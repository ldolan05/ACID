#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code import Config, Data, DataList


def test_datalist_rejects_unset_orders_before_building_paths(tmp_path, completed_datalist):
    data = Data()
    assert data.config.order is None
    with pytest.raises(ValueError, match="explicit order"):
        DataList.from_datalist([data], verbose=0)
    with pytest.raises(ValueError, match="explicit order"):
        completed_datalist.append(data, extend=True)
    with pytest.raises(ValueError, match="Cannot determine"):
        DataList._update_paths_for_data(data, str(tmp_path))
    assert not (tmp_path / "order_None").exists()
    completed_datalist.append(data, force_order=23, extend=True)
    assert completed_datalist[23] is data


def test_datalist_chi_squared_plot_uses_completed_orders():
    # Build completed Data objects directly: plotting needs results, not another MCMC run.
    velocities = np.array([-1.0, 0.0, 1.0])
    data_list = []
    for order, scale in enumerate((1.0, 2.0), start=10):
        data = Data()
        data.config = Config(order=order)
        data.velocities = velocities
        data.profile["final"] = (np.ones(3), np.full(3, 0.01), np.eye(3) * 1e-4)
        data.flux["final"] = np.array([1.0, 0.9, 1.1])
        data.forward_y["final"] = data.flux["final"] - 0.01 * scale
        data.errors["final"] = np.full(3, 0.01)
        data_list.append(data)

    datalist = DataList.from_datalist(data_list)
    fig, ax = datalist.plot_chi2(return_fig=True)

    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [10, 11])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [3.0, 12.0])
    plt.close(fig)


def test_datalist_order_mapping_append_and_range_management(completed_datalist):
    # The class sorts by physical order and supports indexing through that order label.
    datalist = completed_datalist
    assert datalist.orders.tolist() == [20, 21, 22]
    assert datalist.i2o == {0: 20, 1: 21, 2: 22}

    # Duplicates require an explicit overwrite rather than silently replacing data.
    duplicate = Data()
    duplicate.config = Config(order=21)
    duplicate.velocities = datalist.velocities
    with pytest.raises(ValueError, match="already exists"):
        datalist.append(duplicate)
    datalist.append(duplicate, overwrite=True)
    assert datalist[21] is duplicate

    # Extending the range allows a newly observed order to be added safely.
    new_order = Data()
    new_order.config = Config(order=23)
    new_order.velocities = datalist.velocities
    datalist.append(new_order, extend=True)
    assert datalist.orders.tolist() == [20, 21, 22, 23]
    assert datalist.order_range.tolist() == [20, 21, 22, 23]

    # Shrinking away an existing order must be rejected to avoid losing its mapping.
    with pytest.raises(ValueError, match="subset"):
        datalist.set_order_range(np.array([20, 21]))


def test_datalist_callable_and_container_protocol(monkeypatch, completed_datalist):
    # Iteration, length, string output, and order indexing form the basic container API.
    datalist = completed_datalist
    assert len(list(datalist)) == len(datalist) == 3
    assert datalist[20].config.order == 20
    assert "20" in str(datalist)

    # Calling a DataList is documented as a direct forwarding route to run_ACID.
    received = {}

    def fake_run(self, *args, **kwargs):
        received["args"] = args
        received["kwargs"] = kwargs
        return "forwarded"

    monkeypatch.setattr(DataList, "run_ACID", fake_run)
    assert datalist([20], overwrite=True) == "forwarded"
    assert received == {"args": ([20],), "kwargs": {"overwrite": True}}


def test_datalist_setter_and_from_datalist_validate_members_and_velocities(completed_datalist):
    # The property accepts only a list containing Data objects.
    with pytest.raises(ValueError, match="must be a list"):
        completed_datalist.data_list = completed_datalist[20]
    with pytest.raises(ValueError, match="instances of the Data"):
        completed_datalist.data_list = [object()]

    # Every order must have a unique label and share one velocity grid.
    duplicate = Data().from_dict(completed_datalist[20].to_dict())
    with pytest.raises(ValueError, match="unique"):
        DataList.from_datalist([completed_datalist[20], duplicate])
    changed_velocity = Data().from_dict(completed_datalist[21].to_dict())
    changed_velocity.velocities = changed_velocity.velocities + 0.1
    with pytest.raises(ValueError, match="same velocity grid"):
        DataList.from_datalist([completed_datalist[20], changed_velocity])


def test_datalist_combines_profiles_and_exposes_all_diagnostics(completed_datalist):
    # Combine the fabricated final profiles in the same optical-depth space as production.
    datalist = completed_datalist
    datalist.combine_profiles(exclude=22)
    profile, errors, covariance = datalist.combined_profile

    assert profile.shape == datalist.velocities.shape
    assert errors.shape == profile.shape
    assert covariance.shape == (len(profile), len(profile))
    assert datalist.excluded_orders == [22]

    # Each diagnostic should work from stored results without invoking ACID again.
    figures = [datalist.plot_combined_profile(return_fig=True)[0],
               datalist.plot_all_profiles(return_fig=True)[0],
               datalist.plot_mean_profile_errors(return_fig=True)[0],
               datalist.plot_chi2(return_fig=True)[0],
               datalist.fit_profile(return_fig=True)[0]]
    for figure in figures:
        assert figure.axes
        plt.close(figure)


def test_datalist_combination_validation_and_lazy_property(completed_datalist):
    # Accessing combined_profile should calculate it once when no stored combination exists.
    datalist = completed_datalist
    assert datalist._combined_profile is None
    profile = datalist.combined_profile
    assert datalist._combined_profile is profile

    # Exclusions must refer to real instrument orders, and all orders cannot be removed.
    with pytest.raises(ValueError, match="available orders"):
        datalist.combine_profiles(exclude=[99])
    with pytest.raises(ValueError):
        datalist.combine_profiles(exclude=datalist.orders)



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--tb=auto"]))
