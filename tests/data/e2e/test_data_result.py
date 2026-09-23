#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from ACID_code import Config, Data


def test_data_properties_and_result_view(harps_result):
    # Rebuild through the dictionary format used by Data.save and DataList packing.
    payload = harps_result.data.to_dict()
    rebuilt = Data().from_dict(payload)

    assert isinstance(rebuilt.config, Config)
    np.testing.assert_array_equal(rebuilt.velocities, harps_result.data.velocities)
    assert rebuilt.result.data is rebuilt
    assert "Number of velocity points" in repr(rebuilt)

    # The sampler setter accepts an emcee backend and can explicitly discard it again.
    rebuilt.sampler = harps_result.sampler.backend
    np.testing.assert_array_equal(rebuilt.sampler.get_chain(), harps_result.sampler.get_chain())
    rebuilt.sampler = None
    assert rebuilt.sampler is None

    # Incomplete and failed data do not expose misleading Result objects.
    incomplete = Data()
    assert incomplete.result is None
    incomplete.exception = RuntimeError("failed")
    assert incomplete.result is None


def test_data_residual_masking_plot_uses_stored_acid_intermediates(harps_result):
    # The completed shared result already holds every residual-mask diagnostic array.
    before = set(plt.get_fignums())
    harps_result.data.plot_residual_masking()
    created = set(plt.get_fignums()) - before

    # Residuals, masked profile, and forward model are drawn as three separate figures.
    assert len(created) == 3
    for figure_number in created:
        plt.close(figure_number)



if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--tb=auto"]))
