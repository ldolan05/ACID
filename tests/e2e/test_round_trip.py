#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code import Acid, Profiles


def test_harps_public_workflow_returns_completed_result(harps_result, harps_order_40):
    """The reusable order-40 fit is the public end-to-end ACID workflow."""
    _, _, _, _, velocities, _ = harps_order_40

    # The final profile and covariance are the main public output of ACID.
    assert harps_result.data.complete
    assert harps_result["profile"].shape == velocities.shape
    assert harps_result["covariance"].shape == (len(velocities), len(velocities))


@pytest.mark.long
def test_harps_non_deterministic_sampling_until_convergence_limit(harps_order_40):
    """Cover the legacy full-profile sampler and custom moves without slowing normal runs."""
    wavelengths, flux, errors, sn, _, linelist = harps_order_40
    velocities = np.arange(-25, 25, 5.0)
    result = Acid(velocities=velocities, linelist=linelist, verbose=0, seed=1).ACID(
        wavelengths, flux, errors, sn, deterministic_profile=False, max_steps=120,
        check_interval=20, min_checks=1, min_tau_factor=1, tau_tol=1.0,
        moves=[("StretchMove", 0.6, {}), ("DEMove", 0.4)], parallel=False,
        n_bins=8)

    assert result.data.complete
    assert 0 < result.data.nsteps <= 120
    figures = [result.plot_autocorrelation(return_fig=True, min_steps=20)[0],
               result.plot_acf(return_fig=True, max_lag=20)[0]]
    for figure in figures:
        assert figure.axes
        plt.close(figure)

@pytest.mark.long
def test_multiple_frames(pysme_synthetic_spectrum_multiple_frames):
    """The public workflow should accept multiple frames and return a single profile."""
    wavelengths, spectra, errors, sns, velocities, linelist = pysme_synthetic_spectrum_multiple_frames
    acid = Acid(velocities=velocities, linelist=linelist, verbose=2, seed=1)
    result = acid.ACID(wavelengths, spectra, errors, sns, max_steps=5000)

    assert result.data.complete
    assert result.data.profile["final"][0].shape == result.data.velocities.shape
    assert len(result.data.profiles) == len(spectra)

    # We can assume the final profile looks ok if we can fit a gaussian with no errors:
    _popt = Profiles(data=result.data).fit_gaussian()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--long"]))
