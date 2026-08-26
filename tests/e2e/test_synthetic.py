#%%
import numpy as np
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from ACID_code import Acid, Profiles

@pytest.mark.long
def test_multiple_frames(pysme_synthetic_spectrum_multiple_frames):
    """The public workflow should accept multiple frames and return a single profile."""
    wavelengths, spectra, errors, sns, velocities, linelist = pysme_synthetic_spectrum_multiple_frames
    # velocities is none in this dataset
    acid = Acid(velocities=velocities, linelist=linelist, seed=1)
    result = acid.ACID(wavelengths, spectra, errors, sns, max_steps=5000)

    assert result.data.complete
    assert result.data.profile["final"][0].shape == result.data.velocities.shape
    assert len(result.data.profiles) == len(spectra)

    # We can assume the final profile looks ok if we can fit a gaussian with no errors:
    _popt = Profiles(data=result.data).fit_gaussian()

@pytest.mark.long
def test_final_profile_is_sensible_to_input_velocities(pysme_synthetic_spectrum):
    """The final profile should be sensitive to the input velocities."""
    wavelengths, spectra, errors, sns, velocities, linelist = pysme_synthetic_spectrum
    acid = Acid(velocities=velocities, linelist=linelist, seed=1)
    result = acid.ACID(wavelengths, spectra, errors, sns, max_steps=5000)

    assert result.data.complete
    profile = result.data.profile["final"][0]
    error = result.data.profile["final"][1]
    # The final profile is very well constrained for this dataset,
    # check ACID always produces this
    assert np.all(np.isfinite(profile))
    assert np.max(profile) < 1.1
    assert np.min(profile) > 0.5
    assert max(error) < 0.1

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--long"]))
