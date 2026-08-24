#%%
"""Integration tests for the real HARPS spectra included with ACID."""
import numpy as np
import pytest
from astropy.io import fits
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ACID_code import Acid, DataList, LineList
from conftest import extract_harps_e2ds


def _extract_s1d(s1d_path):
    """Extract and continuum-normalise the complete BERV-corrected HARPS s1d spectrum."""
    with fits.open(s1d_path) as s1d:
        header = s1d[0].header
        spectrum = np.asarray(s1d[0].data, dtype=float)

    wavelengths = header["CRVAL1"] + header["CDELT1"] * np.arange(spectrum.size)
    valid = np.isfinite(spectrum) & (spectrum > 0)
    flux = spectrum / np.nanpercentile(spectrum[valid], 99.5)
    errors = flux / 100
    return wavelengths * (1 + header["ESO DRS BERV"] / 299792.458), flux, errors


def test_harps_e2ds_datalist_workflow(tmp_path, harps_paths, linelist_path):
    """Run a real HARPS e2ds order through the public DataList workflow."""
    wavelengths, flux, errors, sn = extract_harps_e2ds(harps_paths["e2ds"], harps_paths["flat"])
    linelist_wavelengths, _ = LineList.validate_linelist(str(linelist_path))
    order = 40
    assert np.count_nonzero((linelist_wavelengths >= wavelengths[order].min()) &
                            (linelist_wavelengths <= wavelengths[order].max())) >= 10
    datalist = DataList(wavelengths[[order]], flux[[order]], errors[[order]], sn[[order]],
                        np.arange(-25, 25, 5.0), str(linelist_path), order_range=[order],
                        save_dir=str(tmp_path), nsteps=20, seed=1)
    datalist.run_ACID()

    assert datalist[order].exception is None
    assert datalist[order].complete
    assert datalist[order].profile["final"][0].shape == datalist.velocities.shape
    assert datalist[order].nsteps == 20


@pytest.mark.long
def test_harps_s1d_full_spectrum_runs_acid(harps_paths, linelist_path):
    """Run the complete s1d spectrum through ACID; enabled only with ``pytest --long``."""
    wavelengths, flux, errors = _extract_s1d(harps_paths["s1d"])

    # The full spectrum has bad edges, so we cut the first 20% and last 20% of the data:
    mask = (wavelengths > np.percentile(wavelengths, 20)) & (wavelengths < np.percentile(wavelengths, 80))
    wavelengths, flux, errors = wavelengths[mask], flux[mask], errors[mask]
    acid = Acid(velocities=np.arange(-25, 25, 1.0), linelist=str(linelist_path))
    result = acid.ACID(wavelengths, flux, errors, nsteps=12, nwalkers=12)

    assert result.data.complete
    assert result["profile"].shape == acid.data.velocities.shape


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "--long"]))
