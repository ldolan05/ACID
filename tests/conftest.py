"""Shared pytest fixtures for the ACID test suite."""
from pathlib import Path
import sys

import matplotlib
import numpy as np
import pytest
from astropy.io import fits


# Resolve every project-relative path from this file so tests do not depend on cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Import the working source tree directly and force non-interactive plotting in CI.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
matplotlib.use("Agg")


def pytest_addoption(parser):
    # Keep expensive integrations opt-in while retaining pytest's normal short run.
    parser.addoption("--long", action="store_true", default=False,
                     help="run integration tests marked as long")


def pytest_configure(config):
    # Register the marker explicitly so misspellings are visible in pytest warnings.
    config.addinivalue_line("markers", "long: slow, full-size integration test")


def pytest_collection_modifyitems(config, items):
    # An explicit --long includes every collected test without changing its marker.
    if config.getoption("--long"):
        return

    # Normal runs collect long tests for visibility, but skip them before execution.
    skip_long = pytest.mark.skip(reason="pass --long to run full-size integration tests")
    for item in items:
        if "long" in item.keywords:
            item.add_marker(skip_long)


@pytest.fixture
def synthetic_spectrum():
    """A small, noiseless spectrum suitable for fast LSD and MCMC tests."""
    # Use a dense wavelength grid but only three velocity bins to keep matrix work small.
    wavelengths = np.linspace(5000, 5010, 501)
    velocities = np.array([-5.0, 0.0, 5.0])

    # Two isolated lines are sufficient to test alpha construction and profile recovery.
    linelist = {"wavelengths": np.array([5003.0, 5007.0]),
                "depths": np.array([0.2, 0.3])}
    profile = 0.12 * np.exp(-(velocities / 7.0) ** 2)

    # Import after the source path is configured, avoiding package import during collection.
    from ACID_code import LSD

    # Forward-model the known optical-depth profile into an exact normalised spectrum.
    alpha = LSD.calc_alpha(wavelengths, linelist["wavelengths"],
                           linelist["depths"], velocities)
    flux = np.exp(-(alpha @ profile))

    # Constant errors and S/N make expected clipping and recovery deterministic.
    return wavelengths, flux, np.full_like(flux, 0.01), 100.0, velocities, linelist


@pytest.fixture
def sample_spectrum_path():
    # Keep legacy sample-file consumers pointed at the repository-level data directory.
    return PROJECT_ROOT / "data" / "sample_spec_1.fits"


@pytest.fixture(scope="session")
def linelist_path():
    # The full line list is immutable, so one path object can be shared for the session.
    return PROJECT_ROOT / "data" / "linelist.txt"


@pytest.fixture(scope="session")
def harps_paths():
    # All real observations now live in data/, never in the private science submodule.
    data_dir = PROJECT_ROOT / "data"

    # Keep the science frame, matching flat, and full s1d product together by role.
    return {
        "e2ds": data_dir / "HARPS.2007-08-29T00-24-57.238_e2ds_A.fits",
        "flat": data_dir / "HARPS.2007-08-28T21-11-56.678_flat_A.fits",
        "s1d": data_dir / "HARPS.2007-08-29T00-24-57.238_s1d_A.fits",
    }


def extract_harps_e2ds(e2ds_path, flat_path):
    """Extract BERV-corrected, blaze-corrected, continuum-normalised HARPS orders."""
    # Read both products inside context managers so FITS handles close immediately.
    with fits.open(e2ds_path) as e2ds, fits.open(flat_path) as flat:
        header = e2ds[0].header
        spectrum = np.asarray(e2ds[0].data, dtype=float)
        blaze = np.asarray(flat[0].data, dtype=float)

    # A matching flat must contain one blaze value for every science pixel.
    assert spectrum.shape == blaze.shape

    # HARPS stores one polynomial wavelength solution for every extracted order.
    degree = header["ESO DRS CAL TH DEG LL"]
    pixels = np.arange(spectrum.shape[1])
    wavelengths = np.empty_like(spectrum)
    for order in range(spectrum.shape[0]):
        # Header coefficients are ordered by order first, then polynomial degree.
        coefficients = [header[f"ESO DRS CAL TH COEFF LL{i + order * (degree + 1)}"]
                        for i in range(degree + 1)]
        wavelengths[order] = np.polyval(coefficients[::-1], pixels)

    # Remove the blaze response and place every order near a unit continuum.
    flux = spectrum / blaze
    flux /= np.nanpercentile(flux, 99, axis=1, keepdims=True)

    # Convert the per-order header S/N into an error array matching the flux shape.
    sn = np.array([header[f"HIERARCH ESO DRS SPE EXT SN{order}"]
                   for order in range(spectrum.shape[0])])
    errors = flux / sn[:, None]

    # Apply the observation's barycentric velocity correction to every wavelength.
    wavelengths *= 1 + header["ESO DRS BERV"] / 299792.458
    return wavelengths, flux, errors, sn


@pytest.fixture(scope="session")
def harps_order_40(harps_paths, linelist_path):
    """The real HARPS e2ds order used by the HD 189733 science workflow."""
    # Import locally so simple unit-test collection does not eagerly import ACID.
    from ACID_code import calc_deltav

    # Extract all orders once, then select the established order-40 regression input.
    wavelengths, flux, errors, sn = extract_harps_e2ds(harps_paths["e2ds"],
                                                        harps_paths["flat"])
    order = 40
    wave = wavelengths[order]

    # Match ACID's velocity spacing to the wavelength sampling of this real order.
    velocities = np.arange(-25, 25, calc_deltav(wave))
    return wave, flux[order], errors[order], sn[order], velocities, str(linelist_path)


@pytest.fixture(scope="session")
def harps_result(harps_order_40):
    """One short deterministic HARPS fit shared by Result and e2e tests."""
    # Import locally because only tests requesting this session fixture need a full run.
    from ACID_code import Acid

    # Reuse one deterministic fit across Result, MCMC, persistence, and plotting tests.
    wavelengths, flux, errors, sn, velocities, linelist = harps_order_40
    acid = Acid(velocities=velocities, linelist=linelist, verbose=0, seed=1)

    # Twelve steps exercise sampler-backed behavior without repeating a convergence run.
    return acid.ACID(wavelengths, flux, errors, sn, nsteps=100, parallel=False)
