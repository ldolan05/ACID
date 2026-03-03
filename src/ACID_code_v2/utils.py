"""
Utility functions for the ACID package. Some functions may not be useful to the user.
"""

from beartype import beartype
import numpy as np
import glob
import scipy.constants as const
import astropy.units as u
from astropy.nddata import StdDevUncertainty
c_kms = float(const.c/1e3)

def validate_args(x, i, allow_none=False, sn=False):
    """Validates the input arguments. This function can be used to ensure inputs to Acid
    are of the correct type and shape. It is performed automatically in Acid.

    Parameters
    ----------
    x : array_like
        array, list, or int to be validated
    i : int
        position of the input argument
    allow_none : bool, optional
        Whether None is allowed as a valid input, by default False
    sn : bool, optional
        Whether the input is a signal-to-noise ratio array, by default False
    Returns
    -------
    array
        The validated and converted numpy array.

    Raises
    ------
    TypeError
        If any of the conditions on inputs are not met.
    """
    if x is None:
        if allow_none:
            return None
        else:
            raise TypeError(f"Argument in position {i} must be a list or numpy array, not None")
    if not isinstance(x, (list, np.ndarray)):
        if sn is False:
            raise TypeError(f"Argument in position {i} must be a list or numpy array")
    if isinstance(x, list):
        if len(x) == 0:
            raise TypeError(f"Argument list in position {i} is empty")
    x = np.array(x)
    if x.ndim > 2:
        raise TypeError(f"Argument in position {i} must be a list or numpy array with at most two dimensions")
    elif x.ndim == 0:
        if sn is False:
            raise TypeError(f"Argument in position {i} must be a list or numpy array with at least one dimension")
        else:
            return np.array([x]) # ensure sn is always 1D
    elif x.ndim == 1:
        if sn:
            return x
        return np.array([x])
    elif x.ndim == 2:
        if sn:
            if x.shape[0] != 1: # ie if 1, an extra [] was added to input to make 2D.
                raise TypeError(f"Argument for sn in position {i} must be a 1D numpy array or list (the input was 2D)")
            else:
                return x[0]
        return x # 2D array, return as is, code later does np.array(x) so no change
    else: # should not reach here, somehow ndim is negative
        raise ValueError(f"Argument in position {i} has invalid (or negative?) number of dimensions ({x.ndim})")

def mask_invalid(wavelengths, flux, errors, return_mask=False):
    """Masks any pixels where the wavelength, flux, or error is infinite or <= 0.
    Replaces bad pixels with NaN, which ACID can handle.

    Parameters
    ----------
    wavelengths : array_like
        The wavelength values of the spectrum.
    flux : array_like
        The flux values of the spectrum.
    errors : array_like
        The error values associated with the spectrum.

    Returns
    -------
    tuple
        A tuple containing the cleaned wavelength, flux, and error arrays.
    """
    mask = (
        np.isfinite(wavelengths) &
        np.isfinite(flux) &
        np.isfinite(errors) &
        (flux > 0) &
        (errors > 0)
    )
    fill_value = np.nan

    w = np.where(mask, wavelengths, fill_value)
    f = np.where(mask, flux, fill_value)
    e = np.where(mask, errors, fill_value)
    if return_mask:
        return w, f, e, mask
    return w, f, e

def drop_invalid(wavelengths, flux, errors, return_mask=False):
    """Drops any pixels where the wavelength, flux, or error is infinite or <= 0.

    Parameters
    ----------
    wavelengths : array_like
        The wavelength values of the spectrum.
    flux : array_like
        The flux values of the spectrum.
    errors : array_like
        The error values associated with the spectrum.

    Returns
    -------
    tuple
        A tuple containing the cleaned wavelength, flux, and error arrays.
    """
    mask = (
        np.isfinite(wavelengths) &
        np.isfinite(flux) &
        np.isfinite(errors) &
        (flux > 0) &
        (errors > 0)
    )
    w = wavelengths[mask]
    f = flux[mask]
    e = errors[mask]
    if return_mask:
        return w, f, e, mask
    return w, f, e

@beartype
def calc_deltav(wavelengths:np.ndarray) -> float:
    """Calculates velocity pixel size

    Calculates the velocity pixel size for the LSD velocity grid based off the spectral wavelengths.

    Args:
        wavelengths (np.ndarray): Wavelengths for Acid input spectrum (in Angstroms), must be a 1D array of positive values.

    Returns:
        float: Velocity pixel size in km/s
    """
    if wavelengths.ndim != 1:
        raise ValueError("Input wavelengths must be a 1D array.")
    if np.any(wavelengths <= 0):
        raise ValueError("Wavelengths must be positive.")
    return c_kms * np.nanmean(np.diff(np.log(wavelengths)))

def guess_SNR(
        frame_wavelengths : np.ndarray | list,
        frame_flux        : np.ndarray | list,
        frame_errors      : np.ndarray | list
        ) -> np.ndarray:
    """Estimates S/N for each frame. Takes the median S/N in the central third of the
    wavelength range. Fully vectorized so that all the frames can be passed at once.

    Parameters
    ----------
    frame_wavelengths : np.ndarray | list
        Array/list of wavelengths for each frame.
    frame_flux : np.ndarray | list
        Array/list of flux values for each frame.
    frame_errors : np.ndarray | list
        Array/list of error values for each frame.

    Returns
    -------
    np.ndarray
        Array of estimated signal-to-noise ratios for each frame.
    """

    # Quick validation check and conversion to numpy arrays
    frame_wavelengths, frame_flux, frame_errors = [
        validate_args(arg, i) for i, arg in enumerate((frame_wavelengths, frame_flux, frame_errors))]

    # Calculate S/N in central third of wavelength range
    wavelength_upper = np.nanmax(frame_wavelengths, axis=-1)
    wavelength_lower = np.nanmin(frame_wavelengths, axis=-1)
    delta_wavelength = wavelength_upper - wavelength_lower
    upper_cut = wavelength_lower + delta_wavelength * (2/3)
    lower_cut = wavelength_lower + delta_wavelength * (1/3)

    # Keep shape; drop out-of-band values as NaN
    mask = (frame_wavelengths > lower_cut[:, None]) & (frame_wavelengths < upper_cut[:, None])

    frame_wavelengths = np.where(mask, frame_wavelengths, np.nan)
    frame_flux = np.where(mask, frame_flux, np.nan)
    frame_errors = np.where(mask, frame_errors, np.nan)

    median = np.nanmedian(frame_flux, axis=-1)
    median_error = np.nanmedian(frame_errors, axis=-1)

    return np.abs(median / median_error)

def get_normalisation_coeffs(wl):
    """Calculates normalization coefficients for wavelength array

    Parameters
    ----------
    wl : array_like
        Wavelength array for which normalization coefficients are calculated.

    Returns
    -------
    tuple
        A tuple containing the normalization coefficients (a, b).
    """
    a = 2 / (np.nanmax(wl)-np.nanmin(wl))
    b = 1 - a * np.nanmax(wl)
    return a, b

def set_dict_defaults(input_dict, default_dict):
    """Sets default values in a dictionary if they are not already present.

    Parameters
    ----------
    input_dict : dict | None
        The dictionary to set defaults in (or none if not provided).
    default_dict : dict
        The dictionary containing default key-value pairs.
    
    Returns
    -------
    dict
        The updated dictionary with defaults set.
    """
    input_dict = dict(input_dict or {})
    for key, value in default_dict.items():
        input_dict.setdefault(key, value)
    return input_dict

def findfiles(directory, file_type):

    filelist_corrected = glob.glob('%s/*/*%s**A_corrected*.fits'%(directory, file_type)) #finding corrected spectra
    filelist = glob.glob('%s/*/*%s**A*.fits'%(directory, file_type)) #finding all A band spectra

    filelist_final = []
    filelist_corrected_set = set(filelist_corrected)

    for file in filelist: #filtering out corrected spectra
        if file not in filelist_corrected_set:
            filelist_final.append(file)

    return filelist_final

def robust_mean(data:np.ndarray, nsig:int|float=1, axis:int=0) -> np.ndarray|float:
    """Calculates the robust mean of the input data by excluding outliers beyond a
    specified number of standard deviations from the median.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    nsig : int | float, optional
        Number of standard deviations to use for outlier rejection. Defaults to 1.
    axis : int, optional
        Axis along which to compute the robust mean. Defaults to 0.

    Returns
    -------
    float
        Robust mean of the input data.
    """
    median = np.median(data, axis=axis)
    std = np.std(data, axis=axis)
    mask = np.abs(data - median) < nsig * std
    robust_data = np.where(mask, data, np.nan)
    return np.nanmean(robust_data, axis=axis)

def flux_to_od(flux=None, errors=None, linelist=None):
    """Converts flux, errors, and linelist to optical depth.

    Parameters
    ----------
    flux : np.ndarray
        Input flux array.
    errors : np.ndarray, optional
        Input errors array. Defaults to None.
    linelist : np.ndarray, optional
        Input linelist array. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the flux in optical depth, errors in optical depth,
        and linelist in optical depth. The tuple length depends on which inputs were provided.
    """
    out = []

    if flux is not None:
        flux_od = -np.log(flux)
        out.append(flux_od)
    else:
        flux_od = None

    if errors is not None:
        if flux_od is None:
            raise ValueError("'flux' must be provided if 'errors' is provided.")
        out.append(errors / flux)

    if linelist is not None:
        out.append(-np.log(1 - linelist))

    return tuple(out)

def od_to_flux(od=None, errors=None, linelist=None):
    """Converts optical depth to flux, errors, and linelist.

    Parameters
    ----------
    od : np.ndarray
        Input optical depth array.
    errors : np.ndarray, optional
        Input errors array. Defaults to None.
    linelist : np.ndarray, optional
        Input linelist array. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the flux, errors, and linelist. The tuple length depends on which inputs were provided.
    """
    out = []

    if od is not None:
        flux = np.exp(-od)
        out.append(flux)
    else:
        flux = None

    if errors is not None:
        if flux is None:
            raise ValueError("'od' must be provided if 'errors' is provided.")
        out.append(errors * flux)

    if linelist is not None:
        out.append(1-np.exp(-linelist))

    return tuple(out)

def configure_mp_environ(os):
    """Configures the multiprocessing environment variables for optimal performance.

    Parameters
    ----------
    os : module
        The os module to use for setting environment variables.
    """
    slurm = "SLURM_JOB_ID" in os.environ
    if slurm:
        if os.getenv("OMP_NUM_THREADS") != "1" or os.getenv("MKL_NUM_THREADS") != "1":
            raise ValueError(f"In a SLURM environment, OMP_NUM_THREADS and MKL_NUM_THREADS must be set to 1 before any imports for parallel MCMC. \n" \
            "Please set this in your SLURM job script or at the top of your python script before any other imports.\n" \
            "See https://acid-v2.readthedocs.io/en/latest/using_ACID.html#multiprocessing for more information.")
    else:
        # Emcee recommendation, after testing this is absolutely a requirement
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

def next_pow_2(n):
    """Calculates the next power of 2 greater than or equal to n.

    Parameters
    ----------
    n : int
        Input number. Must be real and non-negative.

    Returns
    -------
    int
        The next power of 2 greater than or equal to n.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    return 1 if n == 0 else 2**(n - 1).bit_length()

def auto_window(taus: np.ndarray, c: float = 5.0):
    """
    Automated windowing procedure following Sokal (1989) in emcee documentation.
    Returns the window index to use in taus[window].
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

def autocorr_func_1d(x, norm=True):
    """
    Autocorrelation estimate using FFT from the emcee tutorial.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_2(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def autocorr_gw2010(y, c=5.0):
    # Goodman & Weare (2010) autocorrelation estimate from emcee documentation
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    # "New" integrated autocorrelation time estimate from emcee documentation
    # Average ACF across walkers
    # Apply sokal windowing on cumulative sum
    assert y.ndim == 2, "Expects y with shape (nwalkers, nsteps)"
    nwalkers, nsteps = y.shape

    f = np.zeros(nsteps)
    for walker in range(nwalkers):
        f += autocorr_func_1d(y[walker], norm=True)
    f /= nwalkers
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return float(taus[window])