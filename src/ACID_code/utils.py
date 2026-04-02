"""
All of the utility functions for the ACID package. Some functions may not be useful to the user.
"""
from __future__ import annotations
from beartype import beartype
from beartype.vale import IsAttr, IsEqual
import numpy as np
import glob, emcee
import scipy.constants as const
from typing import TypeAlias, Annotated
from numpy.typing import NDArray
c_kms = float(const.c/1e3)
FloatLike: TypeAlias = float | np.floating
IntLike: TypeAlias = int | np.integer
Scalar: TypeAlias = FloatLike | IntLike | Annotated[np.ndarray, IsAttr["ndim", IsEqual[0]]]
NumericArray: TypeAlias = NDArray[np.number]
Array1D: TypeAlias = Annotated[np.ndarray, IsAttr["ndim", IsEqual[1]]] | list[Scalar]
Array2D: TypeAlias = Annotated[np.ndarray, IsAttr["ndim", IsEqual[2]]] | list[list[Scalar]] | list[Array1D]
ArrayAnyD: TypeAlias = NumericArray | list

def convert_moves_to_emcee(moves:list[tuple]):
    """Converts a list of move specifications to emcee moves.

    Parameters
    ----------
    moves : list[tuple], optional
        A list of tuples specifying the moves for the MCMC sampler. The format
        tries to follow the emcee documentation as closely as possible.
        However, the config cannot store classes directly, so move names are
        used instead and converted when building the sampler.

        Each tuple should have the form::

            (move_name: str, fraction: float, move_kwargs: dict | None)

        where:

        - "move_name" is the name of the emcee move. Supported variants currently
          include "RedBlueMove", "StretchMove", "WalkMove",
          "KDEMove", "DEMove", "DESnookerMove", "MHMove",
          and "GaussianMove". Refer to the emcee documentation for more
          details on each move type. Input move names are checked against the
          "emcee.moves" module, so other moves from that module may also work,
          although not all have been tested with ACID.
        - "fraction" is the fraction of walkers to which this move should be applied.
        - "move_kwargs" is an optional dictionary of keyword arguments passed to
          the move class initialisation.

    Returns
    -------
    list
        A list of emcee move objects corresponding to the input specifications.
    """
    emcee_moves = []
    for move in moves:
        if len(move) == 2:
            move_name, fraction = move
            move_kwargs = {}
        elif len(move) == 3:
            move_name, fraction, move_kwargs = move
            if not isinstance(move_kwargs, dict):
                raise ValueError(
                    "Move kwargs must be a dictionary of keyword arguments to pass to " \
                    "the move class initialisation (if passed)."
                )
        else:
            raise ValueError(
                "Each move tuple must have length 2 or 3: " \
                "(move_name, fraction) or (move_name, fraction, kwargs).")

        if not hasattr(emcee.moves, move_name):
            raise ValueError(f"Move '{move_name}' is not a valid emcee move.")
        move_class = getattr(emcee.moves, move_name)
        emcee_moves.append((move_class(**move_kwargs), fraction))
        
    return emcee_moves

def mask_invalid(wavelengths, flux, errors, return_mask=False, verbose=2):
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

    if verbose > 1:
        num_invalid = np.size(wavelengths) - np.count_nonzero(mask)
        if num_invalid > 0:
            print(f"Your spectrum includes {num_invalid} out of {np.size(wavelengths)} non-positive/non-finite/nan values, which will be dropped when necessary, \n"
                  f"but it is still recommended to check your wavelength, spectrum and error arrays for bad pixels and make sure this is intentional.")

    if return_mask:
        return w, f, e, mask
    return w, f, e

def drop_invalid(wavelengths, flux, errors, return_mask=False, verbose=2):
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

    if verbose > 1:
        num_invalid = np.size(wavelengths) - np.count_nonzero(mask)
        if num_invalid > 0:
            print(f"Dropped {num_invalid} invalid pixels out of {np.size(wavelengths)} (non-finite or <= 0).")

    if return_mask:
        return w, f, e, mask
    return w, f, e

def clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist):
    """Clips the linelist to only include lines within the wavelength range of the observed spectrum.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelengths of the observed spectrum
    wavelengths_linelist : np.ndarray
        Wavelengths from the linelist
    depths_linelist : np.ndarray
        Depths from the linelist

    Returns
    -------
    wavelengths_linelist : np.ndarray
        Clipped wavelengths from the linelist
    depths_linelist : np.ndarray
        Clipped depths from the linelist
    """
    lower, upper = np.nanmin(wavelengths), np.nanmax(wavelengths)
    idx = (wavelengths_linelist >= lower) & (wavelengths_linelist <= upper)
    return wavelengths_linelist[idx], depths_linelist[idx]

@beartype
def calc_deltav(wavelengths:Array1D)->Scalar:
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
    wavelengths = np.sort(wavelengths)
    return c_kms * np.nanmean(np.diff(np.log(wavelengths)))

@beartype
def guess_SNR(
        frame_wavelengths : Array1D | Array2D,
        frame_flux        : Array1D | Array2D,
        frame_errors      : Array1D | Array2D
        ) -> np.ndarray:
    """Estimates S/N for each frame. Takes the median S/N in the central two-thirds of the
    wavelength range. Fully vectorized so that all the frames can be passed at once.

    Parameters
    ----------
    frame_wavelengths : Array1D | Array2D
        Array/list of wavelengths for each frame.
    frame_flux : Array1D | Array2D
        Array/list of flux values for each frame.
    frame_errors : Array1D | Array2D
        Array/list of error values for each frame.

    Returns
    -------
    np.ndarray
        Array of estimated signal-to-noise ratios for each frame.
    """
    if np.any(frame_flux) <= 0 or np.any(frame_errors) <= 0 or np.any(frame_wavelengths <= 0):
        raise ValueError("Flux, errors, and wavelengths must all be positive non-zero to estimate S/N.")

    frame_wavelengths = np.atleast_2d(frame_wavelengths)
    frame_flux = np.atleast_2d(frame_flux)
    frame_errors = np.atleast_2d(frame_errors)

    lo, hi = np.nanpercentile(frame_wavelengths, [100/6, 500/6], axis=-1)
    mask = (frame_wavelengths > lo[:, None]) & (frame_wavelengths < hi[:, None])
    frame_flux = np.where(mask, frame_flux, np.nan)
    frame_errors = np.where(mask, frame_errors, np.nan)

    sn_per_pixel = frame_flux / frame_errors
    return np.nanpercentile(sn_per_pixel, 99, axis=-1).squeeze()

def collapse_SNR(sn, wavelengths):
    """Collapses the SN of a 1D or 2D wavelength and sn array to the median of the SNs
    on the central 2/3 wavelengths.
    """
    sn = np.atleast_2d(sn)
    wavelengths = np.atleast_2d(wavelengths)

    lo, hi = np.nanpercentile(wavelengths, [100/6, 500/6], axis=-1)
    mask = (wavelengths > lo[:, None]) & (wavelengths < hi[:, None])
    return np.nanmedian(np.where(mask, sn, np.nan), axis=-1).squeeze()

@beartype
def get_normalisation_coeffs(wl:Array1D)->tuple[Scalar, Scalar]:
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

def robust_mean(data:np.ndarray, nsig:int|float=3, axis:int=0) -> np.ndarray|float:
    """Calculates the robust mean of the input data by excluding outliers beyond a
    specified number of standard deviations from the median.

    Parameters
    ----------
    data : np.ndarray
        Input data array.
    nsig : int | float, optional
        Number of standard deviations to use for outlier rejection. Defaults to 3.
    axis : int, optional
        Axis along which to compute the robust mean. Defaults to 0.

    Returns
    -------
    float
        Robust mean of the input data.
    """
    median = np.median(data, axis=axis)
    mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
    sigma_nmad = 1.4826 * mad
    mask = np.abs(data - median) < nsig * sigma_nmad
    robust_data = np.where(mask, data, np.nan)
    return np.nanmean(robust_data, axis=axis)

@beartype
def combine_profiles(
    spectra : Array2D,
    errors  : Array2D,
    ) -> tuple[Array1D, Array1D]:
    
    spectra = np.asarray(spectra)
    errors = np.asarray(errors)

    weights = 1.0 / errors**2
    combined_spectrum = np.sum(weights * spectra, axis=0) / np.sum(weights, axis=0)
    combined_errors = np.sqrt(1.0 / np.sum(weights, axis=0))

    return combined_spectrum, combined_errors

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

    return tuple(out) if len(out) > 1 else out[0]

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

    return tuple(out) if len(out) > 1 else out[0]

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
            "See https://acid-code.readthedocs.io/en/latest/using_ACID.html#multiprocessing for more information.")
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