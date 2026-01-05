"""
Utility functions for the ACID package. Some functions may not be useful to the user.
"""

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
            raise TypeError(f"Input in position {i} must be a list or numpy array, not None")
    if not isinstance(x, (list, np.ndarray)):
        if sn is False:
            raise TypeError(f"Input in position {i} must be a list or numpy array")
    if isinstance(x, list):
        if len(x) == 0:
            raise TypeError(f"Input list in position {i} is empty")
    x = np.array(x)
    if x.ndim > 2:
        raise TypeError(f"Input in position {i} must be a list or numpy array with at most two dimensions")
    elif x.ndim == 0:
        if sn is False:
            raise TypeError(f"Input in position {i} must be a list or numpy array with at least one dimension")
        else:
            return np.array([x])
    elif x.ndim == 1:
        if sn:
            return x
        return np.array([x])
    elif x.ndim == 2:
        if sn:
            if x.shape[0] != 1: # ie if 1, an extra [] was added to input to make 2D.
                raise TypeError(f"Input for sn in position {i} must be a 1D numpy array or list (the input was 2D)")
            else:
                return x[0]
        return x # 2D array, return as is, code later does np.array(x) so no change
    else: # should not reach here, somehow ndim is negative
        raise ValueError(f"Input in position {i} has invalid (or negative?) number of dimensions ({x.ndim})")

def scale_spectra(spectrum, error):
    """Scales the input spectrum and error to be between 0 and 1, masking any non-positive values by making
    their flux equal to 1 and their error very large (1e12). This is done so that the alpha matrix calculation
    maintains even spacing between wavelength pixels. The flux must be positive regardless as Acid works in
    optical depth.

    Parameters
    ----------
    spectrum : array_like
        The flux values of the spectrum.
    error : array_like
        The error values associated with the spectrum.

    Returns
    -------
    tuple
        A tuple containing the scaled spectrum, and error arrays.
    """

    # Rescale spectrum and error
    fmax = np.max(spectrum)
    scaled_spec = (spectrum) / (fmax)
    scaled_error = error / (fmax)

    # Mask non-positive flux values by setting their flux to 1 and error to a very large number
    mask_idx = scaled_spec <= 0
    scaled_spec[mask_idx] = 1
    scaled_error[mask_idx] = int(1e12)

    return scaled_spec, scaled_error

def calc_deltav(wavelengths):
    """Calculates velocity pixel size

    Calculates the velocity pixel size for the LSD velocity grid based off the spectral wavelengths.

    Args:
        wavelengths (array): Wavelengths for Acid input spectrum (in Angstroms).
        
    Returns:
        float: Velocity pixel size in km/s
    """
    resol = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    return resol / (wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2)) * c_kms

def guess_SNR(
        frame_wavelengths : np.ndarray | list,
        frame_flux        : np.ndarray | list,
        frame_errors      : np.ndarray | list
        ) -> np.ndarray:
    """Estimates S/N for each frame

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

    frame_wavelengths, frame_flux, frame_errors = [
        validate_args(arg, i) for i, arg in enumerate((frame_wavelengths, frame_flux, frame_errors))]

    wavelength_upper = np.max(frame_wavelengths, axis=-1)
    wavelength_lower = np.min(frame_wavelengths, axis=-1)
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
    a = 2 / (np.max(wl)-np.min(wl))
    b = 1 - a * np.max(wl)
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

def od2flux(x):
    return np.exp(x)-1

def flux2od(x):
    return np.log(x+1)