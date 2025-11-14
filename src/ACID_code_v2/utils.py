import numpy as np
import glob
import scipy.constants as const
ckms = float(const.c/1e3)

def validate_args(x, i, allow_none=False, sn=False):
    """Validates the input arguments. This function can be used to ensure inputs to ACID
    are of the correct type and shape. It is performed automatically in ACID.

    Parameters
    ----------
    x : array_like
        array or list to be validated
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
        raise TypeError(f"Input in position {i} must be a list or numpy array")
    if isinstance(x, list):
        if len(x) == 0:
            raise TypeError(f"Input list in position {i} is empty")
    x = np.array(x)
    if x.ndim > 2:
        raise TypeError(f"Input in position {i} must be a list or numpy array with at most two dimensions")
    elif x.ndim == 0:
        raise TypeError(f"Input in position {i} must be a list or numpy array with at least one dimension")
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

def scale_spectra(wavelength, spectrum, error):
    """Scales the input spectrum and error to be between 0 and 1, masking any non-positive values by making
    their flux equal to 1 and their error very large (1e12). This is done so that the alpha matrix calculation
    maintains even spacing between wavelength pixels. The flux must be positive regardless as ACID works in
    optical depth.

    Parameters
    ----------
    wavelength : array_like
        The wavelengths corresponding to the spectrum.
    spectrum : array_like
        The flux values of the spectrum.
    error : array_like
        The error values associated with the spectrum.

    Returns
    -------
    tuple
        A tuple containing the scaled wavelength, spectrum, and error arrays.
    """

    # Rescale spectrum and error
    fmax = np.max(spectrum)
    scaled_spec = (spectrum) / (fmax)
    scaled_error = error / (fmax)

    # Mask non-positive flux values by setting their flux to 1 and error to a very large number
    mask_idx = scaled_spec <= 0
    scaled_spec[mask_idx] = 1
    scaled_error[mask_idx] = int(1e12)

    return wavelength, scaled_spec, scaled_error

def calc_deltav(wavelengths):
    """Calculates velocity pixel size

    Calculates the velocity pixel size for the LSD velocity grid based off the spectral wavelengths.

    Args:
        wavelengths (array): Wavelengths for ACID input spectrum (in Angstroms).
        
    Returns:
        float: Velocity pixel size in km/s
    """
    resol = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    return resol / (wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2)) * ckms

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

def findfiles(directory, file_type):

    filelist_corrected = glob.glob('%s/*/*%s**A_corrected*.fits'%(directory, file_type)) #finding corrected spectra
    filelist = glob.glob('%s/*/*%s**A*.fits'%(directory, file_type)) #finding all A band spectra

    filelist_final = []
    filelist_corrected_set = set(filelist_corrected)

    for file in filelist: #filtering out corrected spectra
        if file not in filelist_corrected_set:
            filelist_final.append(file)

    return filelist_final

def od2flux(x):
    return np.exp(x)-1

def flux2od(x):
    return np.log(x+1)