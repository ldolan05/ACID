import numpy as np
from math import log10, floor
import glob

def validate_args(x, i, allow_none=False, sn=False):
    # Ensure inputs are lists, np.arrays are converted to lists
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
            raise TypeError(f"Input for sn in position {i} must be a 1D numpy array or list")
        return x # 2D array, return as is, code later does np.array(x) so no change
    else: # should not reach here, somehow ndim is negative
        raise ValueError(f"Input in position {i} has invalid (or negative?) number of dimensions ({x.ndim})")

def round_sig(x1, sig):
    return round(x1, sig-int(floor(log10(abs(x1))))-1)

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