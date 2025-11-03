import numpy as np
from math import log10, floor
import glob

def ensure_list(x, allow_none=False, sn=False):
    # Ensure inputs are lists, np.arrays are converted to lists
    if x is None:
        if allow_none:
            return None
        else:
            raise TypeError("Input must be a list or numpy array, not None")
    if not isinstance(x, (list, np.ndarray)):
        raise TypeError("Input must be a list or numpy array, not a string")
    if isinstance(x, list):
        if len(x) == 0:
            raise TypeError("Input list is empty")
        return x
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            raise TypeError("Input must be a list or numpy array with at least one dimension")
        elif x.ndim == 1:
            if sn:
                return x
            return [x]
        else:
            if sn:
                raise TypeError("Input for sn must be a 1D numpy array or list")
            return x

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