from __future__ import annotations
import numpy as np
from ..utils import Array1D, IntLike
import pandas as pd
from .config import Config

class LineList:
    """
    A class to expose the linelist when called in Data. Has validation methods and easy indexing for plotting and other uses.
    """
    __slots__ = ("ll",) # the only thing stored in this class is the linelist
    def __init__(self, ll: dict) -> None:
        self.ll = ll

    def __getitem__(self, k):
        if k == 0:
            return self.ll["wavelengths"]
        if k == 1:
            return self.ll["depths"]
        if isinstance(k, int):
            raise IndexError("LineList only has keys 0 and 1, or 'wavelengths' and 'depths'")
        return self.ll[k]  # allow "wavelengths"/"depths"

    def __iter__(self):
        yield self.ll["wavelengths"]
        yield self.ll["depths"]

    @staticmethod
    def validate_linelist(linelist) -> tuple[np.ndarray, np.ndarray]:
        """
        Validates the linelist according to the description in :py:class:`Acid`, and returns the linelist wavelengths 
        and depths as numpy arrays. This is used internally in the set_linelist method.

        Parameters
        ----------
        linelist : str, dict, LineList, list, or np.ndarray
            See :py:class:`Acid`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The validated linelist wavelengths and depths as numpy arrays.
        """
        # Run through every possible input type and issue, I'm not going to comment everything but the logic is fairly
        # self-explanatory, and the error messages should be helpful for debugging if the input is not in the correct format.
        if linelist is None:
            raise ValueError("A linelist must be provided. For possible inputs, see https://acid-code.readthedocs.io/en/stable/_api/ACID_code.Acid.html")

        # All loops below set linelist_wl and linelist_depths from their own type sof input
        elif isinstance(linelist, str):

            full_linelist = pd.read_csv(
                linelist,
                skiprows=4,
                delimiter=',',
                usecols=[1, 9],
                names=['wavelength', 'depth'],
                dtype=str,
                engine='python',
                on_bad_lines='skip'
            )

            # Clean whitespace / quotes
            full_linelist['wavelength'] = full_linelist['wavelength'].astype(str).str.strip()
            full_linelist['depth'] = full_linelist['depth'].astype(str).str.strip()

            # Convert numeric columns safely
            full_linelist['wavelength'] = pd.to_numeric(
                full_linelist['wavelength'],
                errors='coerce'
            )

            full_linelist['depth'] = pd.to_numeric(
                full_linelist['depth'],
                errors='coerce'
            )

            # Remove rows where numeric conversion failed
            full_linelist = full_linelist.dropna(subset=['wavelength', 'depth'])

            # Convert to NumPy arrays
            linelist_wl = full_linelist['wavelength'].to_numpy(dtype=float)
            linelist_depths = full_linelist['depth'].to_numpy(dtype=float)
        elif isinstance(linelist, LineList):
            linelist_wl = linelist[0]
            linelist_depths = linelist[1]
        elif isinstance(linelist, dict):
            if "wavelengths" not in linelist or "depths" not in linelist:
                raise ValueError("If 'linelist' is a dict, it must contain keys 'wavelengths' and 'depths'")
            linelist_wl = linelist["wavelengths"]
            linelist_depths = linelist["depths"]
        elif isinstance(linelist, (list, np.ndarray)):
            if len(linelist) != 2 and len(linelist) != 3:
                raise ValueError("If 'linelist' is a list or array, it must have length 2, with index 0 being wavelengths, and index 1 being depths")
            linelist_wl = linelist[0]
            linelist_depths = linelist[1]
        else:
            raise ValueError(f"'linelist' must be a string path to a VALD linelist, a dictionary with keys 'wavelengths' and 'depths', \n" \
            "a LineList object, or a list/array indexed such that 0 is wavelengths and 1 is depths.")

        # Convert to numpy arrays to ensure their dimensions are correct
        try:
            linelist_wl = np.array(linelist_wl)
            linelist_depths = np.array(linelist_depths)
        except Exception as e:
            raise ValueError(f"Failed to convert linelist inputs into numpy arrays with exception:\n{e}")
        if linelist_wl.ndim != 1 or linelist_depths.ndim != 1:
            raise ValueError("'wavelengths' and 'depths' must be one-dimensional arrays or lists")
        if linelist_wl.shape != linelist_depths.shape:
            raise ValueError("'wavelengths' and 'depths' must have the same length and shape, \n"
                             f" but have shapes: {linelist_wl.shape}, {linelist_depths.shape}")

        # Finally, sort the arrays by wavelength
        sort_idx = np.argsort(linelist_wl)
        linelist_wl = linelist_wl[sort_idx]
        linelist_depths = linelist_depths[sort_idx]

        return linelist_wl, linelist_depths

    @staticmethod
    def drop_invalid_lines(wavelengths:Array1D, depths:Array1D, return_mask:bool=False, verbose:IntLike|bool|str=None) -> tuple:
        """Removes NaN, non-finite, negative, and greater than 1 values from the wavelengths and depths arrays.
        This is used internally in the set_linelist method.

        Parameters
        ----------
        wavelengths : np.ndarray
            The array of linelist wavelengths.
        depths : np.ndarray
            The array of linelist depths.
        return_mask : bool, optional
            If True, also returns the boolean mask of valid lines. Default is False.
        verbose : int, bool, or str, optional
            The verbosity level for printing warnings about dropped lines. Same format as :py:class:`Acid`.
            Default is 2 as per config defaults.

        Returns
        -------
        tuple or np.ndarray
            If return_mask is True, returns a tuple of (wavelengths, depths, mask).
            Otherwise, returns a tuple of (wavelengths, depths) with invalid lines removed.
        """
        # Set verbose level using config verbose validation, handles a verbose=None input
        verbose = Config(verbose=verbose).verbose

        # Get mask
        mask = np.isfinite(wavelengths) & np.isfinite(depths)
        mask &= (depths >= 0) & (depths < 1)
        mask &= (wavelengths > 0)

        # Count the number of dropped lines for verbose output
        count_dropped = np.count_nonzero(~mask)
        if count_dropped == len(wavelengths):
            raise ValueError(f"All lines in the linelist are non-finite, nan, negative, or greater than 1.\n" \
            "Please check your linelist for invalid values.")
        if verbose >= 1 and count_dropped > 0:
            print(f"Your linelist includes {count_dropped} non-finite, nan, negative, or greater than 1 values.\n"
                  f"These will be removed, but it is still recommended to check your linelist for why this happened.")

        # Apply mask and return results
        if return_mask:
            return wavelengths[mask], depths[mask], mask
        return wavelengths[mask], depths[mask]
