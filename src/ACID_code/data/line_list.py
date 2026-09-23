from __future__ import annotations
from ..diagnostics.errors import *
import numpy as np
from ..utils import Array1D
from ..diagnostics.warnings import *
import warnings
import pandas as pd

class LineList:
    """
    Read, validate, and store a linelist in wavelength order.

    Accepts the same linelist inputs as :py:class:`Acid`: a string path to a
    VALD file, a dictionary with "wavelengths" and "depths", a list or NumPy
    array indexed by wavelength/depth, or another :py:class:`LineList`.
    Invalid lines are removed with a warning. Arrays can be accessed by name,
    by indices 0 and 1, or by unpacking the object.
    With full=True, also read/require "spec_ion" and "lande_factor" (indices
    2 and 3). Metadata already present in in-memory inputs is preserved.
    """
    __slots__ = ("ll",) # the only thing stored in this class is the linelist
    def __init__(self, ll:str|dict|LineList|list|np.ndarray, full:bool=False) -> None:
        self.ll = self.validate_linelist(ll, full=full, return_dict=True)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "ll"), name)

    def __len__(self):
        return len(self.ll)

    def __getitem__(self, k):
        if isinstance(k, (int, np.integer)):
            keys = list(self.ll)
            if k < 0 or k >= len(keys):
                raise ACIDInputError(f"LineList only has indices 0 to {len(keys)-1}, or keys {keys}")
            return self.ll[keys[k]]
        return self.ll[k]  # allow column names

    def __iter__(self):
        yield from self.ll.values()

    @staticmethod
    def validate_linelist(linelist, return_mask:bool=False, *, full:bool=False, return_dict:bool=False) -> tuple|dict:
        """
        Validates the linelist according to the description in :py:class:`Acid`, and returns the linelist columns
        as numpy arrays. Used internally by the constructor; use ``LineList(linelist)`` for a fully
        validated and stored linelist.

        Parameters
        ----------
        linelist : str, dict, LineList, list, or np.ndarray
            See :py:class:`Acid`.
        return_mask : bool, optional
            Also return the validity mask in wavelength-sorted order, before invalid lines are removed.
            For files, rows with nonnumeric wavelengths/depths are discarded before this mask.
        full : bool, optional
            Read/require wavelengths, depths, spec_ion (strings), and lande_factor (floats), in that order.
            Metadata already present in in-memory inputs is preserved even when False.
        return_dict : bool, optional
            Return named columns instead of a tuple of arrays, for storage by the constructor.

        Returns
        -------
        tuple or dict
            The validated columns. If return_mask=True, append the mask to the tuple,
            or return (columns, mask) when return_dict=True.
        """
        # Column names, VALD indices, and types, in the public indexing/unpacking order.
        columns = {"wavelengths": (1, float), "depths": (9, float),
                   "spec_ion": (0, str), "lande_factor": (8, float)}
        required = list(columns) if full else ["wavelengths", "depths"]

        # Run through every possible input type and issue, I'm not going to comment everything but the logic is fairly
        # self-explanatory, and the error messages should be helpful for debugging if the input is not in the correct format.
        if linelist is None:
            raise ACIDInputError("A linelist must be provided. For possible inputs, see https://acid-code.readthedocs.io/en/stable/_api/ACID_code.Acid.html")

        # All loops below set linelist_columns from their own types of input
        elif isinstance(linelist, str):
            # pandas returns selected columns in file order, not usecols order.
            file_columns = sorted(required, key=lambda key: columns[key][0])
            full_linelist = pd.read_csv(
                linelist,
                skiprows=4,
                delimiter=',',
                usecols=[columns[key][0] for key in file_columns],
                names=file_columns,
                dtype=str,
                engine='python',
                on_bad_lines='skip'
            )

            # Clean whitespace / quotes
            for key in file_columns:
                full_linelist[key] = full_linelist[key].str.strip().str.strip("\"'").str.strip()

            # Convert numeric columns safely
            for key in file_columns:
                if columns[key][1] is float:
                    full_linelist[key] = pd.to_numeric(full_linelist[key], errors='coerce')
                else:
                    full_linelist[key] = full_linelist[key].fillna("")

            # Remove rows where numeric conversion failed
            full_linelist = full_linelist.dropna(subset=['wavelengths', 'depths'])

            # Convert to NumPy arrays
            linelist_columns = {key: full_linelist[key].to_numpy(dtype=columns[key][1]) for key in required}
        elif isinstance(linelist, LineList):
            linelist_columns = linelist.ll
        elif isinstance(linelist, dict):
            linelist_columns = linelist
        elif isinstance(linelist, (list, np.ndarray)):
            if len(linelist) not in (2, 3, 4):
                raise ACIDInputError("If 'linelist' is a list or array, it must have length 2, with index 0 being wavelengths, and index 1 being depths, or length 4 to also include spec_ion and lande_factor")
            # The legacy length-3 input ignores its third entry.
            keys = list(columns) if len(linelist) == 4 else ["wavelengths", "depths"]
            linelist_columns = dict(zip(keys, linelist))
        else:
            raise ACIDInputError(f"'linelist' must be a string path to a VALD linelist, a dictionary with keys 'wavelengths' and 'depths', \n" \
            "a LineList object, or a list/array indexed such that 0 is wavelengths and 1 is depths.")

        missing = [key for key in required if key not in linelist_columns]
        if missing:
            raise ACIDInputError(f"The linelist must contain keys {required}; missing {', '.join(missing)}")

        # Convert to numpy arrays to ensure their dimensions are correct
        try:
            linelist_columns = {key: np.array(linelist_columns[key], dtype=dtype)
                                for key, (_, dtype) in columns.items() if key in linelist_columns}
        except Exception as e:
            raise ACIDInputError(f"Failed to convert linelist inputs into numpy arrays with exception:\n{e}")
        if any(values.ndim != 1 for values in linelist_columns.values()):
            raise ACIDInputError("Linelist columns must be one-dimensional arrays or lists")
        if any(values.shape != linelist_columns["wavelengths"].shape for values in linelist_columns.values()):
            raise ACIDInputError("Linelist columns must have the same length and shape, \n"
                             f" but have shapes: {[values.shape for values in linelist_columns.values()]}")

        # Finally, sort the arrays by wavelength
        sort_idx = np.argsort(linelist_columns["wavelengths"], kind="stable")
        linelist_columns = {key: values[sort_idx] for key, values in linelist_columns.items()}

        # Drop invalid lines
        wavelengths = linelist_columns["wavelengths"]
        depths = linelist_columns["depths"]
        mask = np.isfinite(wavelengths) & np.isfinite(depths)
        mask &= (depths >= 0) & (depths < 1)
        mask &= (wavelengths > 0)
        for key in linelist_columns:
            if key not in ("wavelengths", "depths"):
                if columns[key][1] is float:
                    mask &= np.isfinite(linelist_columns[key])
                else:
                    mask &= ~np.isin(np.char.strip(linelist_columns[key]), ["", "None", "nan"])

        # Count the number of dropped lines
        count_dropped = np.count_nonzero(~mask)
        if count_dropped == len(wavelengths):
            raise ACIDInputError(f"All lines in the linelist are non-finite, nan, negative, greater than 1, or have invalid metadata.\n" \
            "Please check your linelist for invalid values.")
        if count_dropped > 0:
            warnings.warn(f"Your linelist includes {count_dropped} non-finite, nan, negative, greater than 1, or invalid metadata values.\n"
                    f"These will be removed, but it is still recommended to check your linelist for why this happened.",
                    ACIDDroppedDataWarning, stacklevel=3)

        # Apply mask and return results
        linelist_columns = {key: values[mask] for key, values in linelist_columns.items()}
        result = linelist_columns if return_dict else tuple(linelist_columns.values())
        if return_mask:
            return (result, mask) if return_dict else (*result, mask)
        return result
