from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

@dataclass(slots=True)
class Data:
    """Stores necessary data for the Acid class which can be conveniently updated and saved.
    Allows ACID to handle data that has already been computed to avoid recalculation. This class
    is designed to be lightweight in memore and hence does not store the sampler as an object."""

    wavelengths: Dict[str, np.ndarray] = field(default_factory=dict)
    flux:        Dict[str, np.ndarray] = field(default_factory=dict)
    errors:      Dict[str, np.ndarray] = field(default_factory=dict)
    sn:          Dict[str, np.ndarray] = field(default_factory=dict)

    # Cached products that are expensive or useful for resuming
    alpha                 : Optional[np.ndarray] = None
    c_factor              : Optional[float]      = None
    residual_masks        : Optional[np.ndarray] = None  # boolean 1D mask on "combined" grid
    initial_profile       : Optional[np.ndarray] = None
    initial_profile_errors: Optional[np.ndarray] = None

    # # Continuum fit products used in get_profiles step
    # poly_inputs    : Optional[np.ndarray] = None
    # poly_cos       : Optional[np.ndarray] = None
    # continuum_error: Optional[np.ndarray] = None

    def set_inputs(
        self,
        input_wavelengths: Optional[np.ndarray] = None,
        input_flux:        Optional[np.ndarray] = None,
        input_errors:      Optional[np.ndarray] = None,
        input_sn:          Optional[np.ndarray] = None,
    ) -> None:
        """Sets the input data for the ACID class. This is used to initialize the data object with the raw spectra,
        and to validate the arguments (previously done within the ACID function).
        Parameters
        ----------
        input_wavelengths : np.ndarray, optional
            Wavelength array for the input spectra, by default None
        input_flux : np.ndarray, optional
            Flux array for the input spectra, by default None
        input_errors : np.ndarray, optional
            Error array for the input spectra, by default None
        input_sn : np.ndarray, optional
        """
