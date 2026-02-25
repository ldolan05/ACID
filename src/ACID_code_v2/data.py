from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from unicodedata import name
import numpy as np
from . import utils

class Config:
    """A simple class to store the configuration of the ACID run."""
    defaults = {
        "verbose": 2,
        "order_range": [1],
    }
    def __init__(self, **kwargs) -> None:
        # Initialize all properties to None, so that we can check if they 
        # have been set or not in the update methods
        self.property_names = self.get_property_names()
        for k in self.property_names:
            setattr(self, f"_{k}", None)

        self.update_hipri(**kwargs) # Set initial values, allowing overwriting of properties

        # self.update_lowpri(**self.defaults) # Could do later if moving all defaults to this class

    # --- Update methods ---
    def update_hipri(self, **kwargs: Any) -> None:
        # Update and overwrite existing keys
        for k, v in kwargs.items():
            if self.is_property(k):
                old = getattr(self, f"_{k}", None)
                try:
                    setattr(self, f"_{k}", None)
                    setattr(self, k, v)
                except Exception:
                    setattr(self, f"_{k}", old)
                    raise
            else:
                setattr(self, k, v)

    def update_lowpri(self, **kwargs: Any) -> None:
        # Update but do not overwrite existing keys
        for k, v in kwargs.items():
            # Property setters automaticall only set if previous value was None
            if self.is_property(k):
                setattr(self, k, v) # setter already implements "only if None"
            else:
                if not hasattr(self, k):
                    setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        for k in self.property_names:
            d[k] = getattr(self, k)
        return d

    @classmethod
    def get_property_names(cls) -> set[str]:
        # Collect @property names from the class and its bases
        names: set[str] = set()
        for c in cls.mro():
            for name, attr in c.__dict__.items():
                if isinstance(attr, property):
                    names.add(name)
        return names

    def is_property(self, name: str) -> bool:
        return name in self.property_names

    # --- Properties ---
    @property
    def verbose(self) -> int:
        return self._verbose
    
    @verbose.setter
    def verbose(self, value) -> None:
        # Make verbosity always an int regardless of input type, and check correct range
        if self._verbose is None:
            if value is True:
                value = 2
            elif value is False:
                value = 0
            elif isinstance(value, int):
                if value < 0 or value > 3:
                    raise ValueError("verbose must be an integer between 0 and 3")
            elif isinstance(value, str):
                value = value.lower()
                if value in ["none", "no", "false"]:
                    value = 0
                elif value in ["low", "1"]:
                    value = 1
                elif value in ["medium", "med", "2"]:
                    value = 2
                elif value in ["high", "3"]:
                    value = 3
                else:
                    raise ValueError("verbose string not recognised, must be one of 'none', 'low', 'medium', 'high' or their common variants")
            elif value is None:
                value = self.defaults["verbose"]
            else:
                raise ValueError("verbose must be an integer between 0 and 3, a boolean, or a string indicating the verbosity level")

            self._verbose = value # Only updates if it was previously None

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
    alpha                  : Optional[np.ndarray] = None
    c_factor               : Optional[float]      = None
    residual_masks         : Optional[np.ndarray] = None  # boolean 1D mask on "combined" grid
    velocities             : Optional[np.ndarray] = None
    initial_profile        : Optional[np.ndarray] = None
    initial_profile_errors : Optional[np.ndarray] = None
    poly_inputs            : Optional[np.ndarray] = None
    model_inputs           : Optional[np.ndarray] = None  # the concatenated array of initial profile and poly coefficents, used as input to emcee
    initial_state          : Optional[np.ndarray] = None  # the initial state of the MCMC walkers, used for resuming and debugging

    # Small cached products needed for MCMC if doing reruns
    nwalkers               : Optional[int]        = None
    ndim                   : Optional[int]        = None

    # Data required/calculated in results
    all_frames  : Optional[np.ndarray] = None  # the array to store all frames of the MCMC sampling
    nsteps      : Optional[int]        = None

    # Other useful data:
    initialisation_time : Optional[float] = None
    mcmc_time           : Optional[float] = None
    get_profiles_time   : Optional[float] = None
    full_run_time       : Optional[float] = None

    # Config data for convenience
    config: Dict[str, Any] = field(default_factory=dict)

    def set_inputs(
        self,
        input_wavelengths: Optional[np.ndarray] = None,
        input_flux:        Optional[np.ndarray] = None,
        input_errors:      Optional[np.ndarray] = None,
        input_sn:          Optional[np.ndarray] = None,
        skips:             int                  = 1,
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
            Signal-to-noise array for the input spectra, by default None
        skips : int, optional
            Number of pixels to skip when processing the spectra, by default 1 (no skipping)
        """

        # Validate input arrays using the validate_args function within utils.py, ensuring inputs are correct shape, or to
        # best guess the user's intentions. See the utils.validate_args function for more details. This also converts
        # inputs to numpy arrays.
        input_wavelengths, input_flux, input_errors = [
            utils.validate_args(arg, i) for i, arg in enumerate((input_wavelengths, input_flux, input_errors))]
        input_sn = utils.validate_args(input_sn, 3, sn=True, allow_none=True)

        # Check all inputs have the same shape
        if not input_wavelengths.shape == input_flux.shape == input_errors.shape:
            raise ValueError("Input wavelengths, spectra and spectral errors must all have the same shape.")

        # Attempt to convert input spectra to be above 0 if they are not already and warning if this is the case
        if np.any(input_flux <= 0):
            if self.config["verbose"] > 1:
                print("Input spectra contain flux values <= 0. ACID will attempt to rescale inputs, and mask " \
                f"negative values.\nHowever, it is recommended to input spectra that are already normalised and positive. " \
                f"Please check your data.\nYou can check acid.scale_spectra for more information on how this is done.")
            input_flux, input_errors = utils.scale_spectra(input_flux, input_errors)
        # Validated frame_sns input
        # If frame_sns is not provided, estimate using specutils, this is a very rudimentary guess and get around for not
        # providing a SNS which should normally come from fits files.
        if input_sn is None:
            input_sn = utils.guess_SNR(input_wavelengths, input_flux, input_errors)
            assert input_sn.ndim == input_flux.ndim - 1, \
            "input_sn.ndim and input_flux.ndim-1 do not match"
        if np.asarray(input_sn).shape[0] != np.asarray(input_flux).shape[0]:
            raise ValueError("input_sn must be a single-valued list/array with the average S/N for each frame, " \
            "not an array of S/N values for each pixel. " \
            "The shape of the input input_sn does not match the number of frames in input_flux.")

        # Apply skips and set inputs to class variables
        self.wavelengths["input"] = input_wavelengths[:, ::skips]
        self.flux["input"]        = input_flux[:, ::skips]
        self.errors["input"]      = input_errors[:, ::skips]
        self.sn["input"]          = input_sn
        
    def initiate_all_frames(self, all_frames: np.ndarray) -> None:
        """Initiates the all_frames variable, used in the ACID method, to eventually store the results of the MCMC sampling.
        This is used to update the all_frames variable after each sampling step, allowing for resuming and avoiding
        recalculation of profiles if the user wishes to continue sampling.

        Parameters
        ----------
        all_frames : np.ndarray
            The array of all frames to be stored in the data class. This should be of shape (n_steps, n_profiles, 2)
            where the last dimension contains the profile and its error.
        """
        if isinstance(all_frames, str):
            if all_frames == "default":
                all_frames = None # legacy behaviour
        if all_frames is None:
            if self.all_frames is None:
                # By default order_range is [1], so len(self.order_range) = 1, which is same as original
                # code behaviour. This change allows self.order_range to be used in ACID_HARPS.
                self.all_frames = np.zeros((len(self.flux["input"]), len(self.config["order_range"]), 2, len(self.velocities)))
        else:
            self.all_frames = all_frames
        if isinstance(self.all_frames, object):
            from .result import Result
            if isinstance(self.all_frames, Result):
                self.all_frames = self.all_frames.all_frames
        if not isinstance(self.all_frames, np.ndarray):
            raise TypeError("'all_frames' must be a numpy array")
        if not self.all_frames.ndim == 4:
            raise ValueError("'all_frames' must be a 4-dimensional numpy array, see docstring for details")
