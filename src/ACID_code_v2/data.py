from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from . import utils
from .result import Result

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
    poly_inputs           : Optional[np.ndarray] = None
    model_inputs          : Optional[np.ndarray] = None # the concatenated array of initial profile and poly coefficents, used as input to emcee
    initial_state         : Optional[np.ndarray] = None # the initial state of the MCMC walkers, used for resuming and debugging

    # Required settings taken from initialisation for methods to run
    verbose: int = 2

    # Other unchanged data specifically required for MCMC to run
    velocities  : Optional[np.ndarray] = None
    seed        : Optional[int]        = None
    fit_profile : bool                 = True

    # Data required for results
    nsteps      : Optional[int]        = None
    all_frames  : Optional[np.ndarray] = None  # the array to store all frames of the MCMC sampling
    order_range : Optional[np.ndarray] = None

    # Other useful data:
    initialisation_time : Optional[float] = None
    mcmc_time           : Optional[float] = None
    get_profiles_time   : Optional[float] = None
    full_run_time       : Optional[float] = None

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

        # Attempt to convert input spectra to be within 0 and 1 if they are not already and warning if this is the case
        if np.any(input_flux <= 0):
            if self.verbose > 1:
                print("Input spectra contain values <= 0. ACID will attempt to rescale inputs, and mask " \
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

        # Apply skips
        self.apply_skips(skips)
        
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
                self.all_frames = np.zeros((len(self.flux["input"]), len(self.order_range), 2, len(self.velocities)))
        else:
            self.all_frames = all_frames
        if isinstance(self.all_frames, Result):
            self.all_frames = self.all_frames.all_frames
        if not isinstance(self.all_frames, np.ndarray):
            raise TypeError("'all_frames' must be a numpy array")
        if not self.all_frames.ndim == 4:
            raise ValueError("'all_frames' must be a 4-dimensional numpy array, see docstring for details")
    
    def apply_skips(self, skips: int) -> None:
        if skips <= 1:
            return
        self.wavelengths["input"] = self.wavelengths["input"][:, ::skips]
        self.flux["input"]        = self.flux["input"][:, ::skips]
        self.errors["input"]      = self.errors["input"][:, ::skips]