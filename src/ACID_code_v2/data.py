from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional
import pickle
import numpy as np
from . import utils

class Config:
    """A simple class to store the configuration of the ACID run."""

    defaults = {
        "verbose" : 2,
        "order_range" : [1],
        "telluric_lines" : [
        3820.33, 3933.66, 3968.47, 4327.74, 4307.90, 4383.55, 4861.34,
        5183.62, 5270.39, 5889.95, 5895.92, 6562.81, 7593.70, 8226.96
        ]
    }

    def __init__(self, **kwargs) -> None:
        # Initialize all properties to None, so that we can check if they 
        # have been set or not in the update methods
        self.property_names = self.get_property_names()
        for k in self.property_names:
            setattr(self, f"_{k}", None)

        self.update_hipri(**kwargs) # Set initial values, allowing overwriting and validation of properties

        # for k, v in self.defaults.items():
        #     if getattr(self, k, None) is None:
        #         setattr(self, k, v)

        self.order_range = self.defaults["order_range"] 

        # self.update_lowpri(**self.defaults) # Could do later if moving all defaults to this class

    # --- Update methods ---
    def update_hipri(self, **kwargs: Any) -> None:
        # Update and overwrite existing keys
        for k, v in kwargs.items():
            if v is None:
                continue
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
            # Property setters automatically only set if previous value was None
            if self.is_property(k):
                setattr(self, k, v) # setter already implements "only if None"
            else:
                if getattr(self, k, None) is None:
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
        if self._verbose is None:
            return self.defaults["verbose"]
        return self._verbose
    
    @verbose.setter
    def verbose(self, value) -> None:
        # Make verbosity always an int regardless of input type, and check correct range
        if self._verbose is not None:
            return
        if value is True:
            value = self.defaults["verbose"]
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

    @property
    def telluric_lines(self) -> int:
        if self._telluric_lines is None:
            return self.defaults["telluric_lines"]
        return self._telluric_lines
    
    @telluric_lines.setter
    def telluric_lines(self, lines) -> None:
        telluric_lines = lines
        # Define telluric_lines with defaults if not input, check type if it is
        if getattr(self, "telluric_lines", None) is not None:
            return
        
        if telluric_lines is None:
            telluric_lines = self.defaults["telluric_lines"]
        if not isinstance(telluric_lines, (list, np.ndarray)):
            raise TypeError("telluric_lines must be a list or numpy array of telluric lines to" \
            "mask in angstroms (could be empty or single-valued)")
        telluric_lines = np.array(telluric_lines)
        if telluric_lines.ndim != 1 or telluric_lines.size == 0:
            raise ValueError("telluric_lines must be a one-dimensional array or list")
        
        self._telluric_lines = telluric_lines


@dataclass(slots=True)
class Data:
    """Stores necessary data for the Acid class which can be conveniently updated and saved.
    Allows ACID to handle data that has already been computed to avoid recalculation. This class
    is designed to be lightweight in memory and hence does not store the sampler as an object."""

    # Standard necessary inputs, stored in dictionaries so we can store their state at multiple different
    # states of the calculations in Acid
    wavelengths : Dict[str, np.ndarray] = field(default_factory=dict)
    flux        : Dict[str, np.ndarray] = field(default_factory=dict)
    errors      : Dict[str, np.ndarray] = field(default_factory=dict)
    sn          : Dict[str, np.ndarray] = field(default_factory=dict)

    # Cached products that are expensive or useful for resuming
    alpha                  : Optional[np.ndarray] = None  # the alpha vector used in the linear model, used for solving the linear system in MCMC
    c_factor               : Optional[tuple]      = None  # tuple generated by np.cho_factor, used for solving the linear system in MCMC
    residual_masks         : Optional[np.ndarray] = None  # boolean 1D mask on "combined" grid, used in final process_results step
    velocities             : Optional[np.ndarray] = None  # velocities array, used throughout Acid and Results
    initial_profile        : Optional[np.ndarray] = None  # initial profile generated in residual masking
    initial_profile_errors : Optional[np.ndarray] = None  # corresponding errors
    poly_inputs            : Optional[np.ndarray] = None  # polynomial inputs for just the continuum model
    model_inputs           : Optional[np.ndarray] = None  # the concatenated array of initial profile and poly coefficents, used as input to emcee
    initial_state          : Optional[np.ndarray] = None  # the initial state of the MCMC walkers, used for resuming and debugging

    # Small cached products needed for MCMC if doing reruns
    nwalkers : Optional[int]        = None
    ndim     : Optional[int]        = None

    # Data required/calculated in results/after MCMC sampling
    all_frames : Optional[np.ndarray] = None  # the array to store all frames of the MCMC sampling
    nsteps     : Optional[int]        = 0
    max_steps  : Optional[int]        = None

    # Other useful data:
    initialisation_time : Optional[float] = None  # time taken for initialization
    mcmc_time           : Optional[float] = None  # time taken for MCMC sampling
    get_profiles_time   : Optional[float] = None  # time taken to get profiles
    full_run_time       : Optional[float] = None  # total time for the full run

    # Initialise the properties
    # Config data for convenience, it is very memory light so not an issue to also store in here
    _config   : Config = field(default_factory=Config) # config stored as class, but converted to dict on save
    _linelist : Optional[Dict[str, np.ndarray]] = None

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
        if input_wavelengths is None or input_flux is None or input_errors is None:
            if "input" in self.wavelengths and "input" in self.flux and "input" in self.errors:
                return # if wavelengths already set, do not overwrite
            else:
                raise ValueError("input_wavelengths, input_flux, and input_errors must be provided either as arguments " \
                "to this function or in a Data object.")

        input_wavelengths, input_flux, input_errors = [
            utils.validate_args(arg, i) for i, arg in enumerate((input_wavelengths, input_flux, input_errors))]
        input_sn = utils.validate_args(input_sn, 3, sn=True, allow_none=True)

        # Check all inputs have the same shape
        if not input_wavelengths.shape == input_flux.shape == input_errors.shape:
            raise ValueError("Input wavelengths, spectra and spectral errors must all have the same shape.")

        # Attempt to convert input spectra to be above 0 if they are not already and warning if this is the case
        if np.any(input_flux <= 0):
            if self.config.verbose > 1:
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
                self.all_frames = np.zeros((len(self.flux["input"]), len(self.config.order_range), 2, len(self.velocities)))
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

    def save(self, filename: str = "data.pkl") -> None:
        """Saves the data object to a file using pickling. This will store just the dictionary of the class, 
        not the actual class itself. The load function then will initialise a new Data class using the dictionary.

        Parameters
        ----------
        filename : str
            The name of the file to save the data object to. This should be a .pkl file.
        """
        payload = self.to_dict() # generates a dictionary of the data object for easy pickling

        with open(filename, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename: str) -> Data:
        with open(filename, "rb") as f:
            payload = pickle.load(f)
        
        obj = cls()

        obj.from_dict(payload)
        return obj

    def to_dict(self) -> dict[str, Any]:
        """Converts the data object to a dictionary payload for saving. This is used internally in the save method, but can also be used for debugging or other purposes."""
        payload: dict[str, Any] = {}
        for f in fields(self):
            name = f.name
            val = getattr(self, name)

            if name == "_config":
                payload["config"] = val.to_dict() # store as dict in payload, but store as class in Data
            else:
                payload[name] = val
        return payload
    
    def from_dict(self, payload: dict[str, Any]) -> None:
        """Updates the data object from a dictionary payload. This is used internally in the load method, but can also be used for debugging or other purposes."""
        for f in fields(self):
            name = f.name
            if name == "_config": # config stored as a dict in payload, but stored here as class
                cfg_dict = payload.get("config", {})
                setattr(self, "_config", Config(**cfg_dict))
            else:
                if name in payload:
                    setattr(self, name, payload[name])

    @property
    def config(self) -> Config:
        """Returns the internally stored config object, which contains the configuration of the ACID run."""
        return self._config

    @config.setter
    def config(self, value: Config) -> None:
        self._config = value

    @property
    def linelist(self) -> Dict[str, np.ndarray]:
        """Returns the internally stored linelist. It has keys "wavelengths" and "depths" or index 0 and 1."""
        return Linelist(self._linelist)if self._linelist is not None else None

    def set_linelist(self, linelist_path=None, linelist_wl=None, linelist_depths=None) -> None:
        if self._linelist is not None: # linelist already set, do not overwrite
            return

        if (linelist_wl is None and linelist_depths is None) and linelist_path is None:
            raise ValueError("One of ('linelist_wl' and 'linelist_depths') or 'linelist_path' must be provided.")
        elif linelist_path is None and (linelist_wl is None or linelist_depths is None):
            raise ValueError("If 'linelist_path' is not provided, both 'linelist_wl' and 'linelist_depths' must be provided.")
        elif isinstance(linelist_path, str):
            # VALD linelist code, will add more linelist formats in the future or if requested
            full_linelist = np.genfromtxt('%s'%linelist_path, skip_header=4, delimiter=',', usecols=(1,9))
            linelist_wl = full_linelist[:,0]
            linelist_depths = full_linelist[:,1]
        elif isinstance(linelist_path, Linelist):
            linelist_wl = linelist_path[0]
            linelist_depths = linelist_path[1]
        elif isinstance(linelist_path, dict):
            if "wavelengths" not in linelist_path or "depths" not in linelist_path:
                raise ValueError("If 'linelist_path' is a dict, it must contain keys 'wavelengths' and 'depths'")
            linelist_wl = linelist_path["wavelengths"]
            linelist_depths = linelist_path["depths"]
        elif isinstance(linelist_path, (list, np.ndarray)):
            if len(linelist_path) != 2:
                raise ValueError("If 'linelist_path' is a list or array, it must have length 2, with index 0 being wavelengths and index 1 being depths")
            linelist_wl = linelist_path[0]
            linelist_depths = linelist_path[1]
        else:
            raise ValueError("'linelist_path' must be a string path to a VALD linelist, a dictionary with keys 'wavelengths' and 'depths', " \
            "a Linelist object, or a list/array indexed such that 0 is wavelengths and 1 is depths.")
        
        linelist_wl = np.array(linelist_wl)
        linelist_depths = np.array(linelist_depths)
        Linelist.validate_dimensions(linelist_wl, linelist_depths)
        self._linelist = {"wavelengths": linelist_wl, "depths": linelist_depths}

class Linelist:
    """A simple class to expose the linelist when called in Data"""
    __slots__ = ("ll",) # the only thing stored in this class is the linelist
    def __init__(self, ll: dict):
        self.ll = ll

    def __getitem__(self, k):
        if k == 0:
            return self.ll["wavelengths"]
        if k == 1:
            return self.ll["depths"]
        if isinstance(k, int):
            raise IndexError("Linelist only has keys 0 and 1, or 'wavelengths' and 'depths'")
        return self.ll[k]  # allow "wavelengths"/"depths"
    
    def __iter__(self):
        yield self.ll["wavelengths"]
        yield self.ll["depths"]
    
    @staticmethod
    def validate_dimensions(wavelengths, depths):
        if wavelengths.ndim != 1 or depths.ndim != 1:
            raise ValueError("'wavelengths' and 'depths' must be a one-dimensional array or list")
        if wavelengths.shape != depths.shape:
            raise ValueError("'wavelengths' and 'depths' must have the same length and shape")