from __future__ import annotations
from dataclasses import dataclass, field, fields
from beartype import beartype
from tqdm import tqdm
import copy
from scipy.interpolate import interp1d
import traceback as tb
from typing import Any, Dict, Optional
from emcee import EnsembleSampler
from emcee.backends.backend import Backend
from emcee.backends.hdf import HDFBackend
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle, os, traceback
import pandas as pd
import numpy as np
from . import utils
from .errors import *
from .utils import IntLike, Array1D, Array2D, Array3D, Scalar, c_kms

class MaskingLines:
    """
    A simple class to expose the telluric lines when called in Config. This will help
    to store telluric lines as a dictionary. With a default itercall to list the line-wise elements,
    but a dictionary index to also store the width of the line, which can then allow for masking Hydrogen
    lines with much wider masks.
    """
    __slots__ = ("lines",) # the only thing stored in this class is this dictionary

    def __init__(self, lines:dict) -> None:
        """
        Sets the lines attribute after validating the input lines dictionary. The format is specified in :py:class:`Acid`.
        """
        self.lines = self.validate_lines(lines)

    def __getitem__(self, key):
        # should work for int or str keys
        return self.lines[key]

    def __iter__(self):
        return iter(self.lines.items())

    def get_masks(self, x, with_names=False) -> list | dict:
        """
        Generates masks for the given input array `x` based on the stored lines and widths.

        Parameters
        ----------
        x : array-like
            The input array for which to generate masks.
        with_names : bool, optional
            Whether to return a dictionary with line names as keys. Useful if plotting. Default is False.

        Returns
        -------
        list | dict
            A list of masks (ie list of 1D mask arrays) or a dictionary of masks keyed by line names.
        """
        mask = [] if not with_names else {}
        for name, line_data in self.lines.items():
            lines = np.asarray(line_data["lines"])
            widths = np.asarray(line_data["widths"])

            limits = 3 + (widths / c_kms) * lines
            conditions = np.abs(x[None, :] - lines[:, None]) <= limits[:, None]
            line_mask = np.any(conditions, axis=0)
            if with_names:
                mask[name] = line_mask
            else:
                mask.append(line_mask)
        return mask
    
    def get_1d_mask_on_grid(self, x:np.ndarray) -> np.ndarray:
        """
        Generates a single 1D mask for the given input array `x` based on the stored lines and widths.

        Parameters
        ----------
        x : array-like
            The input array for which to generate the mask.

        Returns
        -------
        np.ndarray
            A 1D boolean mask array where True indicates that the corresponding element in `x` is within the masking region of any line.
        """
        masks = self.get_masks(x)
        if len(masks) == 0:
            return np.zeros_like(x, dtype=bool)
        combined_mask = np.any(masks, axis=0)
        return combined_mask

    @staticmethod
    def validate_lines(input_lines:dict|MaskingLines) -> dict:
        """
        Standard method to validate linelist input, the format is quite flexible for convenience, but the output is always a standardised dictionary.
        See :ref:`masking_lines`
        """

        # Skip validation if MaskingLines object is input, as it would have already been validated
        if isinstance(input_lines, MaskingLines):
            return input_lines.lines

        # Set error messages for common errors to avoid repetition
        length_mismatch_error = f"The number of lines and inputted widths must be the same if inputting widths.\n" \
        f"If you only wish to input the widths of certain lines, use a list of tuples, see :ref:`masking_lines` for more details."
        default_width_error = "No default width was provided for the masking_lines of {}, see :ref:`masking_lines` for more details."

        # Set variables to be updated within the loop
        final_dict = {}

        for name, line_object in input_lines.items():
        
            default_width = None

            # Allow first dict inputs, convert them first to a array format to be validated like any other array input
            if isinstance(line_object, dict):
                if "default_width" in line_object:
                    default_width = line_object["default_width"]
                if "lines" not in line_object:
                    raise ValueError(f"If the value for {name} is a dictionary, it must contain a 'lines' key with the list/array of lines to mask")
                if "widths" in line_object:
                    line_input = [(l, w) for l, w in zip(line_object["lines"], line_object["widths"])]
                else:
                    line_input = line_object["lines"]
            else:
                line_input = line_object

            if isinstance(line_input, (np.ndarray, list)):
                # Reject empty lists or arrays, as this is likely a user error
                if len(line_input) == 0:
                    raise ValueError(f"The masking_lines for {name} cannot be an empty list or array, use None/remove the input to use the default lines.")

                # For lists of tuples, allow len 1 or 2 depending on if default_width was provided in the dictionary
                if isinstance(line_input[0], tuple):
                    lines = []
                    widths = []
                    for line in line_input:
                        if len(line) == 1:
                            lines.append(line[0])
                            if default_width is None:
                                raise ValueError(default_width_error.format(name))
                            widths.append(default_width)
                        elif len(line) == 2:
                            lines.append(line[0])
                            widths.append(line[1])
                        else:
                            raise ValueError(f"If the masking_lines for {name} is a list or array of tuples, each tuple must have length 1 " \
                            f"(line only) or 2 (line and width). \nGot tuple with length {len(line)}")          

                else:
                    # For arrays or lists, convert to numpy array and check dimensions
                    try:
                        lines = np.array(line_input)
                    except Exception as e:
                        raise ValueError(f"Could not convert the masking_lines for {name} to a numpy array. \n"
                                         f"It's possible the dimensions do not have the same shape. Please check the input format. \nError: {e}")
                    if lines.size == 0:
                        raise ValueError("lines cannot be an empty array or list, use None/remove the input to use the default lines.")                
                    if lines.ndim == 1:
                        if default_width is None:
                            raise ValueError(default_width_error.format(name))
                        widths = [default_width for _ in lines]
                    elif lines.ndim == 2:
                        widths = lines[1]
                        lines = lines[0]
                        if len(lines) != len(widths):
                            raise ValueError(length_mismatch_error + f"\nGot {len(lines)} lines and {len(widths)} widths.")
                    else:
                        raise ValueError("lines must be a one- or two-dimensional array or list")

            else:
                raise ValueError(f"The masking line for {name} does not conform to the accepted formats, see :ref:`masking_lines`"
                                 f" for more details. Got type {type(line_input)}.")

            if len(lines) != len(widths):
                raise ValueError(f"lines and widths should be of same length, got: {len(lines)}, {len(widths)}")
            final_dict[name] = {"lines": np.array(lines), "widths": np.array(widths)}
        return final_dict

@beartype
class Config:
    """The main class for storing ACID configuration settings, with methods to plot and save/load the configuration state."""

    #: The default configuration settings for ACID, used if not set by the user. See :py:class:`Acid` for more details on how these are used in ACID.
    defaults = {
        # INIT CONFIGURATION
        "verbose" : 2,
        "sampler_progress" : None,
        "order" : 0,
        "order_range" : [0],
        "masking_lines" : {
            "narrow" : {
                "default_width" : 200,
                "lines" : [
                    3820.33, # metal?
                    4307.90, # metal?
                    4327.74, # metal?
                    4383.55, # Fe 1
                    5270.39, # Fe 1
                    5889.95, # Na I D2
                    5895.92, # Na I D1
                    7593.70, # O2 telluric
                    8226.96, # H2O telluric?
                ]
            },
            "medium" : {
                "default_width" : 500,
                "lines" : [
                    3933.66, # Ca II K
                    3968.47, # Ca II H
                    5167.32, # Mg I b (1) triplet
                    5172.68, # Mg I b (2) triplet
                    5183.62, # Mg I b (3) triplet
                ]
            },
            "wide" : {
                "default_width" : 2000,
                "lines" : [
                    3835.38, # H eta
                    3889.05, # H zeta
                    4101.74, # H delta
                    4340.47, # H gamma
                    4861.34, # H beta
                    6562.81, # H alpha
                ]
            },
        },
        "seed" : None,
        "dir" : None,
        "save_path" : None,
        "sampler_path" : None,
        "figure_dir" : None,

        # RUN_ACID CONFIGURATION
        "deterministic_profile" : True,
        "poly_ord" : 3,
        "continuum_percentile" : 99,
        "n_bins" : 10,
        "bin_size" : None,
        "pix_chunk" : 50, # TODO: document+test this increase from 20
        "dev_perc" : 25,
        "sigma_lower" : 3,
        "sigma_upper" : 5,
        "skips" : 1,
        "od"    : True,
        "sparse" : True,
        "depth_group_rules" : None,
        "profile_groups" : None,
        "sampler_type" : "emcee",
        "parallel" : True,
        "cores" : None,
        "nwalkers" : None,
        "nsteps" : 10000,
        "max_steps" : None,
        "check_interval" : 1000,
        "min_checks" : 1,
        "min_tau_factor" : 50,
        "tau_tol" : 0.1,
        "moves" : [
            ("StretchMove", 0.20, {}),
            ("DESnookerMove", 0.1, {}),
            ("DEMove", 0.6, {}),
            ("DEMove", 0.1, {"gamma0": 1.0}),
        ],
        "run_mcmc" : True,
        "continuum_method" : None, # forced here or calculated in ACID based on poly order
    }

    #: Property list for error handling
    properties = ["verbose", "masking_lines"]
    _properties = ["_verbose", "_masking_lines"]

    #: For error handling if Data attributes were accidentally set in config. These should be set in :py:class:`Data` instead
    data_attributes = ["linelist", "velocities"]
    data_attributes_input_str = "'{}' is a Data property and should not be set in the Config class.\nSet it directly with 'Data.{}={}' instead."

    def __init__(self, **kwargs) -> None:
        """Initialise with the defaults, overwrite with any inputted kwargs"""
        self.update_hipri(**kwargs) # Set initial values, allowing overwriting and validation of properties

    def __getattr__(self, name: str) -> Any:
        """
        If an attribute is not found, check if it is in the defaults or properties and 
        return the default value if it is. Otherwise, raise an AttributeError.
        """
        if name in self.defaults:
            return self.defaults[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def __getattribute__(self, name):
        """Override the default attribute access to allow for environment variable overrides."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        raw = os.environ.get("_ACID_CONFIG")

        if raw is not None:
            import json

            env_config = json.loads(raw)

            if not isinstance(env_config, dict):
                raise ValueError("_ACID_CONFIG must decode to a dictionary")

            if name in env_config:
                return env_config[name]

        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.data_attributes:
            raise AttributeError(self.data_attributes_input_str.format(name, name, value))

        if value is None:
            # If value is None, do not set the attribute
            return

        if name in self._properties or name in self.defaults:
            super().__setattr__(name, value)
            return

        raise AttributeError(
            f"'Config' object has no attribute '{name}', "
            f"valid attributes are: {list(self.defaults.keys())}"
        )

    def __repr__(self) -> str:
        """String representation of the Config object, showing all settings in a user-friendly format."""
        full_dict = self.to_dict()
        return f"Config instance with the following settings:\n" + "\n".join([f"{k}: {v}" for k, v in full_dict.items()])

    # --- Update methods ---
    def update_hipri(self, **kwargs: Any) -> None:
        """Updates and overwrites existing keys if their values are not None.
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to the configuration settings to be updated. 
            The keys must be valid configuration options as defined in the `defaults` class variable.
        
        Raises
        ------
        KeyError
            If any key in `kwargs` is not a valid configuration option as defined in the `defaults` class variable.
        """
        for k, v in kwargs.items():
            # First raise error if Data attribute was input
            if k in self.data_attributes:
                raise AttributeError(self.data_attributes_input_str.format(k, k, v))
            # Then raise error if trying to set an attribute that is not in defaults
            if k not in self.defaults and k not in self._properties:
                raise KeyError(f"Key '{k}' is not a valid configuration option.")
            if v is None:
                # If input is None, continue, None always makes no change to current value/default
                continue
            else:
                setattr(self, k, v)

    def update_lowpri(self, **kwargs: Any) -> None:
        """Updates but does not overwrite existing stored keys.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to the configuration settings to be updated. 
            The keys must be valid configuration options as defined in the `defaults` class variable.

        Raises
        ------
        KeyError
            If any key in `kwargs` is not a valid configuration option as defined in the `defaults` class variable.
        """
        for k, v in kwargs.items():
            # First raise error if Data attribute was input
            if k in self.data_attributes:
                raise AttributeError(self.data_attributes_input_str.format(k, k, v))
            # Then raise error if trying to set an attribute that is not in defaults
            if k not in self.defaults:
                raise KeyError(f"Key '{k}' is not a valid configuration option.")

            if v is None:
                continue

            stored_name = "_" + k if k in self.properties else k # Add the _ for properties
            if stored_name not in self.__dict__ or self.__dict__[stored_name] is None:
                setattr(self, k, v)

    def to_dict(self) -> dict:
        """Convert the Config object to a dictionary of only the stored/modified attributes,
        if you want the full dictionary including defaults, use `to_full_dict`."""
        return {k: v for k, v in self.__dict__.items()}

    def to_full_dict(self) -> dict:
        """Convert the Config object to a dictionary including all defaults and stored/modified attributes."""
        out = {}

        for k in self.defaults:
            value = getattr(self, k)

            if k == "masking_lines" and hasattr(value, "to_dict"):
                value = value.to_dict()

            out[k] = value

        return out

    # --- Properties ---
    @property
    def verbose(self) -> IntLike:
        """The stored global verbosity setting for ACID. See :py:class:`Acid` for more details on how this is used in ACID."""
        if self.__dict__.get("_verbose", None) is None:
            return self.defaults["verbose"]
        return self._verbose

    @verbose.setter
    def verbose(self, value:IntLike|str|bool|None) -> None:
        """Set the global verbosity setting for ACID. Accepts an integer, boolean, or string indicating the verbosity level."""
        # Make verbosity always an int regardless of input type, and check correct range
        if value is None:
            return
        elif value is True:
            value = self.defaults["verbose"] # normally 2
        elif value is False:
            value = 0
        elif isinstance(value, (int, np.integer)):
            if value < 0 or value > 4:
                raise ValueError("verbose must be an integer between 0 and 4")
        elif isinstance(value, str):
            value = value.lower()
            if value in ["none", "no", "false", "off", "n", "0"]:
                value = 0
            elif value in ["low", "lo", "l", "1"]:
                value = 1
            elif value in ["medium", "med", "m", "2"]:
                value = 2
            elif value in ["high", "all", "hi", "h", "3"]:
                value = 3
            elif value in ["debug", "dbg", "d", "4"]:
                value = 4
            else:
                raise ValueError("verbose string not recognised, must be one of 'none', 'low', 'medium', 'high', 'debug' or their common variants")
        else:
            raise ValueError("verbose must be an integer between 0 and 4, a boolean, or a string indicating the verbosity level")

        self._verbose = value

    @property
    def masking_lines(self) -> MaskingLines:
        """The stored masking lines for ACID. See :ref:`masking_lines` for more details on how this is used in ACID."""
        if self.__dict__.get("_masking_lines", None) is None:
            return MaskingLines(self.defaults["masking_lines"])
        return MaskingLines(self._masking_lines)

    @masking_lines.setter
    def masking_lines(self, masking_lines:dict|MaskingLines|None) -> None:
        """Set the masking lines for ACID. Accepts a dictionary, a MaskingLines object, or None."""
        if masking_lines is not None:
            self._masking_lines = MaskingLines.validate_lines(masking_lines)

    def plot_masking_lines(self, return_fig:bool=False) -> None|tuple:
        """
        Plots the telluric and/or hydrogen lines that will be masked in the residual masking step, with shaded regions indicating 
        the widths of the masks.

        Parameters
        ----------
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False.

        Returns
        -------
        If return_fig is True, returns a tuple of (fig, ax) where fig is the matplotlib figure object and ax is the axis object.
        Otherwise, returns None and shows the plot.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (name, line_data) in enumerate(self.masking_lines):
            for line, width in zip(line_data["lines"], line_data["widths"]):
                delta_lambda = line * width / c_kms
                ax.axvline(line, linestyle='--', color=f'C{i+1}',
                           label=f'{name.capitalize()} line' if line == line_data["lines"][0] else None)
                ax.axvspan(line - delta_lambda, line + delta_lambda, alpha=0.1, color=f'C{i+1}')

        ax.set_title("Lines to be masked")
        ax.set_xlabel("Wavelength (Angstroms)")
        ax.set_ylabel("Masking region")
        ax.legend()
        if return_fig:
             return fig, ax
        utils.show_or_save(plt, self.figure_dir, "masking_lines.png", self.verbose)

    @classmethod
    def print_defaults(cls) -> None:
        """Print the default configuration settings for ACID."""
        print("Default configuration:")
        for k, v in cls.defaults.items():
            print(f"{k}: {v}")

@beartype
@dataclass(slots=True)
class Data:
    """
    Stores necessary data for the Acid class which can be conveniently updated and saved.
    Allows ACID to handle data that has already been computed to avoid recalculation. This class
    is designed to be lightweight in memory and hence does not store the sampler as an object. This is handled in the Result class.
    Note that a Data class should only hold the data for ONE order or observation, but it can hold
    the data for multiple frames of the same order.
    """

    # The standard necessary inputs, stored in dictionaries so we can store their state at multiple different
    # states of the calculations in Acid
    # -------------------------------------------------------------------------------------------------------
    #: The wavelengths for each frame, stored as a dictionary with frame names as keys and 1D numpy arrays as values.
    wavelengths : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The fluxes for each frame, stored as a dictionary with frame names as keys and 1D numpy arrays as values.
    flux        : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The errors for each frame, stored as a dictionary with frame names as keys and 1D numpy arrays as values.
    errors      : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The signal-to-noise ratio for each frame, stored as a dictionary with frame names as keys and 1D numpy arrays as values.
    sn          : Dict[str, np.ndarray] = field(default_factory=dict)

    # Generated arrays from the above inputs, stored in dictionaries so we can store their state at multiple different stages
    # -------------------------------------------------------------------------------------------------------
    #: The normalised wavelengths to be used with the poly_coeffs to generate the continuum
    norm_wavelengths : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The fitted flux from either the initial continuum fits, or the final mcmc fitted continua
    fitted_flux      : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The fitted errors from either the initial continuum fits, or the final mcmc fitted continua
    fitted_errors    : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The fitted continuum from either the initial continuum fits, or the final mcmc fitted continua
    continuum        : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The error on the fitted continuum, only used after mcmc fitting, as this is the only important error for final profiles
    continuum_error  : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The corresponding polynomial coefficients for just the above continuum
    poly_coeffs      : Dict[str, np.ndarray] = field(default_factory=dict)

    # Cached LSD products that can be expensive and useful for resuming or loading runs
    # ---------------------------------------------------------
    #: The alpha vector used in the linear model, used for solving the linear system in MCMC
    alpha                  : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The forward model x-axis for each frame, used for solving the linear system in MCMC
    forward_x              : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The forward model y-axis for each frame, used for solving the linear system in MCMC
    forward_y              : Dict[str, np.ndarray] = field(default_factory=dict)
    #: Tuple generated by np.cho_factor, used for solving the linear system in MCMC - paired with alpha_fitting
    c_factor               : Dict[str, tuple]      = field(default_factory=dict)
    #: The profile for each frame, used for solving the linear system in MCMC idx 0 is the profile, 1 is the errors, and 2 is the cov. matrix
    profile                : Dict[str, list]       = field(default_factory=dict)
    #: The residuals between the forward model and flux used to generate said model. Scaled to the forward model flux.
    residuals              : Dict[str, np.ndarray] = field(default_factory=dict)
    #: The indexes from the full linelist that was used in this LSD run
    ll_mask                : Dict[str, np.ndarray] = field(default_factory=dict)

    # Products generated from residual masking
    # ---------------------------------------------------------
    #: Boolean 1D mask on "initial" grid, used to mask out NaN/invalid values to go from initial to the combined spectra
    nanmask                : Optional[np.ndarray] = None
    #: Boolean 1D mask on "combined" grid, obtained from the MaskingLines class
    line_mask              : Optional[np.ndarray] = None
    #: Boolean 1D mask on "combined" grid, used for masking out pixels with high residuals in the residual masking step
    sigma_mask             : Optional[np.ndarray] = None
    #: Boolean 1D mask on "combined" grid, used for masking out pixels in the initial pixel masking step based on high SN or low SN
    pix_mask               : Optional[np.ndarray] = None
    #: Boolean 1D mask on "combined" grid, that is just a combination of the three above masks
    full_mask              : Optional[np.ndarray] = None

    # Small cached products needed for MCMC if doing reruns
    # -----------------------------------------------------
    #: The initial state of the MCMC walkers, used for resuming and debugging
    initial_state : Optional[np.ndarray] = None
    #: The number of walkers and dimensions for the MCMC sampler, used for reshaping the samples if resuming
    nwalkers      : Optional[int]        = None
    #: The number of dimensions for the MCMC sampler, used for reshaping the samples if resuming
    ndim          : Optional[int]        = None

    # Data required/calculated in results/after MCMC sampling
    # -------------------------------------------------------
    #: Same as the profile and profile errors above, but for all inputted frames and for only the final result. 0 returns the same as profile["final"] for first frame
    profiles : Optional[list] = None
    #: The number of steps taken in the MCMC sampling, used for checking convergence and for resuming
    nsteps   : Optional[int]  = 0
    #: A flag for whether the profiles have been fully calculated to avoid recalculating
    complete : bool           = False # is set to True when the profiles and final profile has been fully calculated

    # Other useful data and figures
    # -----------------------------
    #: The grouping of the profiles based on their depth, used for plotting and analysis
    profile_groups       : Optional[np.ndarray] = None
    #: Defining if the user inputted groups, helps distinguish in the LSD class between generated vs input groups
    input_profile_groups : Optional[np.ndarray] = None
    #: Internal variables used for plotting the continuum_fit and the residual masks
    plotting_variables   : Dict[str, Any]  = field(default_factory=dict)
    #: setup_time (float) - The time taken for initialization
    setup_time           : Optional[float] = 0  # time taken for initialization and setup
    #: mcmc_time (float) - The time taken for MCMC sampling
    mcmc_time            : Optional[float] = 0  # time taken for MCMC sampling
    #: results_time (float) - The time taken to get the final profiles
    results_time         : Optional[float] = 0 
    #: total_time (float) - The total time for the full run
    total_time           : Optional[float] = 0
    #: The exception class if an error was raised during the run
    exception            : Optional[Exception] = None
    #: The traceback string if an error was raised during the run
    traceback            : Optional[str] = None
    #: A tracker for if warnings have been printed in any LSD call
    lsd_warnings_flag    : bool = False
    #: If in debug mode (verbose = 4), debug data is stored as a dictionary here:
    debug                : Dict = field(default_factory=dict)

    # Initialise the properties
    # -------------------------
    #: The sampler object stored as a class, but when saved, only a string to the HDF5 path is stored.
    _sampler    : Optional[EnsembleSampler] = None # stored ensemble sampler
    #: Config data for convenience as a class, but converted to a dictionary on save to avoid pickling issues
    _config     : Config = field(default_factory=Config)
    #: The linelist is stored as a dictionary but exposed as a :py:class:`LineList` object when the property is accessed.
    _linelist   : Optional[Dict[str, np.ndarray]] = None
    #: The velocities are stored as a 1D numpy array
    _velocities : Optional[np.ndarray] = None

    def __repr__(self) -> str:
        """String representation of the Data object, showing all stored attributes in a user-friendly format."""
        ll_wl = self.linelist["wavelengths"]
        full_ll_range = [np.min(ll_wl), np.max(ll_wl)]
        cut_ll_range = [np.min(self.ll_mask["masked"]), np.max(self.ll_mask["masked"])]
        mp_lsd = self.profile_groups is not None

        output = [
            f"Number of velocity points: {len(self.velocities)}",
            f"deltav: {self.velocities[1] - self.velocities[0]} km/s",
            f"Number of linelist points (clipped/full): {len(self.ll_mask['masked'])} / {len(ll_wl)}",
            f"Linelist range (from-to, clipped/full): {cut_ll_range[0]}-{cut_ll_range[1]} / {full_ll_range[0]}-{full_ll_range[1]}",
            f"Order: {self.config.order}",
            f"Order range (min-max): {np.min(self.config.order_range)} - {np.max(self.config.order_range)}",
            f"Verbosity: {self.config.verbose}",
            f"Save path: {self.config.save_path}",
            f"Sampler path: {self.config.sampler_path}",
            f"Figure saving path: {self.config.figure_dir}",
            f"Using deterministic profile?: {self.config.deterministic_profile}",
            f"Polynomial order: {self.config.poly_ord}",
            f"Continuum percentile: {self.config.continuum_percentile}",
            f"Number of bins: {self.config.n_bins}",
            f"Bin size (number of points per bin, can be None): {self.config.bin_size}",
            f"Pixel chunk size: {self.config.pix_chunk}",
            f"Deviation percentage: {self.config.dev_perc}",
            f"Sigma lower: {self.config.sigma_lower}",
            f"Sigma upper: {self.config.sigma_upper}",
            f"Skips: {self.config.skips}",
            f"Using optical depth?: {self.config.od}",
            f"Using multi-profile LSD?: {mp_lsd}",
        ]

        if mp_lsd:
            output += [
                f"Depth group rules (can be None): {self.config.depth_group_rules}",
                f"Number of groups: {len(np.unique(self.profile_groups))}",
            ]

        output += [
            f"Sampler type: {self.config.sampler_type}",
            f"Parallel processing?: {self.config.parallel}",
            f"Number of cores: {self.config.cores}",
            f"Number of walkers: {self.nwalkers}",
            f"Number of dimensions: {self.ndim}",
            f"Number of steps (can be None): {self.config.nsteps}",
            f"Maximum number of steps (can be None): {self.config.max_steps}",
            f"Continuum method: {self.config.continuum_method}",
            f"Exception raised: {self.exception}",
        ]

        return "\n".join(output)

    @property
    def sampler(self) -> EnsembleSampler|None:
        """
        The ensemble sampler object used for MCMC sampling.
        This is stored as a class variable but when saved, only the path to the sampler is stored to avoid pickling issues.
        """
        return self._sampler

    @sampler.setter
    def sampler(self, sampler:EnsembleSampler|Backend|HDFBackend|str|None) -> None:
        """
        Sets the sampler object from various types.
        This is stored as a class variable but when saved, only the path to the sampler is stored to avoid pickling issues.
        """
        if sampler is not None:
            from .mcmc import MCMC
            log_prob_fn = MCMC(self)

        if isinstance(sampler, EnsembleSampler):
            self._sampler = sampler
        elif isinstance(sampler, Backend) or isinstance(sampler, HDFBackend):
            self._sampler = utils.backend_to_sampler(sampler, log_prob_fn)
        elif isinstance(sampler, str):
            if os.path.exists(sampler):
                self._sampler = utils.backend_to_sampler(HDFBackend(sampler), log_prob_fn)
            else:
                if self.config.verbose >= 1:
                    print(f"Warning: The sampler was not found at the provided path '{sampler}', it may have been moved or deleted. \n"
                          f"The sampler will be set to None.", flush=True)
                self._sampler = None
                # TODO: Allow sampler to have completed results, but no sampler, and configured methods with _requiresampler property that need them
        elif sampler is None:
            if self.config.verbose >= 1 and self._sampler is not None:
                print("Warning, you have discarded the sampler.")
            self._sampler = None
        
        if self._sampler is not None and isinstance(self._sampler.backend, HDFBackend):
            self.config.sampler_path = os.path.abspath(self._sampler.backend.filename)

    @property
    def velocities(self) -> Array1D|None:
        """The velocity grid to perform LSD on."""
        return self._velocities

    @velocities.setter
    def velocities(self, value:Array1D|None) -> None:
        """Sets and overwrites the velocity grid if the value is not None.
        If overwriting, resets the Data instance as calculations are dependent on velocities."""
        if value is not None:
            
            # First validate the input array
            velocities = np.array(value)
            if not np.all(np.isfinite(velocities)):
                raise ValueError("The velocity grid you are trying to set must all be finite and not contain NaNs")
            
            # Set overwriting flag if velocities already exists and are different from new input
            overwriting = False
            if self._velocities is not None:
                if len(self._velocities) != len(velocities):
                    overwriting = True
                elif not np.allclose(self._velocities, velocities):
                    overwriting = True

            # Set velocities
            self._velocities = np.array(value)

            # Reset and warn if overwriting
            if overwriting:
                print("Warning: Overwriting existing velocities in Data. The Data instance will be reset to clear calculations that depend on the velocities.\n" \
                "The linelist, config, and original data inputs will not be reset.")
                self.reset()

    @property
    def linelist(self) -> LineList|None:
        """Returns the internally stored linelist. It has keys "wavelengths" and "depths" or index 0 and 1."""
        return LineList(self._linelist) if self._linelist is not None else None

    @linelist.setter
    def linelist(self, linelist:Array2D|str|LineList|dict[str,Array1D]|None) -> None:
        """
        Sets the linelist for the data object. The linelist formats follows that of the doccumentation in the :py:class:`Acid` class,
        which then internally uses this function to set the linelist in the data object. The linelist is stored as a dictionary with 
        keys "wavelengths" and "depths", but is exposed as a :py:class:`LineList` object when accessed through the property. The LineList
        class allows for easy access to plotting, indexing, and validation.

        Parameters
        ----------
        linelist : Array2D, str, LineList, dict[str, Array1D], or None
             The linelist to be set, which can be in various formats for convenience.
             See :py:class:`Acid` init for the accepted linelist formats and parameters.
        """
        # Check if linelist already exists, override with new inputs if provided
        if linelist is not None:
            # The method names are self explaining, see the respective methods for more details on their process
            linelist_wl, linelist_depths = LineList.validate_linelist(linelist)
            linelist_wl, linelist_depths = LineList.drop_invalid_lines(linelist_wl, linelist_depths, verbose=self.config.verbose)

            # Check if the new linelist is different from the existing one
            overwriting = False
            if self._linelist is not None:
                if len(self._linelist["wavelengths"]) != len(linelist_wl) or len(self._linelist["depths"]) != len(linelist_depths):
                    overwriting = True
                elif not np.allclose(self._linelist["wavelengths"], linelist_wl) or not np.allclose(self._linelist["depths"], linelist_depths):
                    overwriting = True

            # Set new linelist
            self._linelist = {"wavelengths": linelist_wl, "depths": linelist_depths}

            # If overwriting, reset variables and warn
            if overwriting:
                if self.config.verbose >= 1:
                    print("Warning: the input linelist has been modified. \n" \
                    f"Resetting variables that need to be recalculated.\nThe velocity grid and input arrays will not be reset.")
                self.reset(preserve_input_profile_groups=False)

    def plot_linelist(self, idx:np.ndarray|list|None=None, bounds:tuple|list|None=None, return_fig:bool=False) -> None|tuple:
        """
        Plots the linelist points with their corresponding depths as delta-function lines.

        Parameters
        ----------
        idx : np.ndarray or list, optional
            The indices of the linelist points to plot. If None, plots all points.
        bounds : tuple or list, optional
            The wavelength bounds (min, max) to clip the linelist for plotting. If None, plots all points.
        return_fig : bool, optional
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple or None
            If return_fig is True, returns a tuple of (figure, axis) objects. Otherwise, returns None.
        """
        if self.linelist is None:
            raise ValueError("No linelist found. Please set a linelist before trying to plot it.")
        wl = self.linelist["wavelengths"]
        depths = self.linelist["depths"]

        if idx is not None:
            wl = wl[idx]
            depths = depths[idx]

        # Clip the linelist to the specified bounds if provided, and to the min_depth
        if bounds is not None:
            idx_in_bounds = (wl >= bounds[0]) & (wl <= bounds[1])
            wl = wl[idx_in_bounds]
            depths = depths[idx_in_bounds]

        # Plot linelist
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.vlines(wl, 0, depths, color='C0', )
        ax.set_title('Line List')
        ax.set_xlabel('Wavelength (Angstroms)')
        ax.set_ylabel('Relative Line Depth')
        ax.legend()
        if return_fig:
            return fig, ax
        utils.show_or_save(plt, self.config.figure_dir, "linelist.png", self.config.verbose)

    # Store config as a property for handling it to/from dictionary on saving
    @property
    def config(self) -> Config:
        """Returns the internally stored config object, which contains the configuration of the ACID run."""
        return self._config

    @config.setter
    def config(self, value: Config) -> None:
        """Sets the internally stored config object."""
        self._config = value

    def set_inputs(
        self,
        input_wavelengths: Array1D|Array2D|None        = None,
        input_flux:        Array1D|Array2D|None        = None,
        input_errors:      Array1D|Array2D|None        = None,
        input_sn:          Array1D|Array2D|Scalar|None = None,
        skips:             IntLike|None                = None,
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
            Allows you to set and override the Config value for skips if skips is not None.
            This allows the inputs to only set one in every skips pixels and is only recommended for testing.
            By default None, and uses the Config default value (1, no skipping).
        """
        # Check if inputs already exist, use a key to name dictionary map to get boolean for if any/all exist for following logic checks
        input_keys = ["wavelengths", "flux", "errors"]
        inputs = {
            "wavelengths": input_wavelengths,
            "flux": input_flux,
            # As of 1.5, SN or errors are guessed from the errors if one is not input, so treat them as a pair
            "errors": input_errors if input_errors is not None else input_sn,
        }
        inputs_already_exist = all(
            getattr(self, attr).get("input", None) is not None for attr in input_keys
        )
        all_inputs_not_none = all(inputs[attr] is not None for attr in input_keys)
        any_inputs_not_none = any(inputs[attr] is not None for attr in input_keys)
        del inputs # it was just a trick to do the input checks in a loop

        # Handle logic for already existing inputs, more or less described in the print statements
        if inputs_already_exist:
            if not all_inputs_not_none and any_inputs_not_none:
                if self.config.verbose >= 1:
                    print(f"Warning: input wavelengths, flux, and errors are already set in the class. \n" \
                        f"Some of the inputs you provided are None. \n" \
                        f"If you are trying to update the input wavelengths, flux, or errors, you must provide all 3. \n"
                        f"The current input wavelengths, flux, and errors will be kept.")
                self.combine_spec(output=False)
                return
            elif not any_inputs_not_none:
                if self.config.verbose >= 3:
                    print("Input wavelengths, flux, and errors are already set in the class. Keeping existing values.")
                self.combine_spec(output=False)
                return
            # Else continue with the rest of the function to update inputs, later on, the code will check if new inputs are 
            # different from the existing ones, if so, deletes variables that need to be recalculated.
        else:
            if not all_inputs_not_none:
                raise ValueError("input_wavelengths, input_flux, and (input_errors or input_sn) must be provided either as arguments " \
                                 "or in the form of a Data object.")
        
        # First check we have not received len=0 or len=1 for wavelengths and flux so that they do not collapse to 0 dimensions on squeezing
        if input_wavelengths is not None and len(input_wavelengths) <= 1:
            raise ValueError("input_wavelengths must have more than 1 value to be valid.")
        if input_flux is not None and len(input_flux) <= 1:
            raise ValueError("input_flux must have more than 1 value to be valid.")
        if input_errors is not None and len(input_errors) <= 1:
            raise ValueError("input_errors must have more than 1 value to be valid.")

        # Convert to arrays, squeeze to remove extra dimensions (as default in legacy inputs)
        try:
            input_wavelengths = np.array(input_wavelengths).squeeze()
            input_flux = np.array(input_flux).squeeze()
            input_errors = np.array(input_errors).squeeze() if input_errors is not None else None
            input_sn = np.array(input_sn).squeeze() if input_sn is not None else None
        except:
            raise ValueError("There was an error converting the input wavelengths, flux, errors, or SN to numpy arrays. Exception message: \n" + tb.format_exc() + "\n")

        # Make any values < 0 or infinite equal to nan, which are gracefully later handled.
        if input_errors is not None:
            input_wavelengths, input_flux, input_errors = utils.mask_invalid(input_wavelengths, input_flux, input_errors, verbose=self.config.verbose)
        else:
            input_wavelengths, input_flux = utils.mask_invalid(input_wavelengths, input_flux, verbose=self.config.verbose)

        # Check that none of the inputs are all nan
        if np.all(np.isnan(input_wavelengths)) or np.all(np.isnan(input_flux)) or (input_errors is not None and np.all(np.isnan(input_errors))):
            raise ValueError("None of the input wavelengths, spectra, and errors can be all NaN. Check your inputs for invalid or negative values")

        # Get SN or errors if one is not provided
        if input_sn is None and input_errors is None:
            raise ValueError("One of input_sn or input_errors must be provided.")

        elif input_errors is not None:
            input_sn = utils.guess_SNR(input_wavelengths, input_flux, input_errors)
            if self.config.verbose >= 2:
                print(f"No input_sn provided and was instead approximated. Guessed value(s):\n {input_sn}")

        elif input_sn is not None:
            # Input SN can accidentally be input with the same shape as the wavelengths, so correct now if thats the case
            if input_sn.ndim == input_wavelengths.ndim:
                if self.config.verbose >= 2:
                    print("Per pixel S/N provided, taking the mean over the central 2/3 of the wavelengths to get a single S/N value for each frame.")
                # Per pixel S/N provided, take the mean over the central 2/3 of the wavelengths
                input_sn = utils.collapse_SNR(input_sn, input_wavelengths)

            input_errors = utils.guess_errors(input_flux, input_sn)
            if self.config.verbose >= 1:
                print(f"No input_errors provided and was instead approximated from the input S/N.\n"\
                      f"It is highly recommended to obtain correct per-pixel errors.")

        # Check they have matching shape
        if not input_wavelengths.shape == input_flux.shape == input_errors.shape:
            raise ValueError("Input wavelengths, spectra and spectral errors must all have the same shape.")

        # Ensure now that the SN becomes just a single value per frame
        if input_sn.ndim == input_wavelengths.ndim:
            if self.config.verbose >= 2:
                print("Per pixel S/N provided, taking the mean over the central 2/3 of the wavelengths to get a single S/N value for each frame.")
            # Per pixel S-N provided, take the mean over the central 2/3 of the wavelengths
            input_sn = utils.collapse_SNR(input_sn, input_wavelengths)
        elif input_sn.ndim != input_flux.ndim-1:
            raise ValueError("input_sn must be either a single-valued list/array with the average S/N for each frame, " \
            f"or an array of S/N values for each pixel. \n" \
            "The shape of the input input_sn does not match the number of frames in input_flux, " \
            "nor does it have one more dimension than input_flux.")
        if input_sn.ndim != input_flux.ndim - 1:
            raise ValueError(f"input_sn.ndim and input_flux.ndim-1 do not match, sn ndim = {input_sn.ndim}, flux ndim = {input_flux.ndim}")
        

        # Ensure all inputs are at least 2D (with the first dimension being the frame number), 
        # to ensure consistent handling of single-frame and multi-frame inputs. 
        input_wavelengths = np.atleast_2d(input_wavelengths)
        input_flux = np.atleast_2d(input_flux)
        input_errors = np.atleast_2d(input_errors)
        input_sn = np.atleast_1d(input_sn)

        if input_sn.shape[0] != input_flux.shape[0]:
            raise ValueError("The number of frames for the SN must match the number of frames in wavelengths, flux, and errors.")

        # Ensure data is sorted by wavelength
        sort_idx = np.argsort(input_wavelengths, axis=-1)
        input_wavelengths = np.take_along_axis(input_wavelengths, sort_idx, axis=-1)
        input_flux = np.take_along_axis(input_flux, sort_idx, axis=-1)
        input_errors = np.take_along_axis(input_errors, sort_idx, axis=-1)

        # Apply skips, this just skips some data for testing and faster runs, but real runs should always leave skips=1
        self.config.skips = skips # no change if skips is None
        input_wavelengths = input_wavelengths[:, ::self.config.skips]
        input_flux       = input_flux[:, ::self.config.skips]
        input_errors     = input_errors[:, ::self.config.skips]

        # In case these are set when input values already exist, check if they are the same, if not, reset variables to be recalculated.
        # This checks basically if self.wavelengths["input"] is the same as input_wavelengths, and same for flux and errors, if they exist. 
        overwriting = False
        for check in ["wavelengths", "flux", "errors", "sn"]:
            if getattr(self, check).get("input", None) is not None and eval(f"input_{check}") is not None:
                if getattr(self, check)["input"].shape != eval(f"input_{check}").shape:
                    overwriting = True
                elif not np.allclose(getattr(self, check)["input"], eval(f"input_{check}"), equal_nan=True):
                    overwriting = True

        # Set inputs to class variables, the self.reset() cleans all arrays except for the inputs, so this is safe
        self.wavelengths["input"] = input_wavelengths
        self.flux["input"]        = input_flux
        self.errors["input"]      = input_errors
        self.sn["input"]          = input_sn

        # If reset is needed, reset calculated values to force recalculation with new inputs and warn the user
        if overwriting:
            if self.config.verbose >= 1:
                print("Warning: input wavelengths, flux, or errors have been changed from their previous values. \n" \
                f"Resetting variables that need to be recalculated.\nThe velocity grid and linelist will not be reset.")
            self.reset(preserve_combined=False)

        # Now generate the combined dataset (previously done in Acid class)
        # Combines spectra from each frame (weighted based of S/N), returns to S/N of combined spectra.
        # If only one frame, just uses that frame. We also check if this step has already been done and skips if so.
        self.combine_spec(output=False)

    def combine_spec(
        self,
        frame_wavelengths: Array1D|Array2D|None = None,
        frame_flux:        Array1D|Array2D|None = None,
        frame_errors:      Array1D|Array2D|None = None,
        frame_sns:         Array1D|Array2D|None = None,
        output:            bool                 = True
        ) -> tuple | None:
        """
        Combines the multiple inputted spectral frames into one spectrum, or just passes through the single frame if only one was input. 
        The frames are interpolated onto a common wavelength grid of the spectrum with the highest S/N, and then a weighted average is used based on the errors. 
        The S/N of the combined spectrum is also calculated based on the input S/N and the weights.

        Parameters
        ----------
        frame_wavelengths : :py:type:`Array1D` | :py:type:`Array2D`, optional
            Wavelengths for the spectral frames, by default None
        frame_flux : :py:type:`Array1D` | :py:type:`Array2D`, optional
            Fluxes for the spectral frames, by default None
        frame_errors : :py:type:`Array1D` | :py:type:`Array2D`, optional
            Errors for the spectral frames, by default None
        frame_sns : :py:type:`Array1D` | :py:type:`Array2D`, optional
            Signal-to-noise ratio for the spectral frames, by default None
        output : bool, optional
            Whether to output the combined spectrum, by default True

        Returns
        -------
        tuple | None, if output is True, containing:
            combined_wavelengths : np.ndarray
                Wavelengths for the combined spectrum
            combined_spectrum : np.ndarray
                Fluxes for the combined spectrum
            combined_errors : np.ndarray
                Errors for the combined spectrum
            combined_sn : float
                Signal-to-noise ratio for the combined spectrum
        None, if output is False, but the combined spectrum is still saved in the data class attributes.
        """

        if frame_wavelengths is not None: # This should only be for testing
            self.wavelengths["input"] = np.copy(frame_wavelengths)
            self.flux["input"]        = np.copy(frame_flux)
            self.errors["input"]      = np.copy(frame_errors)
            self.sn["input"]          = np.copy(frame_sns)

        # Set simple names for variables (just used in this function)
        wavelengths = np.copy(self.wavelengths["input"])
        flux        = np.copy(self.flux["input"])
        errors      = np.copy(self.errors["input"])
        sn          = np.copy(self.sn["input"])

        # Return as is if only one spectrum
        if len(self.wavelengths["input"])==1:
            combined_wavelengths = wavelengths[0]
            combined_flux        = flux[0]
            combined_errors      = errors[0]
            combined_sn          = sn[0]

            self.wavelengths["combined"] = combined_wavelengths
            self.flux["combined"]        = combined_flux
            self.errors["combined"]      = combined_errors
            self.sn["combined"]          = combined_sn

        else:
            # Get wavelength grid with highest S/N
            combined_wavelengths = wavelengths[np.argmax(sn)]

            interpolated_flux   = np.zeros_like(flux)
            interpolated_errors = np.zeros_like(errors)

            # combine all spectra to one spectrum
            for n in range(len(flux)):

                # Interpolate each spectrum onto the combined wavelength grid
                f2 = interp1d(wavelengths[n], flux[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
                f2_err = interp1d(wavelengths[n], errors[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
                interpolated_flux[n] = f2(combined_wavelengths)
                interpolated_errors[n] = f2_err(combined_wavelengths)

                # Mask out out extrapolated areas
                idx_ex = np.logical_and(combined_wavelengths<=np.max(wavelengths[n]),
                                        combined_wavelengths>=np.min(wavelengths[n]))
                idx_ex = tuple([idx_ex==False])

                interpolated_flux[n][idx_ex] = 1.
                interpolated_errors[n][idx_ex] = 1e12

                # Mask out nans and zeros (these do not contribute to the main spectrum)
                where_are_NaNs = np.isnan(interpolated_flux[n])
                interpolated_errors[n][where_are_NaNs] = 1e12
                where_are_zeros = np.where(interpolated_flux[n] == 0)[0]
                interpolated_errors[n][where_are_zeros] = 1e12

                where_are_NaNs = np.isnan(interpolated_errors[n])
                interpolated_errors[n][where_are_NaNs] = 1e12
                where_are_zeros = np.where(interpolated_errors[n] == 0)[0]
                interpolated_errors[n][where_are_zeros] = 1e12

            invvars = 1 / interpolated_errors**2
            invvars[interpolated_errors >= 1e12] = 0

            weights = np.sum(invvars, axis=0)
            non_zero = weights > 0
            
            weighted_flux   = np.sum(interpolated_flux * invvars, axis=0)
            
            combined_flux = np.full_like(weights, 1.0)      # or np.nan
            combined_errors = np.full_like(weights, 1e12)

            combined_flux[non_zero] = weighted_flux[non_zero] / weights[non_zero]
            combined_errors[non_zero] = 1 / np.sqrt(weights[non_zero])

            frame_weights = np.sum(invvars, axis=1)
            combined_sn   = np.sum(frame_weights * sn) / np.sum(frame_weights)

            self.wavelengths["combined"] = combined_wavelengths
            self.flux["combined"]        = combined_flux
            self.errors["combined"]      = combined_errors
            self.sn["combined"]          = combined_sn

        # Clean combined spectra of NaNs
        wavelengths, flux, errors, nanmask = utils.drop_invalid(self.wavelengths["combined"], self.flux["combined"],
                                                                self.errors["combined"], return_mask=True)
        self.wavelengths["combined"] = wavelengths
        self.flux["combined"] = flux
        self.errors["combined"] = errors
        self.nanmask = nanmask

        if output is True:
            # ie if called as a function rather than from ACID function
            return (
            self.wavelengths["combined"],
            self.flux["combined"],
            self.errors["combined"],
            self.sn["combined"]
        )

    def reset(self, preserve_combined:bool=True, preserve_input_profile_groups:bool=True) -> None:
        """
        Resets all derived states while preserving:
        - raw input arrays
        - combined spectrum, unless explicitly invalidated
        - manually supplied profile groups, unless explicitly invalidated
        - linelist
        - velocity grid
        - Config
        """
        # Preserve desired inputs
        inputs = {}
        for name in ("wavelengths", "flux", "errors", "sn"):
            value = getattr(self, name).get("input", None)
            inputs[name] = None if value is None else value.copy()

        combined = {}
        for name in ("wavelengths", "flux", "errors", "sn"):
            value = getattr(self, name).get("combined", None)
            combined[name] = copy.deepcopy(value)

        nanmask = (
            None if self.nanmask is None
            else self.nanmask.copy()
        )

        input_profile_groups = (
            None if self.input_profile_groups is None
            else self.input_profile_groups.copy()
        )

        # Preserve these directly rather than through their property setters,
        # otherwise the setters can trigger reset() again.
        linelist = copy.deepcopy(self._linelist)
        velocities = (
            None if self._velocities is None
            else self._velocities.copy()
        )
        config = self._config

        # Reinitialise every dataclass field to its default.
        self.__init__()

        # Restore persistent state
        self._config = config
        self._linelist = linelist
        self._velocities = velocities

        # Restore desired values
        if preserve_combined:
            for name, value in combined.items():
                if value is not None:
                    getattr(self, name)["combined"] = value
            self.nanmask = nanmask

        if preserve_input_profile_groups:
            self.input_profile_groups = input_profile_groups

        # Restore raw inputs
        for name, value in inputs.items():
            if value is not None:
                getattr(self, name)["input"] = value

    def plot_continuum_fit(self, key:str="masked", return_fig:bool=False, save_fig:str|None=None) -> None:
        """
        Plots the result of the continuum fitting step, showing the original spectrum, the fitted continuum, and the clipped points used for the continuum fit.

        Parameters
        ----------
        plot_type : str, optional
            The type of continuum fit to plot, either "initial" for the initial continuum fit or
            "masked" for the continuum fit after residual masking. Default is "masked".
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False.
        save_fig : str or None, optional
            If provided, the path to save the figure. If None, the figure will not be saved. Default is None.
        """
        # Check we have all inputs needed for plot
        if key not in ["initial", "masked"]:
            raise ValueError("plot_type must be either 'initial' or 'masked'")
        if key not in self.plotting_variables:
            raise ValueError(f"No plotting variables found for plot_type={key!r}. " \
                             "Please ensure that the continuum fit has been performed for this plot_type.")
        if not all(
            attr in self.plotting_variables[key] for attr in [
                "clipped_waves", "clipped_flux", "good"]
            ):
            raise ValueError("To plot the continuum fit, please first run the continuum fitting in ACID step for the specified plot_type.")

        # Unpack variables
        good                     = self.plotting_variables[key]["good"]
        clipped_waves            = self.plotting_variables[key]["clipped_waves"]
        clipped_flux             = self.plotting_variables[key]["clipped_flux"]

        # Normalise wavelengths and plot flux and fit
        fig, ax = plt.subplots(figsize=(15, 9))
       
        ax.plot(self.wavelengths[key], self.flux[key], label='Original Spectrum', color="C0", alpha=0.7)
        ax.plot(self.wavelengths[key], self.continuum[key], label='Fitted Continuum', color='red')
        a, b = utils.get_normalisation_coeffs(self.wavelengths[key])
        ax.plot((clipped_waves[good]-b)/a, clipped_flux[good], 'o', label='Continuum Normalized Spectrum', color='green')

        # Plot the linelist points, with a color corresponding to their depth in the linelist within the range
        # Only plot the 20 strongest lines to avoid overcrowding.
        ll_wl = self.linelist["wavelengths"]
        ll_depths = self.linelist["depths"]
        from .lsd import LSD
        ll_wl, ll_depths, _ = LSD.clip_wavelengths(self.wavelengths[key], ll_wl, ll_depths)
        idx = np.argsort(ll_depths)
        ll_wl = ll_wl[idx]
        ll_depths = ll_depths[idx]
        ll_wl = ll_wl[-20:]
        ll_depths = ll_depths[-20:]

        # Try colouring them, but often the linelist points will be outside the wavelength range so just skip if 
        # there's an error to avoid breaking the plot
        try:
            cmap = plt.cm.viridis_r
            norm = mpl.colors.Normalize(vmin=np.nanmin(ll_depths), vmax=np.nanmax(ll_depths))
            for i, (wl, depth) in enumerate(zip(ll_wl, ll_depths)):
                ax.axvline(
                    wl,
                    color=cmap(norm(depth)),
                    linestyle="--",
                    alpha=1,
                    label="Line List (20 strongest lines in region)" if i == 0 else None,
                )
            # Create colorbar for depth
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])  # needed for some matplotlib versions
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label("Line depth")
        except:
            if self.config.verbose >= 1:
                print("There was an error plotting the linelist points, most likely your linelist range is outside your wavelength range.")
            pass

        # Plot the line masks with their names
        x = self.wavelengths[key]
        line_mask = self.config.masking_lines.get_masks(x, with_names=True)
        for i, (name, masks) in enumerate(line_mask.items()):
            padded = np.concatenate(([False], masks, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for j, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]), color=f'C{i+2}', alpha=0.3,
                        label=f"{name} Line masks" if j == 0 else None)
        
        # Plot the other two masking regions if in the masked plot type, these masking are only done after the initial fit
        if key == "masked":
            masked = self.pix_mask | self.sigma_mask
            padded = np.concatenate(([False], masked, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for i, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]),
                            color='red', alpha=0.15, label="Sigma masking and pixel deviation regions" if i == 0 else None)

        # Add labels and legend, and save or show figure
        plot_title = "Initial Continuum Fit" if key == "initial" else "Continuum Fit after Residual Masking"
        ax.set_title(plot_title)
        ax.legend()
        ax.set_ylim(np.min(self.flux[key])*0.9, np.max(self.flux[key])*1.1)
        if save_fig is not None:
            plt.savefig(save_fig)
        if return_fig:
            return fig, ax
        title = "continuum_fit_initial.png" if key == "initial" else "continuum_fit_masked.png"
        utils.show_or_save(plt, self.config.figure_dir, title, self.config.verbose)

    def plot_residual_masking(self) -> None:
        """
        Creates 3 plots to show the result of the residual masking step, showing the residuals with the sigma clipping thresholds, 
        the masked regions, and the initial profile after masking.
        """
        # Check we have all inputs needed for plot
        if "masked" not in self.plotting_variables:
            raise ValueError("No plotting variables found for masking. Residual masking likely has not been performed in Acid.")
        if "masked" not in self.wavelengths and "masked" not in self.flux:
            raise ValueError("No masked wavelengths or fluxes found. Please ensure that the residual masking step has been performed")

        # Unpack variables
        x = self.wavelengths["combined"]
        y = self.flux["combined"]
        residuals = self.residuals["masked"]
        upper_clip = self.plotting_variables["masked"]["upper_clip"]
        lower_clip = self.plotting_variables["masked"]["lower_clip"]
        uc_finite = np.isfinite(upper_clip) # useful for plotting
        lc_finite = np.isfinite(lower_clip)
        pix_mask = self.pix_mask
        line_mask = self.line_mask
        full_mask = self.full_mask

        nremoved = np.sum(full_mask)
        if self.config.verbose >= 2:
            print(f"{nremoved}/{len(residuals)} pixels were removed after residual masking.")

        # Create plot and add residuals with sigma clipping thresholds and masked regions
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        utils.plot_masked_line(ax, x, residuals, ~full_mask, colors=["blue", "red"], label=["Residuals", "Masked Residuals"])

        # Show sigma clipping
        if uc_finite:
            ax.axhline(upper_clip, color='C0', linestyle='--', label='Sigma Clip Thresholds', linewidth=2)
        if lc_finite:
            lc_label = 'Sigma Clip Thresholds' if not uc_finite else None
            ax.axhline(lower_clip, color='C0', linestyle='--', linewidth=2, label=lc_label)

        # Show line masking regions
        line_mask = self.config.masking_lines.get_masks(x, with_names=True)
        for i, (name, masks) in enumerate(line_mask.items()):
            padded = np.concatenate(([False], masks, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for j, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]), color=f'C{i+2}', alpha=0.3,
                           label=f"{name.capitalize()} line masks" if j == 0 else None)

        # Show pix_chunk masked points:
        masked = pix_mask
        padded = np.concatenate(([False], masked, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i, (start, end) in enumerate(zip(starts, ends)):
            ax.axvspan((x[start]), (x[end-1]),
                        color='red', alpha=0.15, label="Applied chunk deviation masking" if i == 0 else None)
        
        # And show chunk deviation range
        dev = self.config.dev_perc / 100
        ax.hlines([-dev, dev], xmin=np.min(x), xmax=np.max(x), color='C1', linestyle='--', linewidth=2, label="Chunk deviation masking range")

        # Set a good ylim off everything but the masked points
        ymax = np.nanmax([dev, upper_clip if uc_finite else np.nan, np.max(residuals[~full_mask])])
        ymin = np.nanmin([-dev, lower_clip if lc_finite else np.nan, np.min(residuals[~full_mask])])
        diff = (ymax - ymin) * 0.1 # Set an even 10% buffer on either side of max/min
        ax.set_ylim(ymin - diff, ymax + diff)

        ax.set_xlim(np.min(x), np.max(x))
        ax.grid(True)
        ax.set_title('Residuals with Sigma Clipping Thresholds')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Residuals')
        ax.legend()
        utils.show_or_save(plt, self.config.figure_dir, "masking_residuals.png", self.config.verbose)

        # Plot the LSD profile
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.plot(self.velocities, self.profile["masked"][0], label='LSD Profile after Masking and before sampling', color='red')
        except:
            ax.plot(self.velocities, np.median(self.profile["masked"][0], axis=0), label='MEDIAN LSD Profile after Masking and before sampling', color='red')
        ax.set_title('LSD Profile after Residual Masking')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('LSD Profile')
        ax.axhline(1, color='black', linestyle='--')
        ax.legend()
        ax.grid(True)
        utils.show_or_save(plt, self.config.figure_dir, "masking_profile.png", self.config.verbose)

        # Finally plot the forward model
        x = self.wavelengths["combined"]
        y = self.flux["combined"]
        forward = self.forward_y["masked"]
        continuum = self.continuum["masked"]
        residuals = (y - forward) / forward
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax[0].plot(x, y, label='Original data', color='black', linewidth=1)
        ax[0].plot(x, continuum, color='C1', linewidth=1, label='Fitted Continuum', linestyle='--')
        utils.plot_masked_line(ax[0], x, forward, ~full_mask, label=["Forward model", "Masked Forward model"])
        ax[0].set_title('Masked Forward Model')
        ax[0].set_ylabel('Flux')
        ax[0].grid(True)
        ax[0].legend()
        
        utils.plot_masked_line(ax[1], x, residuals, ~full_mask, label=["Residuals", "Masked Residuals"])
        ax[1].axhline(0, color='black', linestyle='--', linewidth=1)

        ymax = np.max(residuals[~full_mask])
        ymin = np.min(residuals[~full_mask])
        diff = (ymax - ymin) * 0.1 # Set an even 10% buffer on either side of max/min
        ax[1].set_ylim(ymin - diff, ymax + diff)

        ax[1].grid(True)
        # ax[1].set_title('Residuals of forward model with masked residuals')
        ax[1].set_xlabel('Wavelength')
        ax[1].set_ylabel('Residuals')
        ax[1].legend()
        plt.tight_layout()

        utils.show_or_save(plt, self.config.figure_dir, "initial_forward_model.png", self.config.verbose)

    def save(self, save_path:str|None=None, sampler_path:str|None=None) -> None:
        """
        Saves the data object to a file using pickling. This will store just the dictionary of the class, 
        not the actual class itself. The load function then will initialise a new Data class using the dictionary.

        Parameters
        ----------
        save_path : str | None
            The path to save the data object. If None, uses the path stored in the config.
            If That is also None, the data object will not be saved.
            Will attempt to create the directory for the filepath if it does not exist.
            The file must end with .pkl to be recognised as a pickled file.
        sampler_path : str | None
            This is used for saving the sampler as a HDF5 file if it has not already been set up as such.
            If a .h5 file is provided, and the sampler is not already saved as a HDF5 file, the sampler backend is converted to a HDF5 backend and saved.
            If the sampler is already set up to save as a HDF5 file, this argument is ignored. 
            If you want to move the file location, move it yourself and set Data.sampler = sampler_path to update the location in the Data object.
            If None, this will do nothing.
        """
        # First we handle the sampler saving, so that when saved it can be added to the sampler path saved in the dictionary later
        if sampler_path is not None:
            # Convert to abspath
            sampler_path = os.path.abspath(sampler_path)
            if self.sampler is not None:
                if not isinstance(self.sampler.backend, HDFBackend):
                    if not sampler_path.endswith(".h5"):
                        raise ValueError("sampler_path must end with .h5 to convert and save the sampler backend as a HDF5 file.")
                    os.makedirs(os.path.dirname(sampler_path), exist_ok=True) # create directory if it does not exist
                    utils.save_backend_to_hdf5(self.sampler.backend, sampler_path)
                    self.config.sampler_path = sampler_path # update config with sampler path for future reference
                    if self.config.verbose >= 2:
                        print(f"Sampler backend converted and saved as HDF5 file to {sampler_path}")
                else:
                    if self.config.verbose >= 2:
                        print("Sampler is already set up to save as a HDF5 file, ignoring sampler_path argument.")
            else:
                if self.config.verbose >= 1:
                    print("Cannot save sampler as sampler does not exist.")

        # TODO: NEEDS UPDATING WITH NEW CONFIG DIR, maybe move them all to Config properties?
        # Now save the Data object itself, with the sampler path included to be used when reloaded
        save_path = os.path.abspath(save_path) if save_path is not None else None
        self.config.save_path = save_path # update and overwrite config with save path for future reference
        save_path = self.config.save_path # now use the save path in the config
        if save_path is None:
            if self.config.verbose >= 1:
                print("No save_path exists or was provided. The Data instance will not be saved.")
            return

        payload = self.to_dict() # generates a dictionary of the data object for easy pickling

        save_dir = os.path.abspath(save_path)
        if save_dir:
            os.makedirs(os.path.dirname(save_dir), exist_ok=True) # create directory if it does not exist
        with open(save_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        if self.config.verbose >= 2:
            print(f"Data object saved to {save_path}")

    @classmethod
    def load(cls, filename:str) -> Data:
        """
        Loads a data object from a file using pickling. This will read the dictionary from the file and 
        then use it to initialise a new Data class.

        Parameters
        ----------
        filename : str
            The name of the file to load the data object from. This should be a .pkl file.
        Returns
        -------
        Data
            The loaded data object.
        """
        with open(filename, "rb") as f:
            payload = pickle.load(f)

        # Initialise a new Data object and update it with the payload dictionary
        return cls().from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the data object to a dictionary payload for saving. This is used internally in the save method, 
        but can also be used for debugging or other purposes.
        """
        if self.sampler is not None and self.config.sampler_type == "dynesty":
            raise ValueError("Storing the sampler is not currently supported for dynesty samplers.\n" \
            "If you really want to, separate the sampler with data.sampler.save('sampler') and add it back later.\n")

        payload: dict[str, Any] = {}
        for f in fields(self):
            name = f.name
            val = getattr(self, name)

            if name == "_config":
                payload["config"] = val.to_dict() # store as dict in payload, but store as class in Data
            elif name == "_sampler":
                if val is not None:
                    if self.config.sampler_path is not None:
                        payload["sampler"] = self.config.sampler_path # stores just the path to the sampler
            else:
                payload[name] = val

        return payload

    def from_dict(self, payload:dict[str,Any]) -> Data:
        """
        Updates the data object from a dictionary payload. This is used internally in the 
        load method, but can also be used for debugging or other purposes.

        Parameters
        ----------
        payload : dict
            The dictionary payload to update the data object from. This should have the same keys as the
            attributes of the data class. The "config" key should be a dictionary 
            that can be used to initialise a Config class.
        """
        for f in fields(self):
            name = f.name
            if name == "_config": # config stored as a dict in payload, but stored here as class
                cfg_dict = payload.get("config", {})
                setattr(self, "_config", Config(**cfg_dict))
            elif name == "_sampler":
                continue # handled after loop
            else:
                if name in payload:
                    setattr(self, name, payload[name])

        # Handle sampler separately
        self.sampler = payload.get("sampler", None) # property handles the loading of the sampler
        if self.sampler is None and self.config.sampler_path is not None:
            try:
                self.sampler = self.config.sampler_path
            except:
                self.sampler = None

        return self

    @property
    def result(self):
        if self.exception is not None:
            if self.config.verbose >= 1:
                print(f"An exception was raised during the run, cannot return results object.\n"
                      f"Returning None instead.")
            return None
        if self.complete is False:
            if self.config.verbose >= 1:
                print(f"Results for order {self.config.order} have not yet been calculated, cannot return results object.\n"
                      f"Returning None instead.")
            return None
        from .result import Result
        return Result(self)

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

@beartype
class DataList:
    """
    A class that stores Data instances in a list indexed by order. The DataList is a useful class for running ACID over multiple orders with parallelization. 
    Fundamentally this class holds Data instances (which ACID updates with the results per order) as a list and can map the true order number
    from the instrument (stored in the config) to the index of the list. It handles missing/incomplete orders, and the ability to append new orders.
    For more information and a full example on how to use the DataList, see :ref:`datalist' in the documentation. Note that the DataList is not 
    strictly necessary to run ACID over multiple orders, you can handle the multiple instances yourself.

    The DataList class works with a required root directory specified by the user to access to the same data across parallel processes, 
    and also to save intermediate results and figures per order.
    """

    def __init__(
        self,
        wavelengths      : Array3D|Array2D|None           = None,
        flux             : Array3D|Array2D|None           = None,
        errors           : Array3D|Array2D|None           = None,
        sn               : Array2D|Array1D|None           = None,
        velocities       : Array1D|None                   = None,
        linelist         : Array2D|None|str|LineList|dict = None,
        order_range      : Array1D|None                   = None,
        config           : Config|list[Config]|None       = None,
        save_dir         : str|None                       = None,
        overwrite        : bool                           = False,
        verbose          : IntLike|bool|str|None          = None,
        _load                                             = None,
        _data_list       : list[Data]|None                = None,
        **config_kwargs,
        ) -> None:
        """
        Initializes the DataList object. The DataList can be initialized in two ways: either by providing the wavelengths, flux, errors, and sn arrays directly in
        the class initialization (here), or using the :py:classmethod:`DataList.from_datalist` method with a list of Data objects. 
        The former is useful for quickly initializing a DataList from raw data, while the latter is useful 
        for loading a saved DataList or for more fine-grained control over the initialization of each Data object.

        Parameters
        ----------
        wavelengths : :py:type:`Array3D` | :py:type:`Array2D` | None, optional
            A 2D or 3D array of wavelengths for the input spectra.
            If a 2D array is provided, it is assumed to have shape (n_orders, n_pixels).
            If a 3D array is provided, it is assumed to have shape (n_orders, n_frames, n_pixels). Default is None.
            The format for the last 1 or 2 dimensions follows that of the "wavelengths" input in the :py:function:`Acid.ACID` method.
            Sometimes, fits files store their frames in shape (n_frames, n_orders, n_pixels), you can swap the axes with np.swapaxes(wavelengths, 0, 1) 
            to get them in the correct shape. It is also possible to input orders with different numbers of pixels, in which case the wavelengths should be a list
            of 2D arrays/lists.
        flux : :py:type:`Array3D` | :py:type:`Array2D` | None, optional
            A 2D or 3D array of fluxes for the input spectra. Same shape assumptions as wavelengths. Default is None.
        errors : :py:type:`Array3D` | :py:type:`Array2D` | None, optional
            A 2D or 3D array of errors for the input spectra. Same shape assumptions as wavelengths. Default is None.
        sn : :py:type:`Array2D` | :py:type:`Array1D` | None, optional
            A 1D or 2D array of signal-to-noise ratios for the input spectra. If a 1D array is provided, it is assumed to have shape (n_orders,). 
            If a 2D array is provided, it is assumed to have shape (n_orders, n_frames). Default is None. Follows the same logic as the "sn" input in 
            the :py:function:`Acid.ACID` method, for approximating the errors (or vice versa) if one is not provided.
        velocities : :py:type:`Array1D` | None, optional
            The velocity grid to be used for all the orders. This should be a 1D array of velocity values in km/s. Follows the same format as the "velocities" 
            input in the :py:function:`Acid.ACID` method. Default is None.
        linelist : :py:type:`Array1D` | str | :py:class:`LineList` | dict | None, optional
            The linelist to be used for all the orders. This can be provided in the same formats as the "linelist" input in the :py:function:`Acid.ACID` method.
        order_range : :py:type:`Array1D` | None, optional
            A 1D array of order labels corresponding to the orders in the input data.
            The index of this array should match to the order of the index of the first dimension of the wavelengths, flux, errors, and sn arrays.
            For example, if your input data has 3 orders and they correspond to orders 100, 101, and 102 in the instrument, then you should input order_range = [100, 101, 102].
            If not provided, it is assumed to be a pythonic 0-indexed range of the same length as the number of orders in the input data. Default is None.
        config : :py:class:`Config` | list[:py:class:`Config`] | None, optional
            A template Config object for all orders or a list of Config objects per order containing the configuration for the ACID run.
            If inputting a list, the index and length of the list must match the first dimension of the input data arrays and the order_range.
            These take higher priority than any config_kwargs passed in the initialization.
            Setting 'order' will not have any effect as they will be overwritten by the order numbers in the order_range.
            If not provided, default Config values will be used. Default is None.
        save_dir : str | None, optional
            The default directory to save results and figures for each order.
            By default the DataList will save data.pkl and sampler.h5 to the directory (named by the order number) to in this directory.
            If the Configs or kwargs passed contain their own save_path or sampler_path (see :py:class:`Acid`), those instead are used.
            If None, no saving will be done, this is however, not recommended. Default is None.
        overwrite : bool, optional
            Whether to overwrite existing with new Data instances when using run_ACID, or to load and use existing Data instance if they exist.
            If True, if a Data instance already exists for an order, it will be overwritten with the new Data instance generated from the ACID run for that order.
            Note, that the saving of this new Data instance only applies when run_ACID is run, otherwise it is just held in memory.
            If False, if a Data instance already exists for an order, it will be loaded and used instead of generating a new Data instance from the ACID run for that order.
            Default is False.
        verbose : int | bool | str | None, optional
            The verbosity level for printing information during the initialization. 
            Follows the same format as the "verbose" input in the :py:class:`Config` class. 
            Default is None.
        _load : Any, optional
            Not yet implemented, do not use. The idea is that you can input a Load object which has its own tools to pull s2d data from common instruments 
            such as ESPRESSO, HARPS, etc. If you want to use this feature, please open an issue or contribute a pull request with the implementation.
        _data_list : list[:py:class:`Data`] | None, optional
            This is an internal argument used for initializing the DataList from a list of Data objects in the :py:classmethod:`DataList.from_datalist` method.
        **config_kwargs : 
            Additional keyword arguments to be passed with low priority to all of the generated Config objects.
            These kwargs will join with but NOT overwrite any existing keys in the input Config object(s).
            Setting 'order' will not have any effect as they will be overwritten by the order numbers in the order_range.
            Inputting kwargs not part of the defaults in the Config class will cause an error.
            If not provided, default Config values will be used.
        """

        # Raise if load was used
        if _load is not None:
            raise NotImplementedError(f"The 'load' argument is not yet implemented. \n"
                                      f"The idea is that you can input a Load object which has its own tools to pull s2d data from common "\
                                      f"instruments such as ESPRESSO, HARPS, etc. \nIf you want to use this feature, please open an issue or "\
                                      f"contribute a pull request with the implementation.")

        # Configure verbosity
        self.verbose = Config(verbose=verbose).verbose

        # All orders should have the same velocity grid and line list
        self.velocities = velocities

        # Configure order_range, creates one if not input from the shape of wavelengths
        self.order_range = order_range # if None, will be set later, otherwise self.from_datalist handles the range from configs     

        # Set class attributes
        self._save_dir = None
        self._data_list = None
        self._combined_profile = None
        self._results = None
        self.overwrite = overwrite
        self.excluded_orders = []
        self.save_dir = save_dir

        if _data_list is not None:
            self.data_list = _data_list # datalist property handles the rest
            return

        # From here, the array inputs must be provided
        if order_range is None:
            order_range = np.arange(len(wavelengths), dtype=np.int32)
        else:
            order_range = np.asarray(order_range, dtype=np.int32)
            if not np.all(np.isfinite(order_range)):
                raise ValueError("order_range must only contain finite values.")
            if len(order_range) != len(wavelengths):
                raise ValueError("The length of the order_range must match the number of frames in the input data.")
            order_range = np.round(order_range).astype(np.int32)
        self.order_range = order_range
        
        # Convert config to dict(s) to be reinitialized in each Data instance
        if isinstance(config, list):
            if len(config) != len(order_range):
                raise ValueError("If inputting a list of Config objects, the length of the list must match the length of the order_range and input arrays.\n" \
                f"len(order_range): {len(order_range)}, len(wavelengths): {len(wavelengths)}, len(config): {len(config)}.")
            config_dict = [cfg.to_dict() for cfg in config]
        else:
            config_dict = config.to_dict() if config is not None else {}

        # --- Create datalist of Data instances for each order ---
        datalist = []
        for idx, order in enumerate(self.order_range):
            data = Data() # create data instance

            # If config_dict is a list, take the dict at the current index, otherwise use the same dict for all orders
            config_dict_input = config_dict[idx] if isinstance(config_dict, list) else config_dict
            data.config = Config(**config_dict_input) # create and set config instance with default config dict

            # Update the config with any kwargs passed in the initialization of the DataList, and with the order number for this Data instance
            data.config.update_lowpri(**config_kwargs)
            data.config.update_hipri(order=order) # order must be overwritten and set last by us to ensure it matches with the order of the input data for this index

            # Set the inputs for this Data instance, taking the idx'th element of the input arrays for this order.
            # If errors or sn are not provided, they will be set to None and approximated in the Data class.
            input_errors = errors[idx] if errors is not None else None
            input_sn = sn[idx] if sn is not None else None
            data.set_inputs(wavelengths[idx], flux[idx], input_errors, input_sn)

            # Set linelist and velocities, which always should be same for all orders
            data.linelist = linelist # sets the linelist for this Data instance, which is shared across all orders in the DataList
            data.velocities = velocities

            if self.save_dir is not None:
                save_path = os.path.abspath(os.path.join(self.save_dir, f"order_{order}", "data.pkl"))
                sampler_path = os.path.abspath(os.path.join(self.save_dir, f"order_{order}", "sampler.h5"))
                data.config.update_hipri(save_path=save_path, # set default save path for this order which can be overwritten by user
                                        sampler_path=sampler_path) # set default sampler path for this order which can be overwritten by user

                # Check if file already exists
                if os.path.exists(data.config.save_path):
                    if self.overwrite:
                        if self.verbose >= 2:
                            print(f"File {data.config.save_path} already exists, but will be overwritten (when using run_ACID) due to setting.")
                    else:
                        if self.verbose >= 1:
                            print(f"File {data.config.save_path} already exists. The data for this order will be loaded from this file.")
                        data = Data.load(data.config.save_path) # load the existing data from the file instead of using the newly initialized data
                else:
                    data.save() # save the newly initialized, but mostly empty data instance to the file for future reference and use

            datalist.append(data) # finally append to the datalist

        self.data_list = datalist # datalist property handles the rest

    @classmethod
    def from_datalist(cls, data_list:list[Data]|Data, save_dir:str|None=None, verbose:IntLike|bool|str|None=None) -> DataList:
        """
        Load a DataList from a list of Data objects. This is useful for loading a saved DataList or for more fine-grained 
        control over the initialization of each Data object. All Data objects should be already properly initialised with linelists, velocities,
        configs and inputs, and the DataList will check for consistency across the list (e.g. all orders should have the same velocity grid, etc.).

        Parameters
        ----------
        data_list : list[:py:class:`Data`] | :py:class:`Data`
            A list of Data objects to initialize the DataList from. If a single Data object is provided, it will be converted to a list with one element.
        save_dir : str | None, optional
            The directory to save intermediate results and figures for each order. If None, no saving will be done. Default is None.
        verbose : int | bool | str | None, optional
            The verbosity level for printing information during the initialization. Follows the same format as the "verbose" input in the 
            :py:class:`Config` class. Default is None.

        Returns
        -------
        :py:class:`DataList`
            A DataList object initialized from the provided list of Data objects.
        """
        if isinstance(data_list, Data):
            data_list = [data_list]

        # Configure verbosity, if None, use highest verbosity in list
        if verbose is None:
            verbose = np.max([data.config.verbose for data in data_list])

        # All configs should have the same order_range so that they are internally aware. We just take the first one to 
        # generate the mapping of order to index in the list. The Load class will configure the correct order range based
        # off extracted fits header info (if provided), otherwise the default is a pythonic 0-indexed order range.
        order_range = data_list[0].config.order_range
        if len(data_list) > 1 and verbose >= 1:
            if not all(np.array_equal(data.config.order_range, order_range) for data in data_list):
                print("Warning: Not all Data instances have the same order_range. Taking the longest order range.")

        # Take the order range with the greatest length, 
        max_order_range_idx = np.argmax([len(data.config.order_range) for data in data_list])
        order_range = data_list[max_order_range_idx].config.order_range

        # Check all velocity grids match, store velocities
        v0 = data_list[0].velocities
        for data in data_list:
            if not np.array_equal(data.velocities, v0):
                raise ValueError("All Data instances must have the same velocity grid.")
        velocities = v0

        return cls(
            _data_list  = data_list, # skips initialisation of the empty datalist in __init__
            save_dir    = save_dir,
            verbose     = verbose,
            order_range = order_range,
            velocities  = velocities,
        )

    def __iter__(self):
        yield from self.data_list

    def __getitem__(self, k):
        """
        Allows for indexing the DataList with the order number, e.g. datalist[order_number] 
        will return the Data instance with that order number. Uses the internal order to index mapping to find 
        the correct index in the data list.

        Parameters
        ----------
        k : int
            The order number to index the DataList with.
        
        Returns
        -------
        :py:class:`Data`
            The Data instance with the specified order number.
        """
        return self.data_list[self.o2i[k]]

    def __str__(self) -> str|None:
        return f"DataList with {len(self.data_list)} Data instances, storing the orders: {self.orders} out of a total order range: {self.order_range}"

    def __call__(self, *args, **kwargs):
        """Runs and returns the results of the :py:function:`DataList.run_ACID` method, which runs ACID on the Data instances in the list for the specified orders.
        
        Parameters
        ----------
        See :py:function:`DataList.run_ACID` for the accepted parameters and their descriptions.
        """
        return self.run_ACID(*args, **kwargs)

    def __len__(self):
        return len(self.data_list)

    def __iter__(self):
        yield from self.data_list

    def append(self, data:Data, overwrite:bool=False, extend:bool=False, force_order:IntLike|None=None) -> None:
        """
        Appends a Data instance to the data list. Note that the order range of the class is kept, 
        if you want to set a new order range, use the set_order_range() method first to change it.

        Parameters
        ----------
        data : :py:class:`Data`
            The Data instance to append to the data list. The order of the Data instance is taken from its config, 
            but can be overridden with the force_order argument.
        overwrite : bool, optional
            If True, will overwrite an existing Data instance with the same order number. Default is False.
        extend : bool, optional
            If True, will extend the order range to include the new order if it is not already present. Default is False.
        force_order : int, optional
            If provided, will set the order of the Data instance to this value, overriding its config. Default is None.
        """
        if force_order is not None:
            data.config.order = force_order
        order = data.config.order
        if order in self.orders and overwrite is False:
            raise ValueError(f"A Data instance with order {order} already exists in the list. " \
            "If you want to overwrite it, set overwrite=True in the append method.")
        if order not in self.order_range:
            if not extend:
                raise ValueError(f"The order of the appended data class does not match the rest of the list. \n" \
                                 f"If you want to extend the order_range to append the new order, set extend=True.")
            else:
                self.order_range = np.append(self.order_range, order).astype(np.int32)
        
        if overwrite and order in self.orders:
            self.data_list[self.o2i[order]] = data
        else:
            self.data_list.append(data)

        self.sort_by_order() # re-sorts the list and updates the o2i mapping

    def set_order_range(self, order_range:Array1D) -> None:
        """Sets the order range for the DataList. The new range must be a superset of the already saved orders in the list, 
        otherwise a ValueError is raised.
        
        Parameters
        ----------
        order_range : :py:type:`Array1D`
            The new order range to set for the DataList. This should be a 1D array of order numbers. 
        """
        if np.any([o not in order_range for o in self.orders]):
            raise ValueError("The already saved orders must be a subset of the inputted order_range.")
        self.order_range = np.array(order_range, dtype=np.int32)
        self.sort_by_order() # re-sorts the list and updates the o2i mapping, and injects the new order_range into each config

    def sort_by_order(self) -> None:
        """
        Sorts the data list by order number, and updates the o2i mapping accordingly. Internally called whenever self.data_list is updated.
        """
        self.data_list.sort(key=lambda data: data.config.order)
        self.o2i = {data.config.order: i for i, data in enumerate(self.data_list)}
        self.i2o = {i: data.config.order for i, data in enumerate(self.data_list)}
        self.orders = np.array([data.config.order for data in self.data_list], dtype=np.int32)
        for data in self.data_list:
            data.config.order_range = self.order_range # inject the order range into each config for internal awareness

        if len(np.unique(self.orders)) != len(self.orders):
            raise ValueError("All Data instances within the inputted list must have unique order numbers.")

    def run_ACID(
        self,
        orders            : Array1D|int|str|None = None,
        use_index_mapping : bool                 = True,
        worker            : IntLike|None         = None,
        nworkers          : IntLike|None         = None,
        overwrite         : bool|None            = None,
        overwrite_kwargs  : bool                 = False,
        pack              : bool                 = False,
        **kwargs,
        ) -> None:
        """
        Runs ACID on the Data instances in the data list for the specified orders. The results are saved in the save_dir if it is not None, 
        with one pickle file per order containing the Result object. The idea is that you can run ACID on any orders you choose

        Parameters
        ----------
        orders : :py:type:`Array1D` | int | str | None, optional
            The orders to run ACID on. This can be provided as a single integer for one order, a list of integers for multiple specific orders, 
            the string "all" to run on all orders, or None to run on all orders. Default is None, which will run all orders.
        use_index_mapping : bool, optional
            If False, will not use the order to index mapping, instead orders are indexed directly. Default is True. Only applies for int or array inputs for orders.
        worker : :py:type:`IntLike` | None, optional
            Used in conjunction with nworkers. If an integer is provided, it specifies the worker number for this process. 
            When both worker and nworkers are provided, all the orders specified in "orders" will be split evenly across the nworkers. 
            For example, if there are 100 orders, and nworkers is 4, then worker 0 will run orders 0-24, worker 1 will run orders 25-49, etc. 
            The workers are 0-indexed. Default is None, which means no splitting and all specified orders will be run in this process.
        nworkers : :py:type:`IntLike` | None, optional
            The total number of workers to use to split the orders. See the "worker" parameter for more details. Default is None.
        store_sampler : bool, optional
            If True, the sampler object from the ACID run will be stored in the same folder as the resulting data pickles.
            This will take up more disk space, but allow for use of the :py:class:`Result` methods requiring the sampler attribute.
            We recommend leaving this on True if using deterministic sampling, otherwise set to False. Default is True.
        size_limit : Scalar | None, optional
            A hard size limit to the sampler in GB.
            If the sampler exceeds this size, it will not be stored regardless of the store_sampler flag.
            This is to avoid accidentally storing very large samplers. If None, no limit is set. Default is 1GB.
            A warning will be printed if this size_limit forces the store_sampler to be False if store_sampler was set to True.
        overwrite : bool, optional
            If True, will allow overwriting existing data and sampler pickles in the save_dir. Default is None, which will use the class
            default behaviour set in initialization (which is False). If False, this will skip running ACID on orders 
            that already have result pickles in the save_dir.
        overwrite_kwargs : bool, optional
            If True, any keys in the kwargs that are also in the config for the Data instance will be overwritten by the kwargs values.
            Use with caution, by default False.
        pack : bool, optional
            If True, a copy of all the DataList instances are packed into a single pickle for faster loading. Only applies if this task is not being split
            over multiple workers, otherwise, reverts back to False. By default, False.
        **kwargs :
            Additional keyword arguments to be passed to the ACID method for each order. These will not overwrite any existing keys unless
            overwrite_kwargs is set to True, in which case they will overwrite existing keys in the config for the Data instance for that order.
            The kwargs passed also allow you to add/overwrite the linelist and velocities in the Data instance with the same overwrite logic.
        """
        from .acid import Acid # local import to avoid circular imports, since Acid imports Data

        # Configure overwrite from class default if not input in the method call
        if overwrite is None:
            overwrite = self.overwrite

        # Validate worker and nworkers inputs for splitting orders across workers, and set defaults if not provided for easier logic below.
        if worker is not None or nworkers is not None:
            if worker is None or nworkers is None:
                raise ValueError("Both worker and nworkers must be provided together to use the worker splitting functionality.")
            if worker < 0 or worker >= nworkers:
                raise ValueError(f"worker must be between 0 and nworkers-1. Got: worker={worker}, nworkers={nworkers}")
        else:
            nworkers = 1 # if no worker splitting, just set nworkers to 1 for the logic below to work

        # Handle formats for orders input
        if isinstance(orders, int):
            orders = orders if use_index_mapping else self.i2o[orders]
            orders = np.array([orders], dtype=np.int32)
        elif isinstance(orders, str):
            if orders.lower() == "all":
                orders = self.orders
            else:
                raise ValueError(f"If orders is a string, it must be 'all' to run ACID on all orders. Got: {orders!r}")
        elif orders is None:
            orders = self.orders
        elif isinstance(orders, (list, np.ndarray)):
            if not all(isinstance(o, (int, np.integer)) for o in orders):
                raise ValueError(f"If orders is a list, all elements must be integers. Got: {orders!r}")
            if use_index_mapping:
                if not all(o in self.orders for o in orders):
                    raise ValueError(f"All orders in the input list must be in the DataList. Got: {orders!r}, but available orders are: {self.orders!r}")
            else:
                if not all(o in self.i2o for o in orders):
                    raise ValueError(f"All orders in the input list must be in the DataList. Got: {orders!r}, but available orders are: {self.orders!r}")
                orders = [self.i2o[o] for o in orders] # converts from order to index if use_index_mapping is False, otherwise assumes orders are indexed directly
            orders = np.array(orders, dtype=np.int32)
        else:
            raise ValueError(f"orders must be an int, a list of ints, 'all', or None. Got: {orders!r}")

        # Now we split the orders across workers and select the orders for this worker
        if nworkers > 1:
            orders = np.array_split(orders, nworkers)[worker]

        # Check if linelist or velocities were in kwargs and remove them now once if so
        ll = kwargs.pop("linelist", None)
        vel = kwargs.pop("velocities", None)

        iterable = tqdm(orders, "Running ACID on orders", unit="order") if self.verbose >= 2 else orders
        for order in iterable:

            data = self.data_list[self.o2i[order]]

            # Check if ACID already ran for this order
            if os.path.exists(data.config.save_path) and overwrite is False:
                if data.complete:
                    if self.verbose >= 2:
                        print(f"An ACID completed result for order {order} already exists. \n"
                                f"Skipping this order. To overwrite existing results, set overwrite=True.")
                    continue
                elif data.exception is not None:
                    if self.verbose >= 2:
                        print(f"An ACID run for order {order} previously encountered an exception. \n"
                                f"Skipping this order. To retry and overwrite existing results, set overwrite=True.")
                    continue

            # Handling if any kwargs were input
            # Only overwrite if overwrite_kwargs is True, otherwise keep the existing linelist/velocities in the Data instance
            if ll is not None:
                data.linelist = ll if overwrite_kwargs else data.linelist
            if vel is not None:
                data.velocities = vel if overwrite_kwargs else data.velocities
            if overwrite_kwargs:
                data.config.update_hipri(**kwargs)
            else:
                data.config.update_lowpri(**kwargs)

            # The following try-except loops came from just testing ACID on a lot of different orders, stars, and instruments
            failed_msg = f"Order {order} (list index {self.o2i[order]}) failed with error:"
            exception_raised = False
            try:
                _result = Acid(data=data).ACID() # All params are stored in Data and Config (in Data)
            except LineListRangeError:
                print(f"{failed_msg} line list range error. Your linelist is likely out of "\
                      f"range of the wavelengths. Skipping this order.", flush=True)
                exception_raised = True
            except ContinuumError:
                print(f"{failed_msg} continuum fitting error. The fitted continuum likely "\
                      f"had negative values. Skipping this order.", flush=True)
                exception_raised = True
            except SNCutError:
                print(f"{failed_msg} S/N cut error. The S/N of the spectrum is likely too "\
                      f"low, and no lines survived the cut. Skipping this order.", flush=True)
                exception_raised = True
            # If no known exception arose, just print the last 3 calls in traceback for debugging and skip the order.
            except Exception as e:
                print(f"{failed_msg} unknown error, see traceback. Skipping this order. Traceback:\n", flush=True)
                tb.print_exc(limit=-3)
                exception_raised = True
                data.exception = str(e)
            
            if exception_raised:
                try:
                    data.traceback = traceback.format_stack() # include the new exception in the data instance for future reference
                    data.save() # save the data instance with the exception for future reference
                except:
                    print(f"Failed to save the Data instance for order {order} after an exception was raised.\n" \
                          f"This is likely due to a corrupted Data instance.", flush=True)

        # Once all the orders have been done, we can repack the all the data instances (if asked) into one to speedup loading time
        # The data instances are very light as they do not store the sampler, so we can usually afford to pack and store duplicates
        if pack and np.array_equal(orders, self.orders):
            self.save()

    def save(self, save_dir:str|None=None) -> None:
        """
        Packs all of the DataList instances into a single DataList pickle, and save the state of the datalist to this pickle.
        Otherwise, the data instances can always be reloaded separately from the inidividual resulting pickle files in the directory for their order.
        Or just wherever the Config has them stored.
        The pickle file contains a dictionary with the list of Data objects (converted to dictionaries) and the save_dir.
        The filename is always "datalist.pkl", as save_dir must be a directory.
        If save_dir is not provided, self.save_dir is used. If that is also None, a ValueError is raised.
        All the orders should be in the memory to run this function, you can ensure they are all loaded with the load() method 
        and pointing to the directory with all the data pickles.

        Parameters
        ----------
        save_dir : str | None, optional
            The directory to save the DataList pickle file. If None, self.save_dir is used. Default is None.
        """
        if save_dir is not None:
            self.save_dir = save_dir
        if self.save_dir is None:
            raise ValueError("No save directory provided and save_dir was not set.")
        save_loc = os.path.join(self.save_dir, "datalist.pkl")
        d = {
            "verbose": self.verbose,
            "data_list": [data.to_dict() for data in self.data_list],
        }
        with open(save_loc, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path:str, verbose:int|str|bool|None=None) -> DataList:
        """
        Loads a DataList from a pickle file. The pickle file should contain a dictionary with the list of Data objects (converted to dictionaries) and the save_dir.
        Will attempt to load from datalist.pkl in the provided path if it is a directory, otherwise will attempt to load from the provided path directly. 
        If neither of those work, it will attempt to load from result pickles in a results directory within the provided path.

        Parameters
        ----------
        path : str
            The directory containing the datalist.pkl file, or the datalist.pkl itself. Note that the directories containing the results should also be in here.
        verbose : int | str | bool | None, optional
            The verbosity level to use when loading the DataList. If None, the verbosity level from the pickle file is used. Default is None. The verbosity only
            affects this function and will not overwrite the verbosity level of the DataList once it is loaded.

        Returns
        -------
        DataList
            The loaded DataList object.
        """
        abspath = os.path.abspath
        join = os.path.join
        isdir = os.path.isdir
        exists = os.path.exists

        path = abspath(path)
        if path.endswith("datalist.pkl"):
            if not exists(path):
                raise ValueError(f"No pickle file found at {path} to load the DataList from.")
            else:
                path = os.path.dirname(path)
        elif not isdir(path):
            raise ValueError(f"The provided path {path} is not a directory, or a datalist pickle file.\n"
                             f"You should provide a path to a directory containing the folders with the data pickles and sampler files.")

        d = {}
        if exists(join(path, "datalist.pkl")):
            with open(join(path, "datalist.pkl"), "rb") as f:
                d = pickle.load(f)

        if verbose is None:
            verbose = d.get("verbose", None)
        verbose = Config(verbose=verbose).verbose

        # If the datalist was repacked, load directly from there
        if "data_list" in d:
            data_list = [Data().from_dict(payload) for payload in d["data_list"]]

            folder_moved_flag = False
            for data in data_list:
                folder_moved_flag |= cls._set_paths_for_data(data, path)

            datalist = cls.from_datalist(data_list, save_dir=path, verbose=verbose)
            datalist.save() # repack with new save locations

            if folder_moved_flag and verbose >= 1:
                print("Warning: At least one Data instance did not match the current location and has been updated.")

            return datalist

        dir_list = os.listdir(path)        
        data_list = []
        folder_moved_flag = False
        dir_list = dir_list if verbose < 2 else tqdm(dir_list, "Loading Data instances from directory", unit="folder")
        for folder in dir_list:
            if isdir(join(path, folder)) and folder.startswith("order_"):
                save_path = abspath(join(path, folder, "data.pkl"))
                sampler_path = abspath(join(path, folder, "sampler.h5"))
                if exists(save_path):
                    data = Data.load(save_path)
                    folder_moved_flag |= cls._set_paths_for_data(data, path)
                    data_list.append(data)

        if folder_moved_flag and verbose is not None and verbose >= 1:
            print(f"Warning: At least one of the Data instances found in the directory does not match the current location, it has been updated.")

        obj = cls.from_datalist(data_list, save_dir=path, verbose=verbose)
        return obj

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, dir):
        if dir is not None:
            os.makedirs(dir, exist_ok=True)
        elif self._save_dir is None:
            if self.verbose >= 1:
                print("Warning: save_dir is set to None. No results will be saved. This is not recommended.")
        self._save_dir = dir
        return

    @property
    def combined_profile(self) -> tuple|None:
        """
        Returns the combined profile and its errors. If the combined profile has not been calculated yet, 
        it will attempt to combine the profiles without any exclusions.

        Returns:
            tuple[Array1D, Array1D]|None: The combined profile and its errors, or None if not available.
        """
        if self._combined_profile is None:
            try:
                self.combine_profiles()
            except Exception as e:
                raise ValueError(f"An attempt was made to combine profiles, as they did not already exist, but there was an exception:\n{e}")
        return self._combined_profile

    @property
    def data_list(self):
        return self._data_list

    @property
    def results(self):
        """
        Returns a list of Result objects for each Data instance in the DataList.
        If a Data instance does not have a result, None is returned for that order.
        This property is useful for not reaccessing ther result each time a plot is made.
        
        Returns:
            list[Result|None]: A list of Result objects or None for each order in the DataList.
        """
        if self._results is None:
            if self.verbose >= 1:
                print("Accessing results, the output below comes from initialising the Result object" \
                " and will only be shown once for this DataList instance.")
            self._results = [data.result for data in self.data_list]

        return self._results

    @data_list.setter
    def data_list(self, data_list):
        """
        Sets the data list and ensures that it is a list of Data instances. 
        Also sorts the list by order and updates the order to index mapping.
        """
        if not isinstance(data_list, list):
            raise ValueError("data_list must be a list of Data instances.")
        if not all(isinstance(data, Data) for data in data_list):
            raise ValueError("All elements in data_list must be instances of the Data class.")
        self._data_list = data_list
        self.sort_by_order() # ensures that the list is sorted and the order to index mapping is updated when setting a new list

    def combine_profiles(self, exclude:int|Array1D|None=None, must_have_converged:bool=False, od:bool=True) -> None:
        """
        Calculates the combined profile and its errors across all orders, excluding any orders specified in the exclude argument.
        
        Parameters
        ----------
        exclude : int | list[int] | None, optional
            Orders to exclude from the combined profile calculation.
        must_have_converged : bool, optional
            If True, only includes orders that have converged in the combined profile calculation. Default is False, which 
            includes all orders regardless of convergence status.
        od : bool, optional
            If True, the combination is done in optical depth. The returned profile is always in flux. By default, True.
        """
        if isinstance(exclude, int):
            exclude = [exclude]
        elif exclude is None:
            exclude = []

        if not all(o in self.orders for o in exclude):
            raise ValueError(f"All orders in the exclude list must be in the DataList. \nGot: {exclude!r}, but available orders are: {self.orders!r}")

        profiles = []
        errors = []
        covariances = []
        for data in self.data_list:
            if data.config.order in exclude:
                continue
            try:
                if (must_have_converged and not data.result.converged):
                    continue
            except:
                continue
            if "final" not in data.profile:
                continue

            # We actually want to combine in optical depth - more stable
            p, e, c = utils.flux_to_od(data.profile["final"][0], data.profile["final"][1], cov_matrix=data.profile["final"][2], od=od)

            profiles.append(p)
            errors.append(e)
            covariances.append(c)

        self._combined_profile = utils.combine_profiles(profiles, errors, covariances)

        self._combined_profile = utils.od_to_flux(self._combined_profile[0], self._combined_profile[1], cov_matrix=self._combined_profile[2], od=od)

        self.excluded_orders = exclude
        return

    def plot_combined_profile(self, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots the combined profile across all orders

        Parameters
        ----------
        return_fig : bool
            If True, returns the figure and axis objects instead of displaying the plot.
        
        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.

        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        for data in self.data_list:
            if "final" not in data.profile or data.config.order in self.excluded_orders:
                continue # failed or excluded orders
            ax.errorbar(self.velocities, data.profile["final"][0], alpha=0.2, color="C0",
                        label=f"All profiles" if data.config.order == self.orders[0] else None)

        ax.errorbar(self.velocities, self.combined_profile[0], self.combined_profile[1], color="black", fmt=".-", ecolor="red", label="Combined profile")
        ax.legend()
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Relative Flux")
        ax.set_title("Combined ACID profiles")
        ax.grid(True)
        if return_fig:
            return fig, ax
        plt.show()

    def fit_profile(self, **kwargs) -> None|tuple:
        """
        Fits the combined profile across all orders.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to the :py:function:`Profiles.plot_fit` method.
        
        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects from the profile fit plot if return_fig is True, otherwise None.
        """
        from .profiles import Profiles
        profiles = Profiles(self.velocities, *self.combined_profile)
        return profiles.plot_fit(**kwargs)

    def plot_all_profiles(self, od:bool=False, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots all the profiles for each order in the DataList. The combined profile is also shown.

        Parameters
        ----------
        od : bool, optional
            If True, shows the profiles in optical depth.
        return_fig : bool, optional
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        norm = mpl.colors.Normalize(vmin=self.order_range[0], vmax=self.order_range[-1])
        cmap = mpl.colormaps.get_cmap("viridis")#, len(self.order_range))

        peak_vel_idx = np.nanargmin(self.combined_profile[0])
        min_prof = 1
        for data in self.data_list:
            order = data.config.order
            color = cmap(norm(order))
            if "final" not in data.profile:
                continue # failed orders
            ax.plot(self.velocities, utils.flux_to_od(data.profile["final"][0], od=od), alpha=0.2, color=color)

            if data.profile["final"][0][peak_vel_idx] < min_prof:
                min_prof = data.profile["final"][0][peak_vel_idx]
        
        ax.errorbar(self.velocities, # self.combined_profile[0], self.combined_profile[1],
                    *utils.flux_to_od(self.combined_profile[0], self.combined_profile[1], od=od),
                    color="black", fmt=".-", ecolor="red", label="Combined profile", zorder=10)

        ax.axhline(1, color="black", linestyle="--", alpha=0.5)

        # Show colour map
        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation="vertical", label="Order")

        # Set sensible limits
        if not od:
            ax.set_ylim(max(min_prof-0.1, 0), 1.2)
        # if od, limits are usually sensible due to log scaling
        ax.set_xlabel("Velocity (km/s)")
        ax.set_ylabel("Relative Flux")
        ax.set_title("All ACID profiles")
        ax.grid(True)
        ax.legend()
        if return_fig:
            return fig, ax
        plt.show()

    def plot_mean_profile_errors(self, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots the errors of all the profiles for each order in the DataList.

        Parameters
        ----------
        return_fig : bool
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        errors = []
        for data in self.data_list:
            if "final" not in data.profile:
                errors.append(np.nan)
                continue
            errors.append(np.mean(data.profile["final"][1]))

        ax.plot(self.orders, errors, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Order")
        ax.set_ylabel("Mean Profile Error")
        ax.set_title("Mean Profile Errors for each order")
        ax.set_yscale("log")
        ax.grid(True)

        if return_fig:
            return fig, ax
        plt.show()

    def plot_chi2(self, return_fig:bool=False) -> None|tuple[plt.Figure, plt.Axes]:
        """
        Plots the chi-squared values against order in the DataList.
        This helps diagnose which orders have a bad fit and may need to be excluded from the combined profile.

        Parameters
        ----------
        return_fig : bool
            If True, returns the figure and axis objects instead of displaying the plot.

        Returns
        -------
        tuple[plt.Figure, plt.Axes] | None
            The figure and axis objects if return_fig is True, otherwise None.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        orders = []
        chi2_values = []
        for data in self.data_list:
            if "final" in data.profile:
                orders.append(data.config.order)
                try:
                    flux = np.asarray(data.flux["final"])
                    model = np.asarray(data.forward_y["final"])
                    err = np.asarray(data.errors["final"])
                    chi2 = np.sum(((flux-model)/err) ** 2)
                    chi2_values.append(chi2)
                except Exception as e:
                    print(f"Warning: Could not calculate chi-squared for order {data.config.order}. :\n{e}")
                    chi2_values.append(np.nan)
        
        ax.plot(orders, chi2_values, marker='o', linestyle='-', color='blue')
        ax.set_xlabel("Order")
        ax.set_ylabel("Chi-squared")
        ax.set_yscale("log")
        ax.set_title("Chi-squared values for each order")
        ax.grid(True)
        
        if return_fig:
            return fig, ax
        plt.show()

    @staticmethod
    def _set_paths_for_data(data: Data, save_dir: str) -> bool:
        """Update and save a Data instance only if its directory paths changed."""
        order = data.config.order

        save_path = os.path.abspath(
            os.path.join(save_dir, f"order_{order}", "data.pkl")
        )
        sampler_path = os.path.abspath(
            os.path.join(save_dir, f"order_{order}", "sampler.h5")
        )

        stored_save_path = data.config.save_path
        stored_sampler_path = data.config.sampler_path
        # TODO: on kelvin this was printing and moving unexpectedly, print the locations and find out why
        changed = (
            stored_save_path is None
            or os.path.abspath(stored_save_path) != save_path
            or stored_sampler_path is None
            or os.path.abspath(stored_sampler_path) != sampler_path
        )

        if changed:
            data.config.save_path = save_path
            data.config.sampler_path = sampler_path

            if os.path.exists(sampler_path):
                data.sampler = sampler_path

            data.save()

        return changed