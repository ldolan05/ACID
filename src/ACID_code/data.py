from __future__ import annotations
from dataclasses import dataclass, field, fields
from beartype import beartype
from tqdm import tqdm
import traceback as tb
from typing import Any, Dict, Optional
from emcee import EnsembleSampler
import emcee.backends.backend as emceebackend
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle, os
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
                    lines = np.array(line_input)
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

            assert len(lines) == len(widths), f"lines and widths should be of same length, got: {len(lines)}, {len(widths)}"
            final_dict[name] = {"lines": np.array(lines), "widths": np.array(widths)}
        return final_dict

#TODO: Tests for config updates, attribute access and error handling
@beartype
class Config:
    """The main class for storing ACID configuration settings, with methods to plot and save/load the configuration state."""

    #: The default configuration settings for ACID, used if not set by the user. See :py:class:`Acid` for more details on how these are used in ACID.
    defaults = {
        # INIT CONFIGURATION
        "verbose" : 2,
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
                "default_width" : 1000,
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

        # RUN_ACID CONFIGURATION
        "deterministic_profile" : True,
        "poly_ord" : 3,
        "continuum_percentile" : 90,
        "bin_size" : 100,
        "pix_chunk" : 20,
        "dev_perc" : 25,
        "n_sig" : 3,
        "skips" : 1,
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
    }

    #: Property list for error handling
    properties = ["verbose", "masking_lines"]

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

    def __repr__(self) -> str:
        full_dict = self.to_dict()
        return f"Config({full_dict})"

    def __str__(self) -> str:
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
            if k not in self.defaults:
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
            value = self.defaults["verbose"]
        elif value is False:
            value = 0
        elif isinstance(value, (int, np.integer)):
            if value < 0 or value > 3:
                raise ValueError("verbose must be an integer between 0 and 3")
        elif isinstance(value, str):
            value = value.lower()
            if value in ["none", "no", "false", "off", "n", "0"]:
                value = 0
            elif value in ["low", "lo", "l", "1"]:
                value = 1
            elif value in ["medium", "med", "m", "2"]:
                value = 2
            elif value in ["high", "hi", "h", "3"]:
                value = 3
            else:
                raise ValueError("verbose string not recognised, must be one of 'none', 'low', 'medium', 'high' or their common variants")
        else:
            raise ValueError("verbose must be an integer between 0 and 3, a boolean, or a string indicating the verbosity level")

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
        plt.show()

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

    # Cached products that are expensive or useful for resuming
    # ---------------------------------------------------------
    #: The alpha vector used in the linear model, used for solving the linear system in MCMC
    alpha                  : Optional[np.ndarray] = None  
    #: Tuple generated by np.cho_factor, used for solving the linear system in MCMC
    c_factor               : Optional[tuple]      = None
    #: Boolean 1D mask on "combined" grid, used in final process_results step
    residual_masks         : Optional[np.ndarray] = None
    #: Boolean 1D mask on "combined" grid, used to mask out NaN values in combined spectra
    nanmask                : Optional[np.ndarray] = None
    #: Initial profile generated in residual masking
    initial_profile        : Optional[np.ndarray] = None
    #: Corresponding errors for the initial profile
    initial_profile_errors : Optional[np.ndarray] = None
    #: Polynomial inputs for just the continuum model
    poly_inputs            : Optional[np.ndarray] = None
    #: The initial_model_inputs if needed for debugging, only set after model_inputs is modified in residual masking
    initial_model_inputs   : Optional[np.ndarray] = None
    #: The concatenated array of initial profile and poly coefficients, used as input to emcee
    model_inputs           : Optional[np.ndarray] = None
    #: The initial state of the MCMC walkers, used for resuming and debugging
    initial_state          : Optional[np.ndarray] = None

    # Small cached products needed for MCMC if doing reruns
    # -----------------------------------------------------
    #: The number of walkers and dimensions for the MCMC sampler, used for reshaping the samples if resuming
    nwalkers : Optional[int]        = None
    #: The number of dimensions for the MCMC sampler, used for reshaping the samples if resuming
    ndim     : Optional[int]        = None

    # Data required/calculated in results/after MCMC sampling
    # -------------------------------------------------------
    #: The list to store all frames of the MCMC sampling, has dimensions (nframes, 3, nvel), where the 3 indexes are the profile, error, and covariance matrix
    profiles          : Optional[list] = None
    #: The list to store the combined frame of the MCMC sampling, has dimensions (3, nvel)
    combined_profile  : Optional[list] = None
    #: The final fitted continuum model and errors
    continuum_model   : Optional[np.ndarray] = None
    #: The number of steps taken in the MCMC sampling, used for checking convergence and for resuming
    nsteps            : Optional[int]  = 0

    #: The samples are stored as an array of shape (nwalkers, nsteps, ndim), and not as an emcee sampler object to save memory.
    #: It has already had the standard burn-in and thinning applied. If the user wants to use their own thinning and burn-in,
    #: they must store the sampler with the result object by configuring the output.
    # TODO: samples not currently used in result, check again why I did the below
    sampler  : Optional[EnsembleSampler] = None # stored ensemble sampler
    #: A flag for whether the profiles have been fully calculated to avoid recalculating
    complete : bool                      = False # is set to True when the profiles and combined_profile have been fully calculated

    # Other useful data and figures
    # -----------------------------
    initialisation_time : Optional[float] = 0  # time taken for initialization
    mcmc_time           : Optional[float] = 0  # time taken for MCMC sampling
    get_profiles_time   : Optional[float] = 0  # time taken to get profiles
    full_run_time       : Optional[float] = 0  # total time for the full run
    plotting_variables  : Dict[str, Any]  = field(default_factory=dict)

    # Initialise the properties
    # -------------------------
    #: Config data for convenience as a class, but converted to a dictionary on save to avoid pickling issues
    _config   : Config = field(default_factory=Config)
    #: The linelist is stored as a dictionary but exposed as a :py:class:`LineList` object when the property is accessed.
    _linelist : Optional[Dict[str, np.ndarray]] = None
    #: The velocities are stored as a 1D numpy array
    _velocities : Optional[np.ndarray] = None

    @property
    def velocities(self):
        """The velocity grid to perform LSD on."""
        return self._velocities

    @velocities.setter
    def velocities(self, value:Array1D|None) -> None:
        """Sets and overwrites the velocity grid if the value is not None.
        If overwriting, resets the Data instance as calculations are dependent on velocities."""
        if value is not None:
            overwriting = self._velocities is not None # boolean flag for whether we are overwriting the existing velocities
            
            velocities = np.array(value)
            if not np.all(np.isfinite(velocities)):
                raise ValueError("The velocity grid you are trying to set must all be finite and not contain NaNs")
            self._velocities = np.array(value)
            
            if overwriting:
                print("Warning: Overwriting existing velocities in Data. The Data instance will be reset to clear calculations that depend on the velocities.\n" \
                "The linelist, config, and original data inputs will not be reset.")
                self.reset()

    def reset(self) -> None:
        """Resets all data attributes to their default empty states, except for the array inputs, linelist, velocities, and config."""
        self.alpha = None
        self.c_factor = None
        self.residual_masks = None
        self.nanmask = None
        self.initial_profile = None
        self.initial_profile_errors = None
        self.poly_inputs = None
        self.initial_model_inputs = None
        self.model_inputs = None
        self.nwalkers = None
        self.ndim = None
        self.profiles = None
        self.combined_profile = None
        self.continuum_model = None
        self.nsteps = 0
        self.sampler = None
        self.complete = False
        self.plotting_variables = {}
        if "input" in self.wavelengths and self.wavelengths["input"] is not None:
            self.wavelengths = {"input": self.wavelengths["input"]}
            self.flux = {"input": self.flux["input"]}
            self.errors = {"input": self.errors["input"]}
            self.sn = {"input": self.sn["input"]}
        else:
            self.wavelengths = {}
            self.flux = {}
            self.errors = {}
            self.sn = {}

    def plot_continuum_fit(self, plot_type:str="initial", return_fig:bool=False, save_fig:str|None=None) -> None:
        """
        Plots the result of the continuum fitting step, showing the original spectrum, the fitted continuum, and the clipped points used for the continuum fit.

        Parameters
        ----------
        plot_type : str, optional
            The type of continuum fit to plot, either "initial" for the initial continuum fit or
            "masked" for the continuum fit after residual masking. Default is "initial".
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False.
        save_fig : str or None, optional
            If provided, the path to save the figure. If None, the figure will not be saved. Default is None.
        """
        # Check we have all inputs needed for plot
        if plot_type not in ["initial", "masked"]:
            raise ValueError("plot_type must be either 'initial' or 'masked'")
        if plot_type not in self.plotting_variables:
            raise ValueError(f"No plotting variables found for plot_type={plot_type!r}. " \
                             "Please ensure that the continuum fit has been performed for this plot_type.")
        if not all(
            attr in self.plotting_variables[plot_type] for attr in [
                "unnormalized_wavelengths", "fluxes", "fit", "clipped_waves", "clipped_flux", "good"]
            ):
            raise ValueError("To plot the continuum fit, the following attributes must be set: unnormalized_wavelengths, fluxes, fit, clipped_waves, clipped_flux, good")

        # Unpack variables
        unnormalized_wavelengths = self.plotting_variables[plot_type]["unnormalized_wavelengths"]
        fluxes                   = self.plotting_variables[plot_type]["fluxes"]
        good                     = self.plotting_variables[plot_type]["good"]
        fit                      = self.plotting_variables[plot_type]["fit"]
        clipped_waves            = self.plotting_variables[plot_type]["clipped_waves"]
        clipped_flux             = self.plotting_variables[plot_type]["clipped_flux"]

        # Normalise wavelengths and plot flux and fit
        a, b = utils.get_normalisation_coeffs(unnormalized_wavelengths)
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(unnormalized_wavelengths, fluxes, label='Original Spectrum', color="C0", alpha=0.7)
        ax.plot(unnormalized_wavelengths, fit, label='Fitted Continuum', color='red')
        ax.plot((clipped_waves[good]-b)/a, clipped_flux[good], 'o', label='Continuum Normalized Spectrum', color='green')

        # Plot bad regions (chunk deviation masking):
        masked = ~good
        padded = np.concatenate(([False], masked, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i, (start, end) in enumerate(zip(starts, ends)):
            ax.axvspan((clipped_waves[start]-b)/a, (clipped_waves[end-1]-b)/a,
                        color='red', alpha=0.15, label="Bad regions" if i == 0 else None)

        # Plot the linelist points, with a color corresponding to their depth in the linelist within the range
        # Only plot the 20 strongest lines to avoid overcrowding.
        ll_wl = self.linelist["wavelengths"]
        ll_depths = self.linelist["depths"]
        ll_wl, ll_depths = utils.clip_wavelengths(unnormalized_wavelengths, ll_wl, ll_depths)
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
            if self.config.verbose > 0:
                print("There was an error plotting the linelist points, most likely your linelist range is outside your wavelength range.")
            pass

        # Plot the line masks with their names
        x = unnormalized_wavelengths
        line_mask = self.config.masking_lines.get_masks(x, with_names=True)
        for i, (name, masks) in enumerate(line_mask.items()):
            padded = np.concatenate(([False], masks, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for j, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]), color=f'C{i+1}', alpha=0.3,
                           label=f"{name} Line masks" if j == 0 else None)

        # Add labels and legend, and save or show figure
        plot_title = "Initial Continuum Fit" if plot_type == "initial" else "Continuum Fit after Residual Masking"
        ax.set_title(plot_title)
        ax.legend()
        if save_fig is not None:
            plt.savefig(save_fig)
        if return_fig:
            return fig, ax
        plt.show()

    def plot_residual_masking(self, save_fig:str|None=None) -> None:
        """
        Creates 3 plots to show the result of the residual masking step, showing the residuals with the sigma clipping thresholds, 
        the masked regions, and the initial profile after masking.

        Parameters
        ----------
        save_fig : str or None, optional
            If provided, a directory to save the figure, will create one if it does not exist. 
            If None, the figure will not be saved. Default is None.
        """
        # Check we have all inputs needed for plot
        if "residual_masking" not in self.plotting_variables:
            raise ValueError("No plotting variables found for residual_masking. ")
        if not all(
            attr in self.plotting_variables["residual_masking"] for attr in [
                "mask", "residuals", "upper_clip", "lower_clip", "pix_mask", "profile_F"]
        ):
            raise ValueError("Not all required plotting variables found for residual_masking. ")
        if "masked" not in self.wavelengths and "masked" not in self.flux:
            raise ValueError("No masked wavelengths or fluxes found. Please ensure that the residual masking step has been performed")
        if save_fig is not None:
            if not os.path.isdir(save_fig):
                raise ValueError(f"save_fig must be a valid path to a directory to save the figures, or None to show the figures. Got: {save_fig}")

        # Unpack variables
        x = self.wavelengths["masked"]
        y = self.flux["masked"]
        mask = self.plotting_variables["residual_masking"]["mask"]
        residuals = self.plotting_variables["residual_masking"]["residuals"]
        upper_clip = self.plotting_variables["residual_masking"]["upper_clip"]
        lower_clip = self.plotting_variables["residual_masking"]["lower_clip"]
        pix_mask = self.plotting_variables["residual_masking"]["pix_mask"]
        profile_F = self.plotting_variables["residual_masking"]["profile_F"]

        nremoved = np.sum(mask)
        if self.config.verbose > 1:
            print(f"Residual masking has removed {nremoved}/{len(residuals)} points.")

        # Create plot and add residuals with sigma clipping thresholds and masked regions
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(x, residuals, label='Residuals', color='blue')
        ax.axhline(upper_clip, color='red', linestyle='--', label='Upper Clip Threshold')
        ax.axhline(lower_clip, color='green', linestyle='--', label='Lower Clip Threshold')
        ax.axhspan(lower_clip, upper_clip, color='gray', alpha=0.3, label='Sigma Clipping masking range')

        # Show line masking regions
        line_mask = self.config.masking_lines.get_masks(x, with_names=True)
        for i, (name, masks) in enumerate(line_mask.items()):
            padded = np.concatenate(([False], masks, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
            for j, (start, end) in enumerate(zip(starts, ends)):
                ax.axvspan((x[start]), (x[end-1]), color=f'C{i+1}', alpha=0.3,
                           label=f"{name} Line masks" if j == 0 else None)

        # Show pix_chunk masked points:
        masked = pix_mask
        padded = np.concatenate(([False], masked, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i, (start, end) in enumerate(zip(starts, ends)):
            ax.axvspan((x[start]), (x[end-1]),
                        color='red', alpha=0.15, label="Chunk deviation masking" if i == 0 else None)
        # And show pix_chunk range
        dev = self.config.dev_perc / 100
        ax.axhspan(-dev, dev, color='green', alpha=0.1, label="Chunk deviation masking range")

        ax.set_xlim(np.min(x), np.max(x))
        ax.grid(True)
        ax.set_title('Residuals with Sigma Clipping Thresholds')
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Residuals')
        ax.legend(loc="lower right")
        if save_fig is not None:
            plt.savefig(f"{save_fig}/residuals.png")
        plt.show()

        # Plot the profile
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.velocities, profile_F, label='LSD Profile after Masking and before sampling', color='red')
        ax.set_title('LSD Profile after Residual Masking')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('LSD Profile')
        ax.axhline(1, color='black', linestyle='--')
        ax.legend()
        ax.grid(True)
        if save_fig is not None:
            plt.savefig(f"{save_fig}/initial_profile.png")
        plt.show()

        # Finally plot the forward model
        from .mcmc import MCMC
        forward_masked, _ = MCMC(self).deterministic_model(self.poly_inputs)
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax[0].plot(x, y, label='Original data', color='black', linewidth=1)
        ax[0].plot(x, forward_masked, label='Forward model with masked residuals', color='C0', linewidth=1)
        ax[0].set_title('Forward model with masked residuals')
        ax[0].set_xlabel('Wavelength')
        ax[0].set_ylabel('Flux')
        ax[1].plot(x, (y-forward_masked)/forward_masked, label='Residuals', color='blue')
        ax[1].axhline(upper_clip, color='red', linestyle='--', label='Upper Clip Threshold')
        ax[1].axhline(lower_clip, color='green', linestyle='--', label='Lower Clip Threshold')
        ax[1].axhspan(lower_clip, upper_clip, color='gray', alpha=0.3, label='Sigma Clipping masking range')
        ax[1].set_title('Residuals of forward model with masked residuals')
        ax[1].set_xlabel('Wavelength')
        ax[1].set_ylabel('Residuals')
        ax[0].legend()
        ax[0].grid(True)
        if save_fig is not None:
            plt.savefig(f"{save_fig}/forward_model.png")
        plt.show()

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
                if self.config.verbose > 0:
                    print(f"Warning: input wavelengths, flux, and errors are already set in the class. \n" \
                        f"Some of the inputs you provided are None. \n" \
                        f"If you are trying to update the input wavelengths, flux, or errors, you must provide all 3. \n"
                        f"The current input wavelengths, flux, and errors will be kept.")
                return
            elif not any_inputs_not_none:
                if self.config.verbose > 2:
                    print("Input wavelengths, flux, and errors are already set in the class. Keeping existing values.")
                return
            # Else continue with the rest of the function to update inputs, later on, the code will check if new inputs are 
            # different from the existing ones, if so, deletes variables that need to be recalculated.
        else:
            if not all_inputs_not_none:
                raise ValueError("input_wavelengths, input_flux, and (input_errors or input_sn) must be provided either as arguments " \
                                 "or in the form of a Data object.")

        # Convert to arrays, squeeze to remove extra dimensions (as default in legacy inputs)
        input_wavelengths = np.array(input_wavelengths).squeeze()
        input_flux = np.array(input_flux).squeeze()
        input_errors = np.array(input_errors).squeeze() if input_errors is not None else None
        input_sn = np.array(input_sn).squeeze() if input_sn is not None else None

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
        if input_sn is None and input_errors is not None:
            input_sn = utils.guess_SNR(input_wavelengths, input_flux, input_errors)
            if self.config.verbose > 1:
                print(f"No input_sn provided and was instead approximated. Guessed value(s):\n {input_sn}")
        if input_errors is None and input_sn is not None:
            input_errors = utils.guess_errors(input_wavelengths, input_flux, input_sn)
            if self.config.verbose > 0:
                print(f"No input_errors provided and was instead approximated from the input S/N.\n"\
                      f"It is highly recommended to obtain correct per-pixel errors.")

        # Check they have matching shape
        if not input_wavelengths.shape == input_flux.shape == input_errors.shape:
            raise ValueError("Input wavelengths, spectra and spectral errors must all have the same shape.")

        # Ensure now that the SN becomes just a single value per frame
        if input_sn.ndim == input_flux.ndim:
            # Per pixel S-N provided, take the mean over the central 2/3 of the wavelengths
            input_sn = utils.collapse_SNR(input_sn, input_wavelengths)
        elif input_sn.ndim != input_flux.ndim-1:
            raise ValueError("input_sn must be either a single-valued list/array with the average S/N for each frame, " \
            f"or an array of S/N values for each pixel. \n" \
            "The shape of the input input_sn does not match the number of frames in input_flux, " \
            "nor does it have one more dimension than input_flux.")
        assert input_sn.ndim == input_flux.ndim - 1, \
            f"input_sn.ndim and input_flux.ndim-1 do not match, sn ndim = {input_sn.ndim}, flux ndim = {input_flux.ndim}"

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
        input_wavelengths = input_wavelengths[:, ::skips]
        input_flux       = input_flux[:, ::skips]
        input_errors     = input_errors[:, ::skips]

        # In case these are set when input values already exist, check if they are the same, if not, reset variables to be recalculated.
        # This checks basically if self.wavelengths["input"] is the same as input_wavelengths, and same for flux and errors, if they exist. 
        overwriting = False
        for check in input_keys:
            if getattr(self, check).get("input", None) is not None and eval(f"input_{check}") is not None:
                if not np.allclose(getattr(self, check)["input"], eval(f"input_{check}"), equal_nan=True):
                    overwriting = True

        # Set inputs to class variables, the self.reset() cleans all arrays except for the inputs, so this is safe
        self.wavelengths["input"] = input_wavelengths
        self.flux["input"]        = input_flux
        self.errors["input"]      = input_errors
        self.sn["input"]          = input_sn

        # If reset is needed, reset calculated values to force recalculation with new inputs and warn the user
        if overwriting:
            if self.config.verbose > 0:
                print("Warning: input wavelengths, flux, or errors have been changed from their previous values. \n" \
                f"Resetting variables that need to be recalculated.\nThe velocity grid and linelist will not be reset.")
            self.reset()

    def save(self, filename:str="data.pkl", store_sampler:bool=True) -> None:
        """
        Saves the data object to a file using pickling. This will store just the dictionary of the class, 
        not the actual class itself. The load function then will initialise a new Data class using the dictionary.

        Parameters
        ----------
        filename : str
            The name of the file to save the data object to. This should be a .pkl file.
        store_sampler : bool
            Whether to store the MCMC sampler object in the data object. This is recommended for determinstic_profile=True as the sampler is
            quite small, otherwise, you should disable this and avoid using Result methods requiring the sampler.
        """
        payload = self.to_dict(store_sampler) # generates a dictionary of the data object for easy pickling

        with open(filename, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filename: str) -> Data:
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

    def to_dict(self, store_sampler:bool=True) -> dict[str, Any]:
        """
        Converts the data object to a dictionary payload for saving. This is used internally in the save method, 
        but can also be used for debugging or other purposes.

        Parameters
        ----------
        store_sampler : bool, optional
            Whether to include the MCMC sampler in the dictionary payload, by default True.
        """
        payload: dict[str, Any] = {}
        for f in fields(self):
            name = f.name
            val = getattr(self, name)

            if name == "_config":
                payload["config"] = val.to_dict() # store as dict in payload, but store as class in Data
            elif name == "sampler":
                if store_sampler and self.sampler is not None:
                    # Store the sampler as its backend dictionary to avoid pickling issues
                    payload["sampler"] = dict(self.sampler.backend.__dict__)
            else:
                payload[name] = val

        return payload

    def from_dict(self, payload: dict[str, Any]) -> Data:
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
            elif name == "sampler":
                sampler = payload.get("sampler", None)
                if sampler is None:
                    continue
                # Reconstruct sampler from backend if in payload, avoids pickling issues
                backend = emceebackend.Backend(dtype=np.float64)
                backend.__dict__.update(sampler)
                nwalkers, ndim = backend.shape
                from .mcmc import MCMC
                self.sampler = EnsembleSampler(nwalkers, ndim, log_prob_fn=MCMC(self), backend=backend) # dummy sampler to hold the backend
            else:
                if name in payload:
                    setattr(self, name, payload[name])
        return self

    # Store config as a property for handling it to/from dictionary on saving
    @property
    def config(self) -> Config:
        """Returns the internally stored config object, which contains the configuration of the ACID run."""
        return self._config

    @config.setter
    def config(self, value: Config) -> None:
        """Sets the internally stored config object."""
        self._config = value

    @property
    def result(self):
        if not self.complete:
            raise ValueError("Results have not yet been calculated, cannot return results object. Please run the MCMC sampling and process the results first.")
        from .result import Result
        return Result(self)

    @property
    def linelist(self) -> LineList|None:
        """Returns the internally stored linelist. It has keys "wavelengths" and "depths" or index 0 and 1."""
        return LineList(self._linelist)if self._linelist is not None else None

    @linelist.setter
    def linelist(self, value: Array2D|str|LineList|dict[str,Array1D]|None) -> None:
        self.set_linelist(value)
        return

    def set_linelist(self, linelist=None, linelist_wl=None, linelist_depths=None) -> None:
        """
        Sets the linelist for the data object. The linelist formats follows that of the doccumentation in the :py:class:`Acid` class,
        which then internally uses this function to set the linelist in the data object. The linelist is stored as a dictionary with 
        keys "wavelengths" and "depths", but is exposed as a :py:class:`LineList` object when accessed through the property. The LineList
        class allows for easy access to plotting, indexing, and validation.

        Parameters
        ----------
        See :py:class:`Acid` for the accepted linelist formats and parameters.
        """
        # TODO: depracate linelist_wl and depths
        # Check if linelist already exists, override with new inputs if provided
        if linelist is not None:
            overwriting = self._linelist is not None
            # The method names are self explaining, see the respective methods for more details
            linelist_wl, linelist_depths = LineList.validate_linelist(linelist, linelist_wl, linelist_depths)
            LineList.validate_dimensions(linelist_wl, linelist_depths) # ensures same shape and are 1D
            linelist_wl, linelist_depths = LineList.drop_invalid_lines(linelist_wl, linelist_depths)
            self._linelist = {"wavelengths": linelist_wl, "depths": linelist_depths}

            # If overwriting, reset variables
            if overwriting:
                if self.config.verbose > 0:
                    print("Warning: the input linelist has been modified. \n" \
                    f"Resetting variables that need to be recalculated.\nThe velocity grid and input arrays will not be reset.")
                self.reset()

    def plot_linelist(self, min_depth:Scalar=0.2, bounds:tuple|list|None=None, return_fig:bool=False) -> None|tuple:
        """
        Plots the linelist points with their corresponding depths as delta-function lines.

        Parameters
        ----------
        min_depth : :py:type:`Scalar`, optional
            The minimum depth for plotting the linelist points. By default 0.2.
        bounds : tuple or list, optional
            The wavelength bounds for clipping the linelist. If None, no clipping is applied.
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

        # Clip the linelist to the specified bounds if provided, and to the min_depth
        if bounds is not None:
            wl = wl[(wl >= bounds[0]) & (wl <= bounds[1])]
            depths = depths[(wl >= bounds[0]) & (wl <= bounds[1])]
        wl = wl[depths >= min_depth]
        depths = depths[depths >= min_depth]

        # Plot linelist
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.vlines(wl, 0, depths, color='C0', )
        ax.set_title('Line List')
        ax.set_xlabel('Wavelength (Angstroms)')
        ax.set_ylabel('Relative Line Depth')
        ax.legend()
        if return_fig:
            return fig, ax
        plt.show()

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
    def validate_linelist(linelist, linelist_wl, linelist_depths) -> tuple[np.ndarray, np.ndarray]:
        """
        Validates the linelist according to the description in :py:class:`Acid`, and returns the linelist wavelengths 
        and depths as numpy arrays. This is used internally in the set_linelist method.

        Parameters
        ----------
        linelist : str, dict, LineList, list, or np.ndarray
            See :py:class:`Acid`.
        linelist_wl : array-like, optional
            See :py:class:`Acid`.
        linelist_depths : array-like, optional
            See :py:class:`Acid`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The validated linelist wavelengths and depths as numpy arrays.
        """

        # Run through every possible input type and issue, I'm not going to comment everything but the logic is fairly
        # self-explanatory, and the error messages should be helpful for debugging if the input is not in the correct format.
        if (linelist_wl is None and linelist_depths is None) and linelist is None:
            raise ValueError("A linelist must be provided. For possible inputs, see https://acid-code.readthedocs.io/en/stable/_api/ACID_code.Acid.html")
        elif linelist is None and (linelist_wl is None or linelist_depths is None):
            raise ValueError("If 'linelist' is not provided, both 'linelist_wl' and 'linelist_depths' must be provided.")
        elif isinstance(linelist, str):
            # VALD linelist code, will add more linelist formats in the future or if requested
            full_linelist = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9), invalid_raise=False)
            linelist_wl = full_linelist[:,0]
            linelist_depths = full_linelist[:,1]
        elif isinstance(linelist, LineList):
            linelist_wl = linelist[0]
            linelist_depths = linelist[1]
        elif isinstance(linelist, dict):
            if "wavelengths" not in linelist or "depths" not in linelist:
                raise ValueError("If 'linelist' is a dict, it must contain keys 'wavelengths' and 'depths'")
            linelist_wl = linelist["wavelengths"]
            linelist_depths = linelist["depths"]
        elif isinstance(linelist, (list, np.ndarray)):
            if len(linelist) != 2:
                raise ValueError("If 'linelist' is a list or array, it must have length 2, with index 0 being wavelengths and index 1 being depths")
            linelist_wl = linelist[0]
            linelist_depths = linelist[1]
        elif linelist_wl is not None and linelist_depths is not None:
            pass # linelist_wl and linelist_depths already set, will be processed below
        else:
            raise ValueError(f"'linelist' must be a string path to a VALD linelist, a dictionary with keys 'wavelengths' and 'depths', \n" \
            "a LineList object, or a list/array indexed such that 0 is wavelengths and 1 is depths.")
        return np.array(linelist_wl), np.array(linelist_depths)

    @staticmethod
    def validate_dimensions(wavelengths, depths) -> None:
        """Validates that the wavelengths and depths are 1D arrays of the same shape. 
        This is used internally in the set_linelist method."""
        if wavelengths.ndim != 1 or depths.ndim != 1:
            raise ValueError("'wavelengths' and 'depths' must be a one-dimensional array or list")
        if wavelengths.shape != depths.shape:
            raise ValueError("'wavelengths' and 'depths' must have the same length and shape")

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
        mask &= (depths >= 0) & (depths <= 1)
        mask &= (wavelengths > 0)

        # Count the number of dropped lines for verbose output
        count_dropped = np.count_nonzero(~mask)
        if count_dropped == len(wavelengths):
            raise ValueError(f"All lines in the linelist are non-finite, nan, negative, or greater than 1.\n" \
            "Please check your linelist for invalid values.")
        if verbose > 0 and count_dropped > 0:
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
    and also to save intermediate results and figures per order. It also can save the whole sampler if you specify save_sampler=True in the run_ACID method,
    which will save the sampler with Data in result to the save_dir after running ACID for each order but take up much more disk space.
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
        verbose          : IntLike|bool|str|None          = None,
        load                                              = None,
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
            The directory to save intermediate results and figures for each order. If None, no saving will be done. Default is None.
        verbose : int | bool | str | None, optional
            The verbosity level for printing information during the initialization. 
            Follows the same format as the "verbose" input in the :py:class:`Config` class. 
            Default is None.
        load : Any, optional
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
        if load is not None:
            raise NotImplementedError(f"The 'load' argument is not yet implemented. \n"
                                      f"The idea is that you can input a Load object which has its own tools to pull s2d data from common "\
                                      f"instruments such as ESPRESSO, HARPS, etc. \nIf you want to use this feature, please open an issue or "\
                                      f"contribute a pull request with the implementation.")
            from .load import Load
            if not isinstance(load, Load):
                raise ValueError("load must be an instance of the Load class. Got: {load!r}")
            wavelengths, flux, errors, sn = load.get_data()
            order_range = load.order_range

        # Configure verbosity
        self.verbose = Config(verbose=verbose).verbose

        # Configure velocities
        self.velocities = velocities

        # Configure order_range, creates one if not input from the shape of wavelengths
        self.order_range = order_range # if None, will be set later, otherwise self.from_datalist handles the range from configs     

        # Configure save_dir, for saving intermediate results and figures per order
        self._save_dir = None
        self.save_dir = save_dir if save_dir is not None else None

        # Set empty class attributes
        self._combined_profile = None
        self.excluded_orders = []

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

            # Set the inputs for this Data instance, taking the idx'th element of the input arrays for this order. If errors or sn are not provided, they will be set to None and approximated in the Data class.
            input_errors = errors[idx] if errors is not None else None
            input_sn = sn[idx] if sn is not None else None
            data.set_inputs(wavelengths[idx], flux[idx], input_errors, input_sn)

            # Set linelist and velocities, which always should be same for all orders
            data.linelist = linelist # sets the linelist for this Data instance, which is shared across all orders in the DataList
            data.velocities = velocities
            
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
        if len(data_list) > 1 and verbose > 0:
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

    def __repr__(self):
        return f"DataList with {len(self.data_list)} Data instances, storing the orders: {self.orders} out of a total order range: {self.order_range}"

    def __call__(self, **kwargs):
        """Runs and returns the results of the :py:function:`DataList.run_ACID` method, which runs ACID on the Data instances in the list for the specified orders.
        
        Parameters
        ----------
        See :py:function:`DataList.run_ACID` for the accepted parameters and their descriptions.
        """
        return self.run_ACID(**kwargs)

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

    def sort_by_order(self) -> None:
        """
        Sorts the data list by order number, and updates the o2i mapping accordingly. Internally called whenever self.data_list is updated.
        """
        self.data_list.sort(key=lambda data: data.config.order)
        self.o2i = {data.config.order: i for i, data in enumerate(self.data_list)}
        self.i2o = {i: data.config.order for i, data in enumerate(self.data_list)}
        self.orders = np.array([data.config.order for data in self.data_list], dtype=np.int32)

        if len(np.unique(self.orders)) != len(self.orders):
            raise ValueError("All Data instances within the inputted list must have unique order numbers.")

    def run_ACID(
        self,
        orders            : Array1D|int|str|None = None,
        use_index_mapping : bool                 = True,
        worker            : IntLike|None         = None,
        nworkers          : IntLike|None         = None,
        store_sampler     : bool                 = False,
        allow_overwrite   : bool                 = False,
        overwrite_kwargs  : bool                 = False,
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
            If True, the sampler object from the ACID run will be stored in the data pickles. This will take up much more disk space, but allow
            for use of the :py:class:`Result` methods requiring the sampler attribute. Default is False.
        allow_overwrite : bool, optional
            If True, will allow overwriting existing result pickles in the save_dir. Default is False, which will skip running ACID on orders 
            that already have result pickles in the save_dir.
        overwrite_kwargs : bool, optional
            If True, any keys in the kwargs that are also in the config for the Data instance will be overwritten by the kwargs values.
            Use with caution, by default False.
        **kwargs :
            Additional keyword arguments to be passed to the ACID method for each order. These will not overwrite any existing keys unless
            overwrite_kwargs is set to True, in which case they will overwrite existing keys in the config for the Data instance for that order.
            The kwargs passed also allow you to add/overwrite the linelist and velocities in the Data instance with the same overwrite logic.
        """
        from .acid import Acid # local import to avoid circular imports, since Acid imports Data

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

        iterable = tqdm(orders, "Running ACID on orders", unit="order") if self.verbose > 1 else orders
        for order in iterable:
            if self.save_dir is not None:
                # Define save location
                results_dir = os.path.join(self.save_dir, "results")
                os.makedirs(results_dir, exist_ok=True)
                save_path = os.path.join(results_dir, f"order_{order}.pkl")
                if os.path.exists(save_path) and not allow_overwrite:
                    if self.verbose > 1:
                        print(f"ACID result for order {order} already exists at {save_path}. \n"
                                f"Skipping this order. To overwrite existing results, set allow_overwrite=True.")
                    continue
            
            # Handling if any kwargs were input 
            data = self.data_list[self.o2i[order]]
            # Only overwrite if overwrite_kwargs is True, otherwise keep the existing linelist/velocities in the Data instance
            if "linelist" in kwargs:
                ll = kwargs.pop("linelist")
                data.linelist = ll if overwrite_kwargs else data.linelist
            if "velocities" in kwargs:
                vel = kwargs.pop("velocities")
                data.velocities = vel if overwrite_kwargs else data.velocities
            if overwrite_kwargs:
                data.config.update_hipri(**kwargs)
            else:
                data.config.update_lowpri(**kwargs)

            # The following try-except loops came from just testing ACID on a lot of different orders, stars, and instruments
            failed_msg = f"Order {order} (list index {self.o2i[order]}) failed with error:"
            try:
                result = Acid(data=data).ACID() # All params are stored in Data and Config (in Data)
            except LineListRangeError:
                print(f"{failed_msg} line list range error. Your linelist is likely out of "\
                      f"range of the wavelengths. Skipping this order.", flush=True)
                continue
            except ContinuumError:
                print(f"{failed_msg} continuum fitting error. The fitted continuum likely "\
                      f"had negative values. Skipping this order.", flush=True)
                continue
            except SNCutError:
                print(f"{failed_msg} S/N cut error. The S/N of the spectrum is likely too "\
                      f"low, and no lines survived the cut. Skipping this order.", flush=True)
                continue
            # If no known exception arose, just print the last 3 calls in traceback for debugging and skip the order.
            except Exception:
                print(f"{failed_msg} unknown error, see traceback. Skipping this order. Traceback:\n", flush=True)
                tb.print_exc(limit=-3)
                continue

            if self.save_dir is not None:
                result.save(save_path, store_sampler=store_sampler)
        return

    def save(self, save_dir:str|None=None) -> None:
        """
        Saves the DataList to a pickle file. 
        The pickle file contains a dictionary with the list of Data objects (converted to dictionaries) and the save_dir.
        The filename is "datalist.pkl". If save_dir is not provided, self.save_dir is used. If that is also None, a ValueError is raised.

        Parameters
        ----------
        save_dir : str | None, optional
            The directory to save the DataList pickle file. If None, self.save_dir is used. Default is None.
        """
        # TODO: Add DataList handling in tests.py
        d = {}
        if save_dir is not None:
            self.save_dir = save_dir
        if self.save_dir is None:
            raise ValueError("No save path provided and save_dir was not set.")
        d["dict_list"] = [data.to_dict() for data in self.data_list]
        save_loc = os.path.join(self.save_dir, "datalist.pkl")
        d["save_dir"] = self.save_dir
        d["verbose"] = self.verbose
        with open(save_loc, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path:str) -> DataList:
        """
        Loads a DataList from a pickle file. The pickle file should contain a dictionary with the list of Data objects (converted to dictionaries) and the save_dir.
        Will attempt to load from datalist.pkl in the provided path if it is a directory, otherwise will attempt to load from the provided path directly. 
        If neither of those work, it will attempt to load from result pickles in a results directory within the provided path.

        Parameters
        ----------
        path : str
            The path to the pickle file or directory to search for the pickle file.

        Returns
        -------
        DataList
            The loaded DataList object.
        """
        if os.path.isdir(path):
            path_check = os.path.join(path, "datalist.pkl")
            if not os.path.exists(path_check):
                # Final attempt to directly load the result pickles
                if path.endswith("results"):
                    result_path = path
                else:
                    result_path = os.path.join(path, "results")
                if os.path.exists(result_path) and os.path.isdir(result_path):
                    result_files = [f for f in os.listdir(result_path) if f.endswith(".pkl")]
                    if not len(result_files) > 0:
                        raise ValueError(f"No datalist.pkl found in {path}, and no result pickles found in {result_path}.")
                    data_list = []
                    from .result import Result
                    for result_file in result_files:
                        result = Result.load(os.path.join(result_path, result_file))
                        data_list.append(result.data)
                    return cls.from_datalist(data_list, save_dir=path)
                else:
                    raise ValueError(f"No datalist.pkl found in {path}, and no results directory found to look for result pickles.")
            else:
                path = path_check
        else:
            if not os.path.exists(path):
                raise ValueError(f"No pickle file found at {path} to load the DataList from.")

        with open(path, "rb") as f:
            d = pickle.load(f)
        data_list = [Data().from_dict(d) for d in d["dict_list"]]
        verbose = d["verbose"] if "verbose" in d else None
        obj = cls.from_datalist(data_list, save_dir=d["save_dir"], verbose=verbose)
        return obj

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, dir):
        if dir is None:
            self._save_dir = None
            return
        os.makedirs(dir, exist_ok=True)
        self._save_dir = dir

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

    def combine_profiles(self, exclude:int|list|None=None) -> None:
        """
        Calculates the combined profile and its errors across all orders, excluding any orders specified in the exclude argument.
        
        Parameters
        ----------
        exclude : int | list[int] | None
            Orders to exclude from the combined profile calculation.
        """
        if isinstance(exclude, int):
            exclude = [exclude]
        elif exclude is None:
            exclude = []

        if not all(o in self.orders for o in exclude):
            raise ValueError(f"All orders in the exclude list must be in the DataList. \nGot: {exclude!r}, but available orders are: {self.orders!r}")

        profiles = [data.combined_profile[0] for data in self.data_list if data.config.order not in exclude]
        errors = [data.combined_profile[1] for data in self.data_list if data.config.order not in exclude]
        covariances = [data.combined_profile[2] for data in self.data_list if data.config.order not in exclude]

        self._combined_profile = utils.combine_profiles(profiles, errors, covariances)
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
            if data.combined_profile is None or data.config.order in self.excluded_orders:
                continue # failed or excluded orders
            ax.errorbar(self.velocities, data.combined_profile[0], alpha=0.2, color="C0",
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

    def fit_profile(self, **kwargs) -> None|tuple[plt.Figure, plt.Axes]:
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
