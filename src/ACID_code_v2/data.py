from dataclasses import dataclass, field, fields
from beartype import beartype
from typing import Any, Dict, Optional
from .utils import Array1D, c_kms
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle, os
import numpy as np
from . import utils
from .utils import IntLike, Array1D, Array2D
from .mcmc import MCMC

class TelluricLines:
    """A simple class to expose the telluric lines when called in Config. This will help
    to store telluric lines as a dictionary. With a default itercall to list the line-wise elements,
    but a dictionary index to also store the width of the line, which can then allow for masking Hydrogen
    lines with much wider masks."""
    __slots__ = ("lines",) # the only thing stored in this class is this dictionary
    def __init__(self, lines:dict):
        if len(lines["widths"]) != len(lines["lines"]):
            raise ValueError("If 'widths' are provided in the telluric_lines dictionary, they must have the same length as 'lines'")
        self.lines = lines

    def __iter__(self):
        yield self.lines["lines"]
        yield self.lines.get("widths", None)

    def __getitem__(self, key):
        if key == 0:
            return self.lines["lines"]
        if key == 1:
            return self.lines["widths"]
        if isinstance(key, int):
            raise IndexError("TelluricLines only has keys 0 and 1, or 'lines' and 'widths'")
        return self.lines[key]

    def get_mask(self, x):
        lines = np.asarray(self.lines["lines"])
        widths = np.asarray(self.lines["widths"])

        limits = 3 + (widths / c_kms) * lines
        conditions = np.abs(x[None, :] - lines[:, None]) <= limits[:, None]
        telluric_mask = np.any(conditions, axis=0)
        return telluric_mask

@beartype
class Config:
    """A simple class to store the configuration of the ACID run."""

    defaults = {

        # INIT CONFIGURATION
        "verbose" : 2,
        "order" : 0,
        "order_range" : [0],
        "telluric_width" : 21, # in km/s, if changed, change below default widths too
        "telluric_lines" : {
            "lines" : [
                3820.33, # metal?
                3835.38, # H eta (new)
                3889.05, # H zeta (new)
                3933.66, # Ca II K
                3968.47, # Ca II H
                4101.74, # H delta (new)
                4307.90, # metal?
                4327.74, # metal?
                4340.47, # H gamma (new)
                4383.55, # Fe 1
                4861.34, # H beta
                5183.62, # Mg I b triplet
                5270.39, # Fe 1
                5889.95, # Na I D2
                5895.92, # Na I D1
                6562.81, # H alpha
                7593.70, # O2 telluric
                8226.96, # H2O telluric?
            ],
            "widths": [
                21, # metal?
                1000, # H eta
                1000, # H zeta
                21, # Ca II K
                21, # Ca II H
                1000, # H delta
                21, # metal?
                21, # metal?
                1000, # H gamma
                21, # Fe 1
                1000, # H beta
                21, # Mg I b triplet
                21, # Fe 1
                21, # Na I D2
                21, # Na I D1
                1000, # H alpha
                21, # O2 telluric
                21, # H2O telluric?
            ],
            },

        # RUN_ACID CONFIGURATION
        "deterministic_profile" : True,
        "poly_ord" : 3,
        "continuum_percentile" : 90,
        "bin_size" : 100,
        "pix_chunk" : 20,
        "dev_perc" : 25,
        "n_sig" : 1,
        "skips" : 1,
        "parallel" : True,
        "cores" : None,
        "nsteps" : 10000,
        "max_steps" : None,
        "check_interval" : 1000,
        "min_checks" : 1,
        "min_tau_factor" : 50,
        "tau_tol" : 0.05,
        "run_mcmc" : True,
    }

    def __init__(self, **kwargs) -> None:
        
        # Set defaults
        self.update_hipri(**self.defaults) # Set defaults as hipri, so they can be overwritten by user inputs in kwargs

        # Override with any inputted kwargs
        self.update_hipri(**kwargs) # Set initial values, allowing overwriting and validation of properties

    # --- Update methods ---
    def update_hipri(self, **kwargs: Any) -> None:
        # Update and overwrite existing keys
        for k, v in kwargs.items():
            if v is None:
                continue
            else:
                setattr(self, k, v)

    def update_lowpri(self, **kwargs: Any) -> None:
        # Update but do not overwrite existing keys
        for k, v in kwargs.items():
            if v is None:
                continue
            if getattr(self, k, None) is None:
                setattr(self, k, v)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    # --- Properties ---
    @property
    def verbose(self) -> int:
        if self._verbose is None:
            return self.defaults["verbose"]
        return self._verbose

    @verbose.setter
    def verbose(self, value) -> None:
        # Make verbosity always an int regardless of input type, and check correct range
        if value is True:
            value = self.defaults["verbose"]
        elif value is False:
            value = 0
        elif isinstance(value, int):
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
        elif value is None:
            value = self.defaults["verbose"]
        else:
            raise ValueError("verbose must be an integer between 0 and 3, a boolean, or a string indicating the verbosity level")

        self._verbose = value

    @property
    def telluric_lines(self) -> int:
        if self._telluric_lines is None:
            return TelluricLines(self.defaults["telluric_lines"])
        return TelluricLines(self._telluric_lines)

    @telluric_lines.setter
    def telluric_lines(self, lines:Array1D|Array2D|dict|TelluricLines) -> None:
        telluric_lines = lines
        # Define telluric_lines with defaults if not input, check type if it is
        # TODO MAKE THE default for inputted telluric lines values to be the widths if lines within +-1 of any of the defaults
        if telluric_lines is None:
            lines = self.defaults["telluric_lines"]["lines"]
            widths = self.defaults["telluric_lines"]["widths"]
        elif isinstance(telluric_lines, (np.ndarray, list)):
            telluric_lines = np.array(telluric_lines)
            if telluric_lines.size == 0:
                raise ValueError("telluric_lines cannot be an empty array or list")                
            if telluric_lines.ndim == 1:
                lines = np.array(telluric_lines)
                widths = self.defaults["telluric_lines"]["widths"]
            elif telluric_lines.ndim == 2:
                lines = telluric_lines[0]
                widths = telluric_lines[1]
            else:
                raise ValueError("telluric_lines must be a one- or two-dimensional array or list")
        elif isinstance(telluric_lines, dict):
            if "lines" not in telluric_lines:
                raise ValueError("If telluric_lines is a dict, it must contain a 'lines' key with the list/array of lines to mask")
            lines = np.array(telluric_lines["lines"])
            widths = telluric_lines.get("widths", None)
            widths = widths if widths is not None else self.defaults["telluric_lines"]["widths"]
            widths = np.array(widths)
        elif isinstance(telluric_lines, TelluricLines):
            lines = telluric_lines[0]
            widths = telluric_lines[1]
        else:
            raise ValueError("telluric_lines must be a list or numpy array of telluric lines to " \
            f"mask in angstroms. \n A dictionary can also be provided with " \
            f"a 'lines' key containing the list/array of lines, \n and an optional 'widths' key containing the widths " \
            f"of the lines for user reference. \n Got type: {type(telluric_lines)}")
        telluric_lines = {"lines": lines, "widths": widths}
        self._telluric_lines = telluric_lines

@dataclass(slots=True)
class Data:
    """Stores necessary data for the Acid class which can be conveniently updated and saved.
    Allows ACID to handle data that has already been computed to avoid recalculation. This class
    is designed to be lightweight in memory and hence does not store the sampler as an object.
    Note that a Data class should only hold the data for ONE order or observation, but it can hold
    the data for multiple frames of the same order."""

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
    nanmask                : Optional[np.ndarray] = None  # boolean 1D mask on "combined" grid, used to mask out NaN values in combined spectra
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
    profiles : Optional[np.ndarray] = None  # the array to store all frames of the MCMC sampling
    nsteps     : Optional[int]        = 0
    max_steps  : Optional[int]        = None

    # Other useful data and figures:
    initialisation_time : Optional[float] = None  # time taken for initialization
    mcmc_time           : Optional[float] = None  # time taken for MCMC sampling
    get_profiles_time   : Optional[float] = None  # time taken to get profiles
    full_run_time       : Optional[float] = None  # total time for the full run
    plotting_variables  : Dict[str, Any]  = field(default_factory=dict)

    # Initialise the properties
    # Config data for convenience, it is very memory light so not an issue to also store in here
    _config   : Config = field(default_factory=Config) # config stored as class, but converted to dict on save
    _linelist : Optional[Dict[str, np.ndarray]] = None

    def plot_continuum_fit(self, plot_type:str="initial", return_fig:bool=False, save_fig:str|None=None) -> None:
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

        unnormalized_wavelengths = self.plotting_variables[plot_type]["unnormalized_wavelengths"]
        fluxes                   = self.plotting_variables[plot_type]["fluxes"]
        good                     = self.plotting_variables[plot_type]["good"]
        fit                      = self.plotting_variables[plot_type]["fit"]
        clipped_waves            = self.plotting_variables[plot_type]["clipped_waves"]
        clipped_flux             = self.plotting_variables[plot_type]["clipped_flux"]

        a, b = utils.get_normalisation_coeffs(unnormalized_wavelengths)
        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(unnormalized_wavelengths, fluxes, label='Original Spectrum', color="C0", alpha=0.7)
        ax.plot(unnormalized_wavelengths, fit, label='Fitted Continuum', color='red')
        ax.plot((clipped_waves[good]-b)/a, clipped_flux[good], 'o', label='Continuum Normalized Spectrum', color='green')

        # Plot bad regions:
        masked = ~good
        padded = np.concatenate(([False], masked, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i, (start, end) in enumerate(zip(starts, ends)):
            ax.axvspan((clipped_waves[start]-b)/a, (clipped_waves[end-1]-b)/a,
                        color='red', alpha=0.15, label="Bad regions" if i == 0 else None)

        ll_wl = self.linelist["wavelengths"]
        ll_depths = self.linelist["depths"]

        ll_wl, ll_depths = utils.clip_wavelengths(unnormalized_wavelengths, ll_wl, ll_depths)
        idx = np.argsort(ll_depths)
        ll_wl = ll_wl[idx]
        ll_depths = ll_depths[idx]
        ll_wl = ll_wl[-20:]
        ll_depths = ll_depths[-20:]

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
            print("There was an error plotting the linelist points, most likely your linelist range is outside your wavelength range.")
            pass
        plot_title = "Initial Continuum Fit" if plot_type == "initial" else "Continuum Fit after Residual Masking"
        ax.set_title(plot_title)
        ax.legend()
        if save_fig is not None:
            plt.savefig(save_fig)
        if return_fig:
            return fig, ax
        plt.show()

    def plot_residual_masking(self, save_fig:str|None=None) -> None:
        if "residual_masking" not in self.plotting_variables:
            raise ValueError("No plotting variables found for residual_masking. ")
        if not all(
            attr in self.plotting_variables["residual_masking"] for attr in [
                "mask", "residuals", "upper_clip", "lower_clip", "telluric_mask", "pix_mask", "profile_F"]
        ):
            raise ValueError("Not all required plotting variables found for residual_masking. ")
        if "masked" not in self.wavelengths and "masked" not in self.flux:
            raise ValueError("No masked wavelengths or fluxes found. Please ensure that the residual masking step has been performed")
        if save_fig is not None:
            if not os.path.isdir(save_fig):
                raise ValueError(f"save_fig must be a valid path to a directory to save the figures, or None to show the figures. Got: {save_fig}")

        x = self.wavelengths["masked"]
        y = self.flux["masked"]
        mask = self.plotting_variables["residual_masking"]["mask"]
        residuals = self.plotting_variables["residual_masking"]["residuals"]
        upper_clip = self.plotting_variables["residual_masking"]["upper_clip"]
        lower_clip = self.plotting_variables["residual_masking"]["lower_clip"]
        telluric_mask = self.plotting_variables["residual_masking"]["telluric_mask"]
        pix_mask = self.plotting_variables["residual_masking"]["pix_mask"]
        profile_F = self.plotting_variables["residual_masking"]["profile_F"]

        nremoved = np.sum(mask)
        print(f"Residual masking has removed {nremoved}/{len(residuals)} points.")

        fig, ax = plt.subplots(figsize=(15, 9))
        ax.plot(x, residuals, label='Residuals', color='blue')
        ax.axhline(upper_clip, color='red', linestyle='--', label='Upper Clip Threshold')
        ax.axhline(lower_clip, color='green', linestyle='--', label='Lower Clip Threshold')
        ax.axhspan(lower_clip, upper_clip, color='gray', alpha=0.3, label='Sigma Clipping masking range')
        
        # Show telluric masking regions
        masked = telluric_mask
        padded = np.concatenate(([False], masked, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends   = np.flatnonzero(padded[:-1] & ~padded[1:])
        for i, (start, end) in enumerate(zip(starts, ends)):
            ax.axvspan((x[start]), (x[end-1]),
                        color='orange', alpha=0.3, label="Telluric masking" if i == 0 else None)

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
        # Validate input arrays using the validate_args function within utils.py, ensuring inputs are correct shape, or to
        # best guess the user's intentions. See the utils.validate_args function for more details. This also converts
        # inputs to numpy arrays.
        if input_wavelengths is None or input_flux is None or input_errors is None:
            if "input" not in self.wavelengths and "input" not in self.flux and "input" not in self.errors:
                raise ValueError("input_wavelengths, input_flux, and input_errors must be provided either as arguments " \
                "to this function or in a Data object.")

        input_wavelengths, input_flux, input_errors = [
            utils.validate_args(arg, i) for i, arg in enumerate((input_wavelengths, input_flux, input_errors))]
        input_sn = utils.validate_args(input_sn, 3, sn=True, allow_none=True)

        # Check all inputs have the same shape
        if not input_wavelengths.shape == input_flux.shape == input_errors.shape:
            raise ValueError("Input wavelengths, spectra and spectral errors must all have the same shape.")

        input_wavelengths, input_flux, input_errors = utils.mask_invalid(input_wavelengths, input_flux, input_errors, verbose=self.config.verbose)

        if input_sn is None:
            input_sn = utils.guess_SNR(input_wavelengths, input_flux, input_errors)
        if input_sn.shape[0] != input_flux.shape[0]:
            raise ValueError("The number of frames for the SN must match the number of frames in wavelengths, flux, and errors.")
        if input_sn.ndim == input_flux.ndim:
            # Per pixel S-N provided, take the mean over the central 2/3 of the wavelengths
            input_sn = utils.collapse_SNR(input_sn, input_wavelengths)
        elif input_sn.ndim != input_flux.ndim-1:
            raise ValueError("input_sn must be either a single-valued list/array with the average S/N for each frame, " \
            f"or an array of S/N values for each pixel. \n" \
            "The shape of the input input_sn does not match the number of frames in input_flux, \
            nor does it have one more dimension than input_flux.")
        assert input_sn.ndim == input_flux.ndim - 1, \
            "input_sn.ndim and input_flux.ndim-1 do not match"

        # Apply skips
        input_wavelengths = input_wavelengths[:, ::skips]
        input_flux       = input_flux[:, ::skips]
        input_errors     = input_errors[:, ::skips]

        # In case these are set when input values already exist, check if they are the same, if not, reset variables to be recalculated.
        to_check = ["wavelengths", "flux", "errors"]
        reset = False
        for check in to_check:
            if getattr(self, check).get("input", None) is not None and eval(f"input_{check}") is not None:
                if not np.array_equal(getattr(self, check)["input"], eval(f"input_{check}")):
                    reset = True
                    continue
                if not np.allclose(getattr(self, check)["input"], eval(f"input_{check}")):
                    reset = True
        if reset:
            if self.config.verbose > 0:
                print("Warning: input wavelengths, flux, or errors have been changed from their previous values. \n" \
                "Resetting variables that need to be recalculated. The velocity grid will not be reset.")
            self.alpha          = None
            self.c_factor       = None
            self.residual_masks = None
            self.wavelengths    = {"input": input_wavelengths}
            self.flux           = {"input": input_flux}
            self.errors         = {"input": input_errors}
            self.sn             = {"input": input_sn}

        # Apply skips and set inputs to class variables
        self.wavelengths["input"] = input_wavelengths
        self.flux["input"]        = input_flux
        self.errors["input"]      = input_errors
        self.sn["input"]          = input_sn

    def save(self, filename:str="data.pkl") -> None:
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
    def load(cls, filename: str):
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

        linelist_wl, linelist_depths = Linelist.validate_linelist(linelist_wl, linelist_depths, linelist_path)
        linelist_wl = np.array(linelist_wl)
        linelist_depths = np.array(linelist_depths)
        linelist_wl, linelist_depths = Linelist.drop_NaNs(linelist_wl, linelist_depths)
        Linelist.validate_dimensions(linelist_wl, linelist_depths)
        self._linelist = {"wavelengths": linelist_wl, "depths": linelist_depths}

@beartype
class Datalist:
    """A class that stores Data instances in a list indexed by order. Holds some useful methods for analysis or to be called
    by result. This can map the order number of an instrument to the 0-indexed python list."""
    def __init__(self, data_list:list[Data]|Data, save_dir:str|None=None, verbose:IntLike|bool|None=None):
        # All configs should have the same order_range so that they are internally aware. We just take the first one to 
        # generate the mapping of order to index in the list. The Load class will configure the correct order range based
        # off extracted fits header info (if provided), otherwise the default is a pythonic 0-indexed order range.
        if isinstance(data_list, Data):
            data_list = [data_list]

        if verbose is not None:
            self.data[0].config.verbose = verbose # verbose in config is a property and has good validation
            self.verbose = self.data[0].config.verbose
        else:
            self.verbose = np.max([data.config.verbose for data in data_list])

        order_range = data_list[0].config.order_range
        if len(data_list) > 1 and self.verbose > 0:
            if not all(np.array_equal(data.config.order_range, order_range) for data in data_list):
                print("Warning: Not all Data instances have the same order_range. Taking the longest order range.")

        max_order_range_idx = np.argmax([len(data.config.order_range) for data in data_list])
        self.order_range = data_list[max_order_range_idx].config.order_range

        self.data_list = data_list
        self.sort_by_order() # generates self.orders and self.o2i

        self._save_dir = None
        self.save_dir = save_dir # property setter handles input

    def __iter__(self):
        yield from self.data_list

    def __getitem__(self, k):
        return self.data_list[self.o2i[k]]

    def __repr__(self):
        return f"DataList with {len(self.data_list)} Data instances, storing the orders: {self.orders} out of a total order range: {self.order_range}"

    def append(self, data:Data, overwrite:bool=False, extend:bool=False, force_order:IntLike|None=None) -> None:
        """Appends a Data instance to the data list
        "Note that the order range of the class is kept, if you want to set a new order range, \n" \
        "Use the set_order_range() method first to change it."
        """
        if force_order is not None:
            data.config.order = force_order
        order = data.config.order
        if order in self.orders and overwrite is False:
            raise ValueError(f"A Data instance with order {order} already exists in the list. " \
            "If you want to overwrite it, set overwrite=True in the append method.")
        if order not in self.order_range:
            if not extend:
                raise ValueError("The order of the appended data class does not match the rest of the list. \n" \
                            "If you want to extend the order_range to append the new order, set extend=True. \n" \
                            "Note that the order range of the class is kept, if you want to set a new order range, \n" \
                            "Use the set_order_range() method first to change it.")
            else:
                self.order_range = np.append(self.order_range, order).astype(np.int32)
        
        if overwrite and data.config.order in self.orders:
            self.data_list[self.o2i[order]] = data
        else:
            self.data_list.append(data)
        
        self.sort_by_order() # re-sorts the list and updates the o2i mapping

    def set_order_range(self, order_range:Array1D):
        "A list or numpy array of the order range"
        if np.any([o not in order_range for o in self.orders]):
            raise ValueError("The already saved orders must be a subset of the inputted order_range.")
        self.order_range = np.array(order_range, dtype=np.int32)

    def sort_by_order(self):
        """Sorts the data list by order number, and updates the o2i mapping accordingly."""
        self.data_list.sort(key=lambda data: data.config.order)
        self.o2i = {data.config.order: i for i, data in enumerate(self.data_list)}
        self.i2o = {i: data.config.order for i, data in enumerate(self.data_list)}
        self.orders = [data.config.order for data in self.data_list]

        if len(np.unique(self.orders)) != len(self.orders):
            raise ValueError("All Data instances within the inputted list must have unique order numbers.")

    def save(self, save_dir:str=None):
        d = {}
        self.save_dir = save_dir
        if self.save_dir is None:
            raise ValueError("No save path provided and save_dir was not set.")
        d["dict_list"] = [data.to_dict() for data in self.data_list]
        save_loc = os.path.join(self.save_dir, "datalist.pkl")
        d["save_dir"] = self.save_dir
        with open(save_loc, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path:str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        data_list = [Data().from_dict(d) for d in d["dict_list"]]
        obj = cls(data_list, d["save_dir"])
        return obj

    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, dir):
        if dir is None:
            return # do not change save_dir
        if not os.path.isdir(dir):
            raise ValueError(f"save_dir must be a valid path to a directory to save the datalist, or None to not save to disk. Got: {dir}")
        self._save_dir = dir

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
    def validate_linelist(linelist_wl, linelist_depths, linelist_path=None):
        if (linelist_wl is None and linelist_depths is None) and linelist_path is None:
            raise ValueError("One of ('linelist_wl' and 'linelist_depths') or 'linelist_path' must be provided.")
        elif linelist_path is None and (linelist_wl is None or linelist_depths is None):
            raise ValueError("If 'linelist_path' is not provided, both 'linelist_wl' and 'linelist_depths' must be provided.")
        elif isinstance(linelist_path, str):
            # VALD linelist code, will add more linelist formats in the future or if requested
            full_linelist = np.genfromtxt('%s'%linelist_path, skip_header=4, delimiter=',', usecols=(1,9), invalid_raise=False)
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
        elif linelist_wl is not None and linelist_depths is not None:
            pass # linelist_wl and linelist_depths already set, will be processed below
        else:
            raise ValueError(f"'linelist_path' must be a string path to a VALD linelist, a dictionary with keys 'wavelengths' and 'depths', \n" \
            "a Linelist object, or a list/array indexed such that 0 is wavelengths and 1 is depths.")
        return linelist_wl, linelist_depths

    @staticmethod
    def validate_dimensions(wavelengths, depths):
        if wavelengths.ndim != 1 or depths.ndim != 1:
            raise ValueError("'wavelengths' and 'depths' must be a one-dimensional array or list")
        if wavelengths.shape != depths.shape:
            raise ValueError("'wavelengths' and 'depths' must have the same length and shape")

    @staticmethod
    def drop_NaNs(wavelengths, depths, return_mask=False, verbose=0):
        mask = np.isfinite(wavelengths) & np.isfinite(depths)
        count_dropped = np.count_nonzero(~mask)
        mask &= (wavelengths > 0) & (depths > 0)
        if verbose > 0 and count_dropped > 0:
            print(f"Your linelist includes {count_dropped} non-finite/nan values, these will be removed, but it is still recommended to check your linelist.")

        if return_mask:
            return wavelengths[mask], depths[mask], mask
        return wavelengths[mask], depths[mask]
