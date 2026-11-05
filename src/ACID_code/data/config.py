from __future__ import annotations
import os
import numpy as np
from beartype import beartype
from .. import utils
from beartype.typing import Any
from ..utils import IntLike, Scalar
from ..diagnostics.logging import get_logger
from ..diagnostics.output import Output, normalize_verbose, with_output_settings
from ..diagnostics.errors import ACIDInputError
from .masking_lines import MaskingLines
import matplotlib.pyplot as plt
try:
    import dynesty # type: ignore
except ImportError:
    dynesty = None

logger = get_logger(__name__)

@beartype
class Config:
    """The main class for storing ACID configuration settings, with methods to plot and save/load the configuration state."""

    #: The default configuration settings for ACID, used if not set by the user. See :py:class:`~ACID_code.Acid` for more details on how these are used in ACID.
    defaults = {
        # INIT CONFIGURATION
        "verbose" : 2,
        "sampler_progress" : None,
        "order" : None,
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
        "n_bins" : 20,
        "bin_size" : None,
        "pix_chunk" : 50,
        "dev_perc" : 25,
        "sigma_lower" : 3,
        "sigma_upper" : 5,
        "skips" : 1,
        "od"    : True,
        "sparse" : True,
        "regularization" : 0.0,
        "scale_regularization" : True,
        "depth_group_rules" : None,
        "profile_groups" : None,
        "profile_names" : None,
        "seperate_species" : None,
        "sampler_type" : "emcee",
        "use_jax" : False,
        "parallel" : True,
        "cores" : None,
        "nwalkers" : None,
        "nsteps" : 10000,
        "max_steps" : None,
        "check_interval" : 1000,
        "min_checks" : 1,
        "min_tau_factor" : 50,
        "tau_tol" : 0.1,
        "reduced_tau_tol" : 0.05, # defaults to this if tau_tol is not explicitly set
        "moves" : [
            ("StretchMove", 0.20, {}),
            ("DESnookerMove", 0.1, {}),
            ("DEMove", 0.6, {}),
            ("DEMove", 0.1, {"gamma0": 1.0}),
        ],
        "run_mcmc" : True,
        "resume" : False,
        "continuum_method" : None,
    }

    #: Property list for error handling
    properties = ["verbose", "masking_lines", "dir", "save_path", "sampler_path", "figure_dir", "continuum_method", "sampler_type",
                  "regularization", "tau_tol"]
    _properties = [f"_{prop}" for prop in properties]

    #: For error handling if Data attributes were accidentally set in config. These should be set in :py:class:`~ACID_code.Data` instead
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

    def __getattribute__(self, name:str) -> Any:
        """Override the default attribute access to allow for environment variable overrides."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)

        raw = os.environ.get("_ACID_CONFIG")

        if raw is not None:
            import json

            env_config = json.loads(raw)

            if not isinstance(env_config, dict):
                raise ACIDInputError("_ACID_CONFIG must decode to a dictionary")

            if name in env_config:
                return env_config[name]

        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.data_attributes:
            raise ACIDInputError(self.data_attributes_input_str.format(name, name, value))

        if value is None:
            # If value is None, do not set the attribute
            return

        # Validate naming and automatic species inputs even when Config is used directly.
        if name == "profile_names":
            utils.profile_name_map(value)
        elif name == "seperate_species":
            utils.validate_seperate_species(value)

        if name in self._properties or name in self.defaults:
            super().__setattr__(name, value)
            return

        raise ACIDInputError(
            f"'Config' object has no attribute '{name}', "
            f"valid attributes are: {list(self.defaults.keys())}"
        )

    def __repr__(self) -> str:
        """String representation of the Config object, showing all settings in a user-friendly format."""
        full_dict = self.to_dict()
        return f"Config instance with the following settings:\n" + "\n".join([f"{k}: {v}" for k, v in full_dict.items()])

    # --- Update methods ---
    def update_hipri(self, force:bool|np.bool_=False, **kwargs: Any) -> None:
        """Updates and overwrites existing keys if the overwriting value is not None or if force is True.
        
        Parameters
        ----------
        force : bool, optional
            If True, will overwrite existing configuration regardless of None values. Default is False.
        **kwargs : dict
            Keyword arguments corresponding to the configuration settings to be updated. 
            The keys must be valid configuration options as defined in the :py:attr:`~ACID_code.Config.defaults` class variable.
        
        Raises
        ------
        ACIDInputError
            If any key in ``kwargs`` is not a valid configuration option as defined in the :py:attr:`~ACID_code.Config.defaults` class variable.
        """
        for k, v in kwargs.items():
            # First raise error if Data attribute was input
            if k in self.data_attributes:
                raise ACIDInputError(self.data_attributes_input_str.format(k, k, v))
            # Then raise error if trying to set an attribute that is not in defaults
            if k not in self.defaults and k not in self._properties:
                raise ACIDInputError(f"Key '{k}' is not a valid configuration option.")
            if v is None:
                if force: # If forced, super set the attribute
                    stored_name = "_" + k if k in self.properties else k
                    super().__setattr__(stored_name, None)
                # else: just skips the None
            else:
                setattr(self, k, v)

    def update_lowpri(self, **kwargs: Any) -> None:
        """Updates but does not overwrite existing stored keys.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments corresponding to the configuration settings to be updated. 
            The keys must be valid configuration options as defined in the :py:attr:`~ACID_code.Config.defaults` class variable.

        Raises
        ------
        ACIDInputError
            If any key in ``kwargs`` is not a valid configuration option as defined in the :py:attr:`~ACID_code.Config.defaults` class variable.
        """
        for k, v in kwargs.items():
            # First raise error if Data attribute was input
            if k in self.data_attributes:
                raise ACIDInputError(self.data_attributes_input_str.format(k, k, v))
            # Then raise error if trying to set an attribute that is not in defaults
            if k not in self.defaults:
                raise ACIDInputError(f"Key '{k}' is not a valid configuration option.")

            if v is None:
                continue

            stored_name = "_" + k if k in self.properties else k # Add the _ for properties
            if stored_name not in self.__dict__ or self.__dict__[stored_name] is None:
                setattr(self, k, v)

    def to_dict(self) -> dict:
        """Convert the Config object to a dictionary of only the stored/modified attributes,
        if you want the full dictionary including defaults, use :py:meth:`~ACID_code.Config.to_full_dict`."""
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
    def tau_tol(self) -> float:
        """Tau stability tolerance; defaults to reduced_tau_tol when neff is disabled."""
        stored_tau_tol = self.__dict__.get("_tau_tol")
        if stored_tau_tol is not None:
            return stored_tau_tol
        return self.reduced_tau_tol if self.min_tau_factor == 0 else self.defaults["tau_tol"]

    @tau_tol.setter
    def tau_tol(self, value:Scalar) -> None:
        self._tau_tol = float(value)

    # - Path and directory properties
    @property
    def regularization(self) -> float:
        """First-difference smoothing strength; zero disables regularization."""
        return self.__dict__.get("_regularization", self.defaults["regularization"])

    @regularization.setter
    def regularization(self, value:Scalar) -> None:
        if not np.isfinite(value) or value < 0:
            raise ACIDInputError("regularization must be finite and non-negative.")
        self._regularization = float(value)

    @property
    def dir(self) -> str|None:
        """Root directory for data, sampler, and figure output."""
        return self.__dict__.get("_dir", self.defaults["dir"])

    @dir.setter
    def dir(self, value:str|None) -> None:
        """Store the output root; directories are created when writing output."""
        if value is None:
            return

        if not value:
            raise ACIDInputError("The data output directory path cannot be empty.")
        self._dir = os.path.abspath(os.path.expanduser(value))

    @property
    def save_path(self) -> str|None:
        """Path to save the ACID data object."""
        stored_save_path = self.__dict__.get("_save_path", self.defaults["save_path"])
        if self._no_save(stored_save_path):
            return None
        stored_dir = self.__dict__.get("_dir", self.defaults["dir"])
        if stored_save_path is not None:
            return stored_save_path
        elif stored_dir is not None:
            return os.path.join(stored_dir, "data.pkl")
        else:
            return None

    @save_path.setter
    def save_path(self, save_path:str|None) -> None:
        """Set the path to save the ACID data object."""
        if save_path is None:
            return
        if self._no_save(save_path):
            self._save_path = save_path
            return

        if not save_path.endswith(".pkl"):
            raise ACIDInputError("'save_path' must end with '.pkl'.")
        save_path = os.path.abspath(save_path)
        self._save_path = save_path

    @property
    def sampler_path(self) -> str|None:
        """Path to save the ACID sampler object."""
        stored_sampler_path = self.__dict__.get("_sampler_path", self.defaults["sampler_path"])
        if self._no_save(stored_sampler_path):
            return None
        stored_dir = self.__dict__.get("_dir", self.defaults["dir"])
        if stored_sampler_path is not None:
            return stored_sampler_path
        elif stored_dir is not None:
            return os.path.join(stored_dir, "sampler.h5")
        else:
            return None

    @sampler_path.setter
    def sampler_path(self, sampler_path:str|None) -> None:
        """Set the path to save the ACID sampler object."""
        if sampler_path is None:
            return
        if self._no_save(sampler_path):
            self._sampler_path = sampler_path
            return

        if not sampler_path.endswith(".h5"):
            raise ACIDInputError("'sampler_path' must end with '.h5'.")
        sampler_path = os.path.abspath(sampler_path)
        self._sampler_path = sampler_path

    @property
    def figure_dir(self) -> str|None:
        """Directory to save figures generated by ACID."""
        stored_figure_dir = self.__dict__.get("_figure_dir", self.defaults["figure_dir"])
        if self._no_save(stored_figure_dir):
            return None
        stored_dir = self.__dict__.get("_dir", self.defaults["dir"])
        if stored_figure_dir is not None:
            return stored_figure_dir
        elif stored_dir is not None:
            return os.path.join(stored_dir, "figures")
        else:
            return None

    @figure_dir.setter
    def figure_dir(self, figure_dir:str|None) -> None:
        """Set the directory to save figures generated by ACID."""
        if figure_dir is None:
            return
        if self._no_save(figure_dir):
            self._figure_dir = figure_dir
            return

        if not figure_dir:
            raise ACIDInputError("The figure directory path cannot be empty.")
        self._figure_dir = os.path.abspath(os.path.expanduser(figure_dir))

    @staticmethod
    def _no_save(path:object) -> bool:
        """Allow no saving overrides for paths over the dir."""
        if isinstance(path, str):
            path = path.lower()
            if path in ["none", "no", "false", "off", "n", "0"]:
                return True
        return False

    @property
    def verbose(self) -> IntLike:
        """The stored verbosity setting for ACID. See :py:class:`~ACID_code.Acid` for more details on how this is used in ACID."""
        return self.__dict__.get("_verbose", self.defaults["verbose"])

    @verbose.setter
    def verbose(self, value:IntLike|str|bool|None) -> None:
        """Store verbosity without changing logging or another instance's output."""
        if value is not None:
            self._verbose = normalize_verbose(value, self.defaults["verbose"])

    @property
    def output(self) -> Output:
        """Runtime output policy, resolved from the active scope or this Config."""
        return Output.current(self)

    @property
    def masking_lines(self) -> MaskingLines:
        """The stored masking lines for ACID. See :ref:`masking_lines` for more details on how this is used in ACID."""
        return MaskingLines(self.__dict__.get("_masking_lines", self.defaults["masking_lines"]))

    @masking_lines.setter
    def masking_lines(self, masking_lines:dict|MaskingLines|None) -> None:
        """Set the masking lines for ACID. Accepts a dictionary, a MaskingLines object, or None."""
        if masking_lines is not None:
            self._masking_lines = MaskingLines.validate_lines(masking_lines)

    @property
    def sampler_type(self) -> str:
        """The stored sampler type for ACID. See :py:class:`~ACID_code.Acid` for more details on how this is used in ACID."""
        stored_sampler_type = self.__dict__.get("_sampler_type", self.defaults["sampler_type"])

        # We validate on access (after deterministic_profile and max_steps are set)
        if stored_sampler_type == "dynesty":
            if not self.deterministic_profile:
                raise ACIDInputError("The 'dynesty' sampler can only be run with deterministic_profile=True (otherwise you'll be waiting days for a single result)")
            if self.max_steps is not None:
                raise ACIDInputError("Cannot use max_steps as dynesty already natively supports this with live points, set nsteps=nlive. See the dynesty docs for more details.")

        return stored_sampler_type

    @sampler_type.setter
    def sampler_type(self, sampler_type:str) -> None:
        if sampler_type not in ["emcee", "dynesty"]:
            raise ACIDInputError("Invalid 'sampler_type' input, must be one of ['emcee', 'dynesty'].")
        if sampler_type == "dynesty":
            if dynesty is None:
                raise ImportError("The 'dynesty' sampler requires the 'dynesty' package to be installed.\nPlease install it with 'pip install dynesty' or choose a different sampler type.")
        self._sampler_type = sampler_type
    
    @property
    def continuum_method(self) -> str:
        """The stored continuum method for ACID. See :py:class:`~ACID_code.Acid` for more details on how this is used in ACID."""
        continuum_method = self.__dict__.get("_continuum_method", self.defaults["continuum_method"])

        # Return the default continuum method based on the polynomial order if not explicitly set
        if continuum_method is None:
            if self.poly_ord <= 5:
                continuum_method = "polyval"
            else:
                continuum_method = "chebval"

        return continuum_method

    @continuum_method.setter
    def continuum_method(self, continuum_method:str) -> None:
        if continuum_method not in ["polyval", "chebval"]:
            raise ACIDInputError("Invalid 'continuum_method' input, must be one of ['polyval', 'chebval'].")

        self._continuum_method = continuum_method
    
    # --- Plotting/info methods ---
    @with_output_settings
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
                delta_lambda = line * width / utils.c_kms
                ax.axvline(line, linestyle='--', color=f'C{i+1}',
                           label=f'{name.capitalize()} line' if line == line_data["lines"][0] else None)
                ax.axvspan(line - delta_lambda, line + delta_lambda, alpha=0.1, color=f'C{i+1}')

        ax.set_title("Lines to be masked")
        ax.set_xlabel("Wavelength (Angstroms)")
        ax.set_ylabel("Masking region")
        ax.legend()
        if return_fig:
             return fig, ax
        utils.show_or_save(plt, self.figure_dir, "masking_lines.png", output_dir=self.dir)

    @classmethod
    def print_defaults(cls, use_logger:bool=True) -> None:
        """Print the default configuration settings for ACID.

        Parameters
        ----------
        use_logger : bool, optional
            Whether to use the logger for output, else uses print(), by default True.
        """
        # Determine output function
        if use_logger:
            out = logger.info
        else:
            out = print

        # Print defaults
        out("Default configuration:")
        for k, v in cls.defaults.items():
            out(f"{k}: {v}")
