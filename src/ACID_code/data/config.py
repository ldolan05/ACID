from __future__ import annotations
import os
import numpy as np
from beartype import beartype
from .. import utils
from beartype.typing import Any
from ..utils import IntLike
from .masking_lines import MaskingLines
import matplotlib.pyplot as plt

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
        "n_bins" : 20,
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
    properties = ["verbose", "masking_lines", "dir"]
    _properties = ["_verbose", "_masking_lines", "_dir"]

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
    def update_hipri(self, force=False, **kwargs: Any) -> None:
        """Updates and overwrites existing keys if the overwriting value is not None or if force is True.
        
        Parameters
        ----------
        force : bool, optional
            If True, will overwrite existing configuration regardless of None values. Default is False.
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
    def dir(self) -> str|None:
        """Root directory for data, sampler, and figure output."""
        return self.__dict__.get("_dir", self.defaults["dir"])

    @dir.setter
    def dir(self, value:str|None) -> None:
        """Set the output root, creating only the final directory component."""
        if value is None:
            return

        old_dir = self.__dict__.get("_dir", None)
        value = utils.ensure_directory(value, "output directory")
        self._dir = value

        derived_paths = {
            "save_path": os.path.join(value, "data.pkl"),
            "sampler_path": os.path.join(value, "sampler.h5"),
            "figure_dir": os.path.join(value, "figures"),
        }
        old_paths = {} if old_dir is None else {
            "save_path": os.path.join(old_dir, "data.pkl"),
            "sampler_path": os.path.join(old_dir, "sampler.h5"),
            "figure_dir": os.path.join(old_dir, "figures"),
        }

        for name, path in derived_paths.items():
            current = self.__dict__.get(name, None)
            if current is None or (
                name in old_paths
                and os.path.abspath(current) == os.path.abspath(old_paths[name])
            ):
                if name == "figure_dir":
                    path = utils.ensure_directory(path, "figure directory")
                super().__setattr__(name, path)

    @property
    def verbose(self) -> IntLike:
        """The stored global verbosity setting for ACID. See :py:class:`Acid` for more details on how this is used in ACID."""
        return self.__dict__.get("_verbose", self.defaults["verbose"])

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
        return MaskingLines(self.__dict__.get("_masking_lines", self.defaults["masking_lines"]))

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
        utils.show_or_save(plt, self.figure_dir, "masking_lines.png", self.verbose)

    @classmethod
    def print_defaults(cls) -> None:
        """Print the default configuration settings for ACID."""
        print("Default configuration:")
        for k, v in cls.defaults.items():
            print(f"{k}: {v}")
