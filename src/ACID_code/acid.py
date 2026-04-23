from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")
import sys, emcee, os, time, inspect, inspect, contextlib
from emcee import EnsembleSampler
import numpy as np
from math import log10, floor
from scipy.interpolate import interp1d
import multiprocessing as mp
from beartype import beartype
from . import utils, mcmc
from .lsd import LSD
from .result import Result
from .data import Data, Config, MaskingLines, LineList, DataList
from .errors import ContinuumError
from .utils import IntLike, Scalar, Array1D, Array2D

@beartype
class Acid:
    """
    Accurate Continuum fItting and Deconvolution (ACID) class. This class contains the ACID method 
    which fits the continuum of spectra and performs Least Squares Deconvolution (LSD) to obtain
    LSD profiles for each spectrum. It also contains many internal methods used within the main ACID 
    method. See Dolan et al (2024) for more details on the ACID method and its applications.
    """

    def __init__(
        self,
        velocities      : Array1D|None                                       = None,   # Data
        linelist        : Array2D|str|LineList|dict|None                     = None,   # Data
        order           : IntLike|None                                       = None,   # Config
        order_range     : Array1D|None                                       = None,   # Config
        verbose         : IntLike|bool|str|None                              = None,   # Config
        masking_lines   : dict|MaskingLines|None                             = None,   # Config
        seed            : IntLike|None                                       = None,   # Config
        data            : Data|DataList|None                                 = None,   # Data
        config          : Config|None                                        = None,   # Config
        **kwargs,
        ) -> None:
        """
        Notes
        -----
        Initialises the Acid class with inputted parameters. The class keeps calculations stored in the :py:class:`Data` class and run configurations
        in the :py:class:`Config` class (stored in Data for convenience). Both :py:class:`Data` and the :py:class:`Result` class (passed after ACID) have save and load 
        methods which can save their state, with the :py:class:`Result` class handling saving the :py:class:`Data` class together, see :ref:`result`.
        
        As of 2.0, ACID is now designed to be run on one order at a time, for running and keeping track of multiple orders, please see the :py:class:`DataList` class for a natural
        implementation of running ACID on multiple orders and keeping track of which orders have been run and which haven't, as well as storing 
        the results for each order. The :py:class:`DataList` class has been designed with parallelization on HPC's in mind, allowing orders (which are
        independent) to be run by different jobs. See also the :ref:`multiprocessing` and :ref:`datalists` sections.

        Important note: All defaults in the signature are None, meaning if any values are input, they will override the default :py:class:`Config` and/or :py:class:`Data` values or
        any values that have already been input. The defaults within the config are written below. The config defaults can also be accessed via 
        :py:attr:`ACID_code.Config.defaults` (returning a dictionary of defaults for both initialisation and run_acid).

        All parameters below and in run_ACID are stored in the :py:class:`Config` instance, unless explicitly stated to be in the :py:class:`Data` instance.
        The :py:class:`Config` instance is for runtime settings and the :py:class:`Data` instance is for storing data and any calculations. 

        Parameters
        ----------
        velocities : :py:type:`Array1D`, optional
            Velocity grid for LSD profiles (in km/s). For example, use: np.arange(-25, 25, 0.82) to create one. If None, a default grid
            from -25 to 25 km/s is used with a spacing calculated by calc_deltav after the wavelengths are provided. It is highly recommended to 
            choose your own velocity grid, by default None, stored in the Data instance.
        linelist : :py:type:`Array2D | str` | :py:class:`LineList` | dict`, optional
            The linelist to use for LSD. The linelist should have wavelengths in angstroms and relative depths between 0 and 1.
            This is a required parameter if linelist_wl and linelist_depths are not provided. It can be of the forms:
            - String: A path to a VALD linelist in string format. Support for other linelists may be added in the future or on request.
            - :py:type:`Array2D`: A 2D array-like object indexed such that 0 is wavelengths and 1 is depths.
            - dict: A dictionary with keys "wavelengths" and "depths", each containing array-like objects for the wavelengths and depths respectively.
            - :py:class:`LineList`: The :py:class:`LineList` class is used to expose the linelist for masking or getting/plotting the linelist. You can input an instance if you have one.
            - If None, linelist_wl and linelist_depths must be provided (see below), by default None, stored in the Data instance.
        order : :py:type:`IntLike`, optional
            If this ACID instance is intended as a run on a specific order, then you can designate this instance for that order. This will allow
            the resulting Data instance to track of which order the profiles correspond to. Note that orders can be indexed by the correct indexing
            of the spectrograph (ie. some spectrographs start at order ~20). By default 0.
        order_range : :py:type:`Array1D`, optional
            Optionally also give ACID the full order range of the spectograph for the observation. ACID only ever runs on one order at a time,
            but this will allows ACID and eventually the DataList to keep track of which orders have been run and which haven't, and will be 
            used in the future for plotting and saving results. As with order (above), the orders can be indexed to the spectrograph orders. 
            By default [0]
        verbose : :py:type:`bool | IntLike | str`, optional
            The verbosity for printing and plotting the progress and warnings of ACID. The verbosities are natively stored as integers corresponding to:
            0: No printing or plotting, all warnings are ignored.
            1: Only printing warnings.
            2: Printing progress and warnings.
            3: Printing progress and warnings, as well as additional plots and helpful information about the run.
            The possible input types are described below:
            - Integer: Must be between 0 and 3, corresponding to the verbosities described above.
            - Boolean: If True, defaults to 2. If False, defaults to 0.
            - String: Can be one of ["none", "low", "medium", "high"] or their common variants.
        masking_lines : :py:type:`dict` | :py:class:`MaskingLines`, optional
            Telluric lines (in angstroms) and widths in (km/s) to mask from the wavelength regions from. Unless you'd like to change the default masking
            lines, we recommend just using the defaults (leaving this as None), which are based on telluric lines and strong hydrogen/metal lines in the 
            optical and near infrared. For a guide on using your own/modifying the defaults, see :ref:`masking_lines`. By default None, stored in the Config instance.
        seed : :py:type:`IntLike`, optional
            Random seed for reproducibility, leave it on None for a random seed, by default None.
        data : :py:class:`Data` | :py:class:`DataList`, optional
            An optional backend :py:class:`Data` object to use for storing data. Allows previously calculated results to be used skipped.
            If None, a new :py:class:`Data` object is created. Please note that if the :py:class:`Data` class already has a saved ACID config
            class, then any inputs to the :py:class:`Acid` initialisation and ACID method will overwrite these config values. If a 
            :py:class:`DataList` instance is inputted, the :py:class:`Data` instance corresponding to the inputted order is used.
        config : :py:class:`Config`, optional
            An optional :py:class:`Config` object to use for storing the configuration. Allows you to override the config values stored in 
            the :py:class:`Data` object, otherwise, inputs to the initialisation here and the ACID method will overwrite these config values again (if entered).
            If None, an empty Config is created and stored in the Data instance.
        **kwargs : :py:type:`dict`, optional
            Unused except to catch if users use the "linelist_path" input rather than the now "linelist" input.

        Raises
        ------
        BeartypeError
            See :ref:`type_validation` to understand input validation errors.
        """
        # Initialise the data class to store calculations in ACID        
        if data is not None:
            if isinstance(data, DataList):
                data = data[order]
            else:
                self.data = data
        else:
            self.data = Data()

        # If a config is inputted, this will overwrite any config values already in the data class, 
        # otherwise, the config values in the data class will be used and updated by any inputs to the init or ACID method. 
        if config is not None:
            self.data.config = config

        self.config = self.data.config # Either was None (on Data initialisation above) or had a previous config stored in the old or
        # overwritten Data class

        # Validate velocities input, if None, this is handled in ACID function later when a input spectrum is provided
        if velocities is not None:
            if velocities.ndim != 1:
                raise ValueError("'velocities' must be a one-dimensional array")
        # data.velocities defaults to None in Data class, will be set in ACID function from wavelengths if not provided
        self.data.velocities = velocities if velocities is not None else self.data.velocities

        # Verbosity validation handled in config property setter
        self.config.verbose = verbose

        # Catch for the linelist_path, linelist_wl, or linelist_depths arguments, which was old way to input a linelist
        if "linelist_path" in kwargs:
            linelist = kwargs.pop("linelist_path")
            if self.config.verbose > 0:
                print("Warning: 'linelist_path' is a legacy argument for inputting a linelist, " \
                f"please use 'linelist' instead.\n The 'linelist_path' argument does not support full input validation.")
        if "linelist_wl" in kwargs or "linelist_depths" in kwargs:
            raise ValueError("The 'linelist_wl' and 'linelist_depths' arguments are legacy arguments for inputting a linelist, " \
                             "please use 'linelist' instead.")
        # Anything left in kwargs is invalid
        if kwargs:
            raise TypeError(
                f"Unexpected keyword argument(s) for Acid.__init__: {', '.join(sorted(kwargs))}"
            )

        # Set linelist in the Data class, the property setter handles input validation
        self.data.linelist = linelist

        # Set the lines to mask, the property in the class handles input validation and None check
        self.config.masking_lines = masking_lines

        # Set seed if not already done in config, in this way, seed is only explicitly set once
        if getattr(self.config, "seed", None) is None:
            self.config.seed = seed
            if self.config.seed is not None:
                np.random.seed(self.config.seed) # In principle this is only ever called once
            # else: user may define a seed at the top of their seed, so can use that
        # else: seed already in config, so seed would already have been set when put in

        # Default order range for ACID, can be updated in ACID_HARPS. Eventually will add option to add this to inputs
        self.config.order_range = order_range if order_range is not None else self.config.order_range
        self.config.order = order if order is not None else self.config.order

        # Save config to data class
        self.data.config = self.config

        # Determine if running in SLURM environment, independent of any previous configs
        self.slurm = "SLURM_JOB_ID" in os.environ

        self.sampler = None # sampler is a uniquely ACID attribute, so set here as needed in Results class

        return

    # Get init keys to be checked in ACID function for any potential conflicts in input arguments.
    # This is to avoid confusion for users who may accidentally input an argument that is meant for
    # the class initialisation rather than the ACID function, which takes different arguments.
    _INIT_KEYS = set(inspect.signature(__init__).parameters) - {"self", "kwargs"}

    def ACID(
        self,
        wavelengths           : Array1D|Array2D|None        = None,   # Data
        flux                  : Array1D|Array2D|None        = None,   # Data
        errors                : Array1D|Array2D|None        = None,   # Data
        sn                    : Array1D|Array2D|Scalar|None = None,   # Data
        deterministic_profile : bool|None                   = None,   # Config
        poly_ord              : IntLike|None                = None,   # Config
        continuum_percentile  : IntLike|None                = None,   # Config
        bin_size              : IntLike|None                = None,   # Config
        pix_chunk             : IntLike|None                = None,   # Config
        dev_perc              : IntLike|None                = None,   # Config
        n_sig                 : IntLike|None                = None,   # Config
        skips                 : IntLike|None                = None,   # Config
        parallel              : bool|None                   = None,   # Config
        cores                 : IntLike|None                = None,   # Config
        nwalkers              : IntLike|None                = None,   # Config, then Data just before MCMC
        nsteps                : IntLike|None                = None,   # Config as the initial steps, Data.nsteps is the true count of steps taken, which can be higher
        max_steps             : IntLike|None                = None,   # Config
        check_interval        : IntLike|None                = None,   # Config
        min_checks            : IntLike|None                = None,   # Config
        min_tau_factor        : IntLike|None                = None,   # Config
        tau_tol               : float|None                  = None,   # Config
        moves                 : list|None                   = None,   # Config
        run_mcmc              : bool|None                   = True,   # Config
        _all_frames                                         = None,   # To work with legacy code, not to be used, silently ignored
        **kwargs,
        ) -> Result | None:
        """
        Notes
        -----
        Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra,
        returning an LSD profile for each spectrum given. Spectra must cover a similiar wavelength range.

        Important note: All defaults in the signature are None, meaning if any values are input, they will override the default Config and/or Data values or
        any values that have already been input. The defaults within the config are written below. The config defaults can also be accessed by:
        ACID_code.Config.defaults (returning a dictionary of defaults for both initialisation and the ACID method).

        All parameters below are stored in the :py:class:`Config` instance, unless explicitly stated to be in the :py:class:`Data` instance.
        The :py:class:`Config` instance is for runtime settings and the :py:class:`Data` instance is for storing data and any calculations. 

        Parameters
        ----------
        wavelengths : :py:type:`Array1D | Array2D`, optional
            An array of wavelengths for each frame (in Angstroms). For multiple frames this should be a 2D array such that
            wavelengths[i] corresponds to the wavelengths for the ith frame. Can only be None if a data instance was 
            provided in initialisation. If a 2D array is provided, they are treated as multiple frames (not orders), by default None, stored in the Data instance.
        flux : :py:type:`Array1D | Array2D`, optional
            An array of spectral frames (in flux). For multiple frames this should be a 2D array such that 
            flux[i] corresponds to the spectral fluxes for the ith frame. Can only be None if a data instance was 
            provided in initialisation. If a 2D array is provided, they are treated as multiple frames (not orders), by default None, stored in the Data instance.
        errors : :py:type:`Array1D | Array2D`, optional
            Errors for each frame (in flux). For multiple frames this should be a 2D array such that
            errors[i] corresponds to the spectral errors for the ith frame. If a 2D array is provided, they are treated as multiple frames (not 
            orders). If no errors are provided, but the SN is provided, the errors will be estimated from the flux and SN, but we highly recommend 
            providing errors if possible, by default None, stored in the Data instance.
        sn : :py:type:`Scalar | IntLike | Array1D`, optional
            Average signal-to-noise ratio for each frame (used to calculate minimum line depth to consider from line list).
            Each frame should have only one S/N value, so for multiple frames this should be a 1D array such that
            sn[i] corresponds to the S/N for the ith frame. If you prefer to use a per-pixel SN value, ACID will use the :py:function:`utils.collapse_SNR` 
            function to calculate a single S/N value for each frame from the central 2/3rds of the input spectra. In which case, a 2D array can be 
            If None, the S/N will be estimated from the input spectra and errors, by default None, stored in the Data instance.
        deterministic_profile : bool, optional
            If True, fits both the continuum and the LSD profile simultaneously. If False, only fits the continuum in mcmc, the
            profile is inferred from the continuum fit. This is a new feature that has been set to the default as it significantly
            decrease convergence time and computation time per step, while fully maintaining accuracy. Setting this to False will 
            match legacy behaviour, by default True.
        poly_ord : :py:type:`IntLike`, optional
            Order of polynomial to fit as the continuum, by default 3
        continuum_percentile : :py:type:`IntLike`, optional
            The percentile to use when fitting the continuum, by default 90. For example, if 90, the continuum fit will be performed
            on the points in the spectra that are above the 90th percentile in flux in each spectral bin (determined by bin_size below).
        bin_size : :py:type:`IntLike`, optional
            The size of bins to use when performing the continuum fit. The spectra are split into bins with this number of pixels, and 
            the continuum is fit to the median wavelength and the specified percentile of flux in each bin. By default 100 pixels.
        pix_chunk : :py:type:`IntLike`, optional
            Size of 'bad' regions in pixels. 'bad' areas are identified by the residuals between an inital model
            and the data. If the residuals deviate by a specified percentage (see dev_perc below) for this number (pix_chunk) of pixels,
            then this chunk of pixels are masked in the spectra. By default 20
        dev_perc : :py:type:`IntLike`, optional
            Allowed deviation percentage. 'bad' areas are identified by the residuals between an inital model
            and the data. If a residual deviates by this percentage for a specified number of pixels,
            then this chunk of pixels are masked in the spectra. By default 25
        n_sig : :py:type:`IntLike`, optional
            Number of sigma to keep in sigma clipping. Ill fitting lines are identified by sigma-clipping the
            residuals between an inital model and the data. Regions that lie outside the median +- n_sig STDEVs are clipped.
            The clipped regions will be masked in the spectra. This masking is only applied to find the continuum fit and is removed when
            LSD is applied to obtain the final profiles, by default 3
        skips : :py:type:`IntLike`, optional
            An option to only run acid on one in every n pixels, where n is the integer argument. This is only useful for
            testing to get a quicker result especially for larger wavelength ranges or datasets, by default 1 (no skipping)
        parallel : :py:type:`bool`, optional
            If True uses multiprocessing to calculate the profiles for each frame in parallel, see
            https://acid-code.readthedocs.io/en/stable/using_ACID.html#multiprocessing for more details. By default True
        cores : :py:type:`IntLike`, optional
            Number of cores to use if parallel=True. If None, all available cores will be used, by default None
        nwalkers : :py:type:`IntLike`, optional
            A manual override for the number of walkers for the MCMC sampler. By default, uses the emcee recommendation which is 3 times the number of
            dimensions. For the deterministic model, this is just the poly_ord + 1, for the non-deterministic model, it is poly_ord + 1 + nvelocity points.
        nsteps : :py:type:`IntLike`, optional
            Number of steps for the MCMC to run, by default 10000, the initial steps are stored in the config as nsteps, 
            but the true count of steps taken is stored in the Data instance as Data.nsteps, which can be higher than this if 
            continue_sampling is used to continue sampling after the initial run.
        max_steps : :py:type:`IntLike`, optional
            If set, the sampler will run until max_steps or convergence is reached by estimation using the emcee autocorrelation 
            time (tau). The sampler will check for convergence every 'check_interval' steps, and will require a minimum number 
            of checks ('min_checks') and a minimum tau factor ('min_tau_factor') before it can stop. The stopping criterion 
            is met when the change in tau is less than 'tau_tol' for all parameters. By default None, which means no maximum. 
            If a value is inputted, the nsteps parameter is ignored. The continue_sampling method in Result or Acid can still
            be used normally to continue sampling after either stopping criterion is reached.
        check_interval : :py:type:`IntLike`, optional
            Interval (in steps) at which to check for MCMC convergence if max_steps is set, by default 1000. 
            Only used if max_steps is set.
        min_checks : :py:type:`IntLike`, optional
            Minimum number of checks before MCMC can be stopped, by default 1. Only used if max_steps is set.
        min_tau_factor : :py:type:`IntLike`, optional
            Minimum tau factor for MCMC stopping criterion, by default 50, which is the emcee recommendation, it's not
            recommend to set a value below 50 unless you want to force convergence for the deterministic_profile=False option.
            Only used if max_steps is set.
        tau_tol : :py:type:`float`, optional
            Tolerance for tau convergence in MCMC stopping criterion, by default 0.1. Only used if max_steps is set.
        moves : list[tuple], optional
            A list of tuples specifying the moves for the MCMC sampler. The format
            tries to follow the emcee documentation as closely as possible.
            However, the config cannot store classes directly, so move names are
            used instead and converted when building the sampler.

            Each tuple should have the form::

                (move_name: str, fraction: float, move_kwargs: dict | None)

            where:

            - "move_name" is the name of the emcee move. Supported variants currently
              include "RedBlueMove", "StretchMove", "WalkMove",
              "KDEMove", "DEMove", "DESnookerMove", "MHMove",
              and "GaussianMove". Refer to the emcee documentation for more
              details on each move type. Input move names are checked against the
              "emcee.moves" module, so other moves from that module will work.
            - "fraction" is the fraction of walkers to which this move should be applied.
            - "move_kwargs" is an optional dictionary of keyword arguments passed to
              the move class initialisation.
        run_mcmc : :py:type:`bool`, optional
            If True, runs the MCMC to fit the model, by default True. Can be set to False to perform all of the preparation
            for MCMC without actually running it. The ACID function will still update the class and data attributes.
            If True, the method returns a :py:class:`Result` object, and if False, the method returns None, but attributes are updated.
        **kwargs : :py:type:`dict`, optional
            Unused except to catch accidental inputs of initialisation arguments into the ACID method and warn if so.

        Returns
        -------
        Result | None
            A :py:class:`Result` object containing the LSD profiles and associated data.
            See the :py:class:`Result` class for available methods and attributes.

            If "run_mcmc" is False, "None" is returned, but the class
            attributes are still updated.

        Raises
        ------
        BeartypeError
                See :ref:`type_validation` to understand input validation errors.
        ValueError
            If other input arguments do not conform to the expected formats and requirements.
        """
        init_t0 = time.time()
        if self.config.verbose>1:
            print('Initialising...')
        # Setup and data validation done in data class and applies skips
        self.data.set_inputs(wavelengths, flux, errors, sn, skips)

        # Check for any potential conflicts in input arguments that are meant for the class initialisation.
        overlap = self._INIT_KEYS & kwargs.keys()
        if overlap and self.config.verbose > 0:
            for key in sorted(overlap):
                print(f"'{key}' is set in Acid initialisation, not the ACID method. The inputted value will be ignored.")

        # Raise an error if the kwargs are not part of the ACID init
        invalid_keys = set(kwargs.keys()) - self._INIT_KEYS
        if invalid_keys:
            raise ValueError(
                f"The following keyword arguments are not recognised by ACID or Acid.__init__: "
                f"{', '.join(sorted(invalid_keys))}."
            )

        # Assign inputted configuration to config dictionary plus or minus a few variables
        ACID_config = {
            "poly_ord"              : poly_ord,
            "continuum_percentile"  : continuum_percentile,
            "bin_size"              : bin_size,
            "pix_chunk"             : pix_chunk,
            "dev_perc"              : dev_perc,
            "n_sig"                 : n_sig,
            "parallel"              : parallel,
            "cores"                 : cores,
            "nwalkers"              : nwalkers,
            "deterministic_profile" : deterministic_profile,
            "nsteps"                : nsteps,
            "max_steps"             : max_steps,
            "check_interval"        : check_interval,
            "min_checks"            : min_checks,
            "min_tau_factor"        : min_tau_factor,
            "tau_tol"               : tau_tol,
            "moves"                 : moves,
            "run_mcmc"              : run_mcmc,
        }

        # Update config if any of the above config settings are new
        self.config.update_hipri(**ACID_config) # self.config overwrites ACID_config if overlapping
        self.data.config = self.config # update dataclass config as well, although I think this line is redundant

        if self.config.parallel and sys.platform == "win32":
            if self.config.verbose > 0:
                # This doesn't work, needs serious modifications to make work, so just run serially for now
                print("Parallel MCMC on Windows is not currently supported. Running MCMC serially.")
            self.config.parallel = False

        # Now that the data is set, we can check if the velocities were set in the initialisation or not, and if not,
        # calculate a default velocity grid using the input wavelengths.
        if self.data.velocities is None:
            if self.config.verbose > 0:
                print("Velocity grid not input, using a grid calculated from input wavelengths with default range of -25 to 25 km/s.\n " \
                "It is highly recommended to input your own velocity grid, especially if you need a different wavelength range.")
            deltav = utils.calc_deltav(self.data.wavelengths["input"][0])
            self.data.velocities = np.arange(-25, 25 + deltav, deltav) # default velocity grid from -25 to 25 km/s with spacing calculated from input wavelengths

        ### Begin ACID process

        # Combines spectra from each frame (weighted based of S/N), returns to S/N of combined spectra.
        # If only one frame, just uses that frame (handled in the function).
        # This function requires assigned values:
        # self.data.wavelengths["input"], self.data.flux["input"], self.data.errors["input"], self.data.sn["input"]
        # To generate:
        # As of 1.0.4, this generates self.wavelengths["combined"], self.flux["combined"], self.errors["combined"]
        # As of 1.4.0, this now instead goes to the data class, so generates self.data.wavelengths["combined"], etc.
        # As of 1.4.0, this procedure is skipped if the outputs already exists in self.data to avoid recalculation
        if all((
            hasattr(self.data.wavelengths, "combined"),
            hasattr(self.data.flux, "combined"),
            hasattr(self.data.errors, "combined"),
            hasattr(self.data.sn, "combined"),
        )):
            if self.config.verbose > 2:
                print("Combined spectra already exists, skipping combination step.")
        else:
            if self.config.verbose > 2:
                print("Combining spectra...")
            self.combine_spec(output=False)

            # Clean combined spectra of NaNs
            wavelengths, flux, errors, nanmask = utils.drop_invalid(self.data.wavelengths["combined"], self.data.flux["combined"],
                                                                    self.data.errors["combined"], return_mask=True)
            self.data.wavelengths["combined"] = wavelengths
            self.data.flux["combined"] = flux
            self.data.errors["combined"] = errors
            self.data.nanmask = nanmask

        # Perform line masking before initial fit to avoid ill-fitting lines biasing the continuum fit
        # The code for telluric masking is contained without the MaskingLines class, which both telluric_lines
        # and hydrogen_lines are instances of.
        line_mask = self.config.masking_lines.get_masks(self.data.wavelengths["combined"])
        line_mask = np.all(line_mask, axis=0)
        self.data.errors["combined"][line_mask] = 1e12

        # Get the initial polynomial coefficents
        if not hasattr(self.data.wavelengths, "combined_normalized"):
            a, b = utils.get_normalisation_coeffs(self.data.wavelengths["combined"])
            self.data.wavelengths["combined_normalized"] = (self.data.wavelengths["combined"]*a)+b

        # Compute an initial continuum fit
        # poly inputs has polynomial coefficients and scale at the end
        if all((
            hasattr(self.data.flux, "fitted"),
            hasattr(self.data.errors, "fitted"),
            self.data.poly_inputs is not None
        )):
            if self.config.verbose > 2:
                print("Continuum fit already exists, skipping initial fit step.")
        else:
            if self.config.verbose > 2:
                print("Performing initial continuum fit...")
            self.data.poly_inputs, self.data.flux["fitted"], self.data.errors["fitted"] = self.continuumfit(
                self.data.flux["combined"],
                self.data.wavelengths["combined"],
                self.data.errors["combined"],
                poly_ord = self.config.poly_ord,
                plot_result = self.config.verbose > 2,
                plot_type = "initial"
            )
        self.data.wavelengths["fitted"] = np.copy(self.data.wavelengths["combined"]) # Just to keep track
        self.data.sn["fitted"]          = np.copy(self.data.sn["combined"]) # SN also is not changed here

        # Get the initial LSD profile and set the alpha matrix (unchanged from masking) and model_inputs
        if all((
            self.data.model_inputs is not None,
            self.data.alpha is not None
        )):
            if self.config.verbose > 1:
                print("Initial LSD profile already exists, skipping initial LSD step.")
        else:
            if self.config.verbose > 1:
                print("Calculating initial LSD profile...")
            # Get the initial LSD profile using the initial fit
            initial_LSD = LSD(self.data) # Initialise LSD class with standard Acid attributes (verbosity, linelist, velocities, etc)
            initial_LSD.run_LSD(self.data.wavelengths["fitted"], self.data.flux["fitted"], self.data.errors["fitted"], self.data.sn["fitted"])

            # Use alpha matrix and initial profile class variables from initial LSD run
            self.data.initial_profile = initial_LSD.profile # in optical depth
            self.data.initial_profile_errors = initial_LSD.profile_errors # Not used, saved for debugging
            self.data.alpha = initial_LSD.alpha
            # Set x, y, yerr, and model_inputs for emcee
            self.data.model_inputs = np.concatenate((self.data.initial_profile, self.data.poly_inputs))

        # Masking based off residuals
        # Requires: self.x, self.y, self.yerr, self.data.model_inputs, self.poly
        # Sets: self.c_factor, self.residual_masks
        # Modifies: self.yerr, and as of 1.5, self.data.model_inputs
        # As of 1.4.0, this is all applied to the data class (not internal ACID variables)
        if all((
            self.data.residual_masks is not None,
            self.data.c_factor is not None
        )):
            if self.config.verbose > 1:
                print("Residual masks already exists, skipping residual masking step.")
        else:
            if self.config.verbose>1:
                print('Residual masking...')
            self.residual_mask()

        # Setting number of walkers and their start values(pos)
        self.data.ndim = len(self.data.model_inputs)
        # emcee recommendation is 3 times the number of dimensions, but can be overridden by user input
        self.data.nwalkers = self.data.ndim * 3 if self.config.nwalkers is None else self.config.nwalkers
        rng = np.random.default_rng(self.config.seed)

        # Starting values of walkers with independent variation
        sigma = 0.8 * 0.005
        initial_state = []
        for i in range(0, self.data.ndim):
            if i < len(self.data.velocities):
                pos = rng.normal(self.data.model_inputs[i], sigma, (self.data.nwalkers, ))
            else:
                x1 = self.data.model_inputs[i]
                rounded_sigma = round(x1, 1-int(floor(log10(abs(x1))))-1)
                sigma = abs(rounded_sigma) / 10
                pos = rng.normal(self.data.model_inputs[i], sigma, (self.data.nwalkers, ))
            initial_state.append(pos)
        initial_state = np.array(initial_state)

        # Configure the nwalkers for the deterministic option (now default as of 1.4)
        if self.config.deterministic_profile is True:
            self.data.ndim = self.config.poly_ord + 1
            self.data.nwalkers = self.data.ndim * 3 if self.config.nwalkers is None else self.config.nwalkers
            initial_state = initial_state[-self.data.ndim:, :self.data.nwalkers]

        # Transpose initial state to have shape (nwalkers, ndim) for emcee
        self.data.initial_state = initial_state.T # Saved for debugging if needed, otherwise class variable not used for now

        ### ACID initialialised ###
        self.data.initialisation_time += time.time() - init_t0
        mcmc_t0 = time.time()
        if self.config.verbose>1:
            print('Initialised in %ss'%round((self.data.initialisation_time), 3))
        if self.config.verbose>2:
            print('ACID Configuration before MCMC run:')
            print(f"Polynomial order: {self.config.poly_ord}")
            print(f"Deterministic profile: {self.config.deterministic_profile}")
            print(f"Number of walkers: {self.data.nwalkers}")
            print(f"Number of dimensions: {self.data.ndim}")

        # Run MCMC
        if self.config.run_mcmc is True:
            if self.config.max_steps is None:
                if self.config.verbose > 1:
                    print("Running MCMC for %s steps..."%self.config.nsteps)
                self.run_mcmc(self.config.nsteps, initial_state)
                self.data.nsteps += self.config.nsteps
            else:
                if self.config.verbose > 1:
                    print(f"Running MCMC with a maximum of {self.config.max_steps} steps or until convergence is reached...")
                self.run_mcmc_until_converged(self.config.max_steps, initial_state)
                self.data.nsteps = self.step_number
            self.data.mcmc_time += time.time() - mcmc_t0
            return Result(self)

        else:
            if self.config.verbose > 0:
                print("MCMC not run, returning None. Class attributes have been updated.")
            return None

    def ACID_HARPS(self, *args, **kwargs):
        """
        This method is no longer supported in ACID. Please use the ACID function with the appropriate inputs for HARPS spectra instead. 
        Future versions of ACID will provide functions to load and configure data from a range of different standard instruments. 
        If you still really wish to use ACID_HARPS, the last stable version of ACID with the method is 1.4.5. Try: pip install ACID_code==1.4.5
        """
        raise NotImplementedError(f"ACID_HARPS is no longer supported in ACID. \n"
        f"Please use the ACID function with the appropriate inputs for HARPS spectra instead. \n"
        f"Future versions of ACID will provide functions to load and configure data from a range of different standard instruments. \n"
        f"If you still really wish to use ACID_HARPS, the last stable version of ACID with the method is 1.4.5. Try: pip install ACID_code==1.4.5")

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
            self.data.wavelengths["input"] = np.copy(frame_wavelengths)
            self.data.flux["input"]        = np.copy(frame_flux)
            self.data.errors["input"]      = np.copy(frame_errors)
            self.data.sn["input"]          = np.copy(frame_sns)

        # Set simple names for variables (just used in this function)
        wavelengths = np.copy(self.data.wavelengths["input"])
        flux        = np.copy(self.data.flux["input"])
        errors      = np.copy(self.data.errors["input"])
        sn          = np.copy(self.data.sn["input"])

        # Return as is if only one spectrum
        if len(self.data.wavelengths["input"])==1:
            self.data.wavelengths["combined"] = np.copy(self.data.wavelengths["input"][0])
            self.data.flux["combined"]        = np.copy(self.data.flux["input"][0])
            self.data.errors["combined"]      = np.copy(self.data.errors["input"][0])
            self.data.sn["combined"]          = np.copy(self.data.sn["input"][0])

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

            self.data.wavelengths["combined"] = combined_wavelengths
            self.data.flux["combined"]        = combined_flux
            self.data.errors["combined"]      = combined_errors
            self.data.sn["combined"]          = combined_sn

        if output is True:
            # ie if called as a function rather than from ACID function
            return combined_wavelengths, combined_flux, combined_errors, combined_sn

    def continuumfit(
        self,
        fluxes      : Array1D,
        wavelengths : Array1D,
        errors      : Array1D,
        poly_ord    : IntLike = 3,
        plot_result : bool    = False,
        plot_type   : str     = "initial"
        ) -> tuple:
        """Provides an initial, normalised continuum fit using inputted spectra.

        Parameters
        ----------
        fluxes : np.ndarray
            The flux values of the spectrum.
        wavelengths : np.ndarray
            The wavelengths corresponding to the spectrum.
        errors : np.ndarray
            The error values associated with the spectrum.
        poly_ord : int
            The order of the polynomial to fit to the continuum. By default 3.
        plot_result : bool, optional
            Whether to plot the continuum fit result, by default False.
        plot_type : str, optional
            The type of plot to generate, either "initial" or "masked", by default "initial"

        Returns
        -------
        tuple containing:
            - Polynomial coefficients: The coefficients of the fitted polynomial, ordered from highest degree to lowest.
            - Normalized flux: The flux values normalized by the fitted continuum.
            - Normalized errors: The error values normalized by the fitted continuum.
        """
        # Normalise wavelengths
        a, b = utils.get_normalisation_coeffs(wavelengths)
        unnormalized_wavelengths = np.copy(wavelengths)
        wavelengths = (wavelengths*a)+b

        # Sort to ensure smooth binning and fitting
        idx = np.argsort(wavelengths)
        w = wavelengths[idx]
        f = fluxes[idx]
        e = errors[idx]

        # Get nbins and bin_size, reshape into 2D array of bins
        binsize = self.config.bin_size
        n = len(w) // binsize  # full bins only
        w2 = w[:n*binsize].reshape(n, binsize)
        f2 = f[:n*binsize].reshape(n, binsize)
        e2 = e[:n*binsize].reshape(n, binsize)

        # Get the median wavelength, specified percentile flux, and median error in each bin
        clipped_flux = np.nanpercentile(f2, self.config.continuum_percentile, axis=1)
        clipped_waves = np.nanmedian(w2, axis=1)
        clipped_errs = np.nanmedian(e2, axis=1)

        # Remove bad points for the polynomial fit, defined as non-finite values or errors that are non-positive or above 1e11
        good = (
            np.isfinite(clipped_waves)
            & np.isfinite(clipped_flux)
            & np.isfinite(clipped_errs)
            & (clipped_errs > 0)
            & (clipped_errs < 1e11) # 1e12 is the default mask error value, which can be picked up in the median error binning
        )

        # Fit with np.polyfit
        coeffs = np.polyfit(clipped_waves[good], clipped_flux[good], poly_ord, w=1/clipped_errs[good])
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)

        # Get the model fitted flux and errors from the fit
        flux_obs = fluxes / fit
        new_errors = errors / fit

        # Flip coefficients for use in MCMC (ordered from lowest degree to highest)
        poly_coeffs = np.flip(coeffs)

        # Save to Data the required variables for the plot
        if plot_type not in self.data.plotting_variables:
            self.data.plotting_variables[plot_type] = {}
        self.data.plotting_variables[plot_type]["unnormalized_wavelengths"] = unnormalized_wavelengths
        self.data.plotting_variables[plot_type]["fluxes"]                   = fluxes
        self.data.plotting_variables[plot_type]["fit"]                      = fit
        self.data.plotting_variables[plot_type]["clipped_waves"]            = clipped_waves
        self.data.plotting_variables[plot_type]["clipped_flux"]             = clipped_flux
        self.data.plotting_variables[plot_type]["good"]                     = good
        if plot_result is True:
            self.data.plot_continuum_fit(plot_type=plot_type)

        if np.any(flux_obs <= 0) or np.any(new_errors <= 0):
            raise ContinuumError("Continuum fit resulted in non-positive flux or errors, which is not physical.\n " \
            "Consider adjusting the polynomial order or continuum percentile. Use verbose=3 to see the plot of the continuum fit.\n " \
            "Note that this will only work for interactive terminals or displays which work with plt.show()")

        return poly_coeffs, flux_obs, new_errors

    def residual_mask(self) -> None:
        """
        Masks regions of the spectrum based on residuals from an initial model fit. A purely class method not to be used elsewhere.
        This function is really only supposed to be used in the class, so no inputs are accepted. It is only used once in ACID 
        and could be put directly in the method, but this allows a clearer checkpoint which segments saving the result of the mask for analysis.
        """

        # Residual masking - mask continuous areas first - then possibly progress to masking the narrow lines

        # Set standard variables
        x = self.data.wavelengths["combined"]
        y = self.data.flux["combined"]
        yerr = self.data.errors["combined"]
        sn = self.data.sn["combined"]

        # Use the initial LSD run to get the forward model and scaled residuals
        forward, _profile = mcmc.MCMC(x, y, yerr, self.data.alpha).full_model(self.data.model_inputs)
        residuals = (y - forward) / forward

        # Chunk masking based on deviation from residuals
        # -----------------------------------------------

        # Get bad pixels that deviate by a percentage greater than dev_perc
        bad_idx = np.abs(residuals) > (self.config.dev_perc / 100)

        # A trick to get the mask for continous regions of bad pixels, by padding the bad_idx 
        # with False on both sides and finding the start and end indices of the True regions
        padded = np.concatenate(([False], bad_idx, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        pix_mask = np.zeros_like(residuals, dtype=bool)

        # Then mask those regions that are greater than pix_chunk in length
        for start, end in zip(starts, ends):
            if (end - start) >= self.config.pix_chunk:
                yerr[start:end] = 1e12
                pix_mask[start:end] = True

        # Warn if more than 50% of spectrum is masked this way
        if np.sum(pix_mask) > 0.5 * len(pix_mask):
            if self.config.verbose > 0:
                print(f"Warning: More than 50% of the spectrum is masked based on residuals. \n" \
                "Please check your initial continuum fit (by using verbose=3 when initialising), \n" \
                "or consider adjusting the pix_chunk and dev_perc parameters. If you are aware that you \n" \
                "have bad spectra, then this can be ignored.")

        # Note that this is used to keep track of the residual masks for later use when processing the results
        self.data.residual_masks = tuple([yerr >= 1e12])

        # Sigma clipping
        # --------------

        # Get median, sigma, and clip limits
        m = np.median(residuals)
        sigma = np.std(residuals)
        clip = self.config.n_sig * sigma
        lower_clip = m - clip
        upper_clip = m + clip

        # Find and apply mask
        mask = (residuals <= lower_clip) | (residuals >= upper_clip)
        yerr[mask] = 1e12

        # Now do another continuum fit with masked yerr, continuumfit removes high error points from the fit
        poly_inputs, fitted_flux, fitted_errors  = self.continuumfit(y, x, yerr, self.config.poly_ord,
                                                   plot_result=self.config.verbose > 2,
                                                   plot_type="masked")

        # Run LSD again with the new fitted flux and errors
        LSD_masking = LSD(self.data)
        # Since the above ONLY modifies yerr, and the alpha matrix is independent of yerr, we can input previous 
        # alpha since it wil be the same. We still run LSD to get c_factor and the profile
        # alpha is only dependent on wavelengths and linelist, which are unchanged
        LSD_masking.run_LSD(x, fitted_flux, fitted_errors, sn, alpha=self.data.alpha)

        # Update and set new variables
        self.data.c_factor = LSD_masking.c_factor
        self.data.initial_model_inputs = np.copy(self.data.model_inputs) # Save the initial model inputs before masking for later use if needed
        self.data.model_inputs = np.concatenate((LSD_masking.profile, poly_inputs))

        # Set masked variables
        self.data.wavelengths["masked"] = x
        self.data.flux["masked"]        = y # x and y dont change in this func
        self.data.errors["masked"]      = yerr # yerr is modified in this func
        self.data.sn["masked"]          = np.copy(self.data.sn["combined"]) # SN is not changed in this func
        # self.alpha is also modified in this func to get new alpha with masked residuals using pix chunk and dev perc

        # Set required variables for plotting in the Data class
        if "residual_masking" not in self.data.plotting_variables:
            self.data.plotting_variables["residual_masking"] = {}
        self.data.plotting_variables["residual_masking"]["mask"] = mask
        self.data.plotting_variables["residual_masking"]["residuals"] = residuals
        self.data.plotting_variables["residual_masking"]["upper_clip"] = upper_clip
        self.data.plotting_variables["residual_masking"]["lower_clip"] = lower_clip
        self.data.plotting_variables["residual_masking"]["pix_mask"] = pix_mask
        self.data.plotting_variables["residual_masking"]["profile_F"] = LSD_masking.profile_F
        if self.config.verbose > 2:
            self.data.plot_residual_masking()

        return

    def run_mcmc(
        self,
        nsteps:IntLike,
        state = None,        
        ) -> None:
        """
        Runs MCMC for a specified number of steps. A purely class method that I do not recommend you use directly. Use
        Acid.ACID(run_mcmc=True) to run MCMC for the first pass if not already done, which will skip already performed calculations.
        Otherwise, use Acid.continue_sampling or Result.continue_sampling if you have already run MCMC and want to continue.
        """

        # Get default sampler kwargs from initial state
        sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps, state)

        if self.config.parallel:
            utils.configure_mp_environ(os) # Raises error is not configured correctly, otherwise does nothing

            if self.config.verbose>1:
                print(f"Using {self.config.cores} cores for MCMC")

            # For some reason, unspecified pooling as was before (as in case of windows in the else statement)
            # leds to a hung computer. So specify mp.get_context required, default is spawn, but spawn
            # causes multiple instances of this script to rerun, causing alpha matrix calculation to be redone
            # in each child process. Therefore, fork, which is legacy mp behavior on unix, is used.
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=self.config.cores, initializer=mcmc._mp_init_worker, initargs=(self.data,)) as pool:
                self.sampler = EnsembleSampler(**sampler_kwargs, pool=pool, log_prob_fn=mcmc._mp_log_probability)
                self.sampler.run_mcmc(**mcmc_kwargs)

        else:
            MCMC = mcmc.MCMC(self.data)
            self.sampler = EnsembleSampler(**sampler_kwargs, log_prob_fn=MCMC)
            self.sampler.run_mcmc(**mcmc_kwargs)

    def run_mcmc_until_converged(
        self,
        max_steps:IntLike,
        state=None,
        ) -> None:
        """
        Runs MCMC until convergence is reached. A purely class method that I do not recommend you use directly. Use
        Acid.ACID(run_mcmc=True) to run MCMC for the first pass if not already done, which will skip already performed calculations.
        Otherwise, use Acid.continue_sampling or Result.continue_sampling if you have already run MCMC and want to continue.
        """

        # Get sampler kwargs for the first run based on initial state, then update nsteps in mcmc_kwargs for subsequent runs
        sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps=self.config.check_interval, state=state)

        # Set the stopping arguments to save space
        stopping_criterion_args = (self.config.min_checks, self.config.min_tau_factor, self.config.tau_tol)

        # Set variables to be updated within the convergence loop
        step_number = 0
        tau_list = []
        max_samples = max_steps // self.config.check_interval
        last_tolerance = np.inf
        last_neff = 0

        if self.config.parallel:

            utils.configure_mp_environ(os) # Raises error is not configured correctly, otherwise does nothing, mainly only for HPC environments

            ctx = mp.get_context("fork")
            with ctx.Pool(processes=self.config.cores, initializer=mcmc._mp_init_worker, initargs=(self.data,)) as pool:
                self.sampler = EnsembleSampler(**sampler_kwargs, pool=pool, log_prob_fn=mcmc._mp_log_probability)
                for i in range(max_samples):
                    tol_str, neff_str = mcmc.MCMC._get_tqdm_desc(last_tolerance, last_neff, self.config)
                    desc_dict = {"desc": f"Iteration {i+1}/{max_samples}, last tolerance: {tol_str}, neff: {neff_str}"}
                    mcmc_kwargs["progress_kwargs"] = desc_dict
                    self.sampler.run_mcmc(**mcmc_kwargs, skip_initial_state_check=True)
                    
                    mcmc_kwargs["initial_state"] = None # only use initial state for first run
                    step_number += self.config.check_interval

                    try:
                        # We want to keep the time for get_autocorr_time to run constant, so thin accordingly
                        # It scales with the number of steps, so thin by the number of steps taken divided by 
                        # the check interval to keep the same number of samples for get_autocorr_time to process.
                        with open(os.devnull, "w") as devnull, \
                            contextlib.redirect_stdout(devnull), \
                            contextlib.redirect_stderr(devnull): # Suppresses outputs from get_autocorr_time
                            tau = self.sampler.get_autocorr_time(tol=0, thin=step_number//self.config.check_interval)
                    except emcee.autocorr.AutocorrError:
                        continue

                    tau_list.append(tau)
                    # The stopping criterion function below handles the logic for determining stopping condition
                    condition, last_tolerance, last_neff = mcmc.MCMC._get_mcmc_stopping_criterion(tau_list, step_number, *stopping_criterion_args)
                    if condition is True and self.config.verbose > 1:
                        print(f"Converged at step {step_number}. Final tolerance: {last_tolerance:.4f}, final effective sample size: {last_neff:.2f}.")
                        break
        else:
            # Comments for the non-parallel version are mostly the same as for the parallel version, see the above if confused
            MCMC = mcmc.MCMC(self.data)
            self.sampler = EnsembleSampler(**sampler_kwargs, log_prob_fn=MCMC)

            for i in range(max_samples):
                tol_str, neff_str = mcmc.MCMC._get_tqdm_desc(last_tolerance, last_neff, self.config)
                desc_dict = {"desc": f"Iteration {i+1}/{max_samples}, last tolerance: {tol_str}, neff: {neff_str}"}
                mcmc_kwargs["progress_kwargs"] = desc_dict
                self.sampler.run_mcmc(**mcmc_kwargs, skip_initial_state_check=True)
                mcmc_kwargs["initial_state"] = None
                step_number += self.config.check_interval

                try:
                    with open(os.devnull, "w") as devnull, \
                        contextlib.redirect_stdout(devnull), \
                        contextlib.redirect_stderr(devnull):
                        tau = self.sampler.get_autocorr_time(tol=0, thin=step_number//self.config.check_interval)
                except emcee.autocorr.AutocorrError:
                    continue

                tau_list.append(tau)
                condition, last_tolerance, last_neff = MCMC._get_mcmc_stopping_criterion(tau_list, step_number, *stopping_criterion_args)
                if condition is True and self.config.verbose > 1:
                    print(f"Converged at step {step_number}. Final tolerance: {last_tolerance:.4f}, final effective sample size: {last_neff:.2f}.")
                    break
        
        # Warn if convergence not reached after either parallel or non-parallel version
        if self.config.verbose > 1 and condition is False:
                print(f"Not converged after reaching max steps of {step_number}. Final effective sample size: {last_neff:.2f}, final tolerance: {last_tolerance:.4f}.\n"
                        f"Consider increasing max_steps.")
        
        # Update step_number once mcmc has finished in both cases
        self.step_number = step_number

    def _get_sampler_kwargs(self, nsteps, state=None):
        # Gets sampler kwargs for the emcee EnsembleSampler and run_mcmc functions based on the current state of the
        # ACID instance and the inputted nsteps and state.

        sampler_verbosity = True if self.config.verbose>1 else False
        backend = None
        if state is None:
            if not hasattr(self, 'sampler'):
                raise ValueError("No existing sampler found. Please run 'ACID' first or provide a state.")
            backend = self.sampler.backend # This includes previous seed

        if self.config.cores is None:
            if self.slurm:
                self.config.cores = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
            else:
                self.config.cores = os.cpu_count()

        # Configure moves based on config, this function converts a Config moves dictionary format to a class format
        # accepted for emcee moves.
        moves = utils.convert_moves_to_emcee(self.config.moves)

        sampler_kwargs = {
            "nwalkers"   : self.data.nwalkers,
            "ndim"       : self.data.ndim,
            "moves"      : moves,
            "backend"    : backend,
        }
        mcmc_kwargs = {
            "initial_state": state,
            "nsteps"       : nsteps,
            "progress"     : sampler_verbosity,
            "store"        : True,
            "tune"         : True
        }
        return sampler_kwargs, mcmc_kwargs

    def continue_sampling(
        self,
        sampler,
        nsteps           : IntLike|None = None,
        max_steps        : IntLike|None = None,
        max_steps_kwards : dict|None    = None,
        return_sampler   : bool         = True
        ) -> EnsembleSampler | None:
        """
        Continue MCMC sampling for additional steps. This should be called in Result class by the user.
        This necessarily requires a Data instance to have been put into the ACID init.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            The existing MCMC sampler to continue sampling from.
        nsteps : :py:type:`IntLike`, optional
            Number of additional steps to run the MCMC for.
        max_steps : :py:type:`IntLike`, optional
            Maximum number of steps to run the MCMC for in total (including previous steps).
            If specified, the MCMC will stop if this number of steps is reached even if convergence has not been reached, by default None.
            If input, nsteps is ignored.
        max_steps_kwards : dict, optional
            Additional keyword arguments to be passed to the run_mcmc_until_converged function if max_steps is specified, by default None.
            The kwargs description can be found in Acid.ACID(), they are the 4 kwargs appearing after max_steps. Typos for kwargs are silently
            ignored.
        return_sampler : bool, optional
            Whether to return the sampler after continuing sampling. Default is True.


        Returns
        -------
        emcee.EnsembleSampler | None
            The MCMC sampler after running for the additional steps, or None if return_sampler is False.
        """
        assert self.data.alpha is not None, "Data instance must have alpha matrix calculated to continue sampling."

        self.sampler = sampler
        self.config = self.data.config

        if max_steps is not None:
            if max_steps_kwards is not None:
                self.config.update_hipri(**max_steps_kwards)
            self.run_mcmc_until_converged(max_steps, state=None) # continue from current state
            self.data.nsteps += self.step_number
        else:
            self.run_mcmc(nsteps, state=None) # continue from current state
            self.data.nsteps += nsteps

        if return_sampler:
            return self.sampler

    def get_result(
        self=None,
        ) -> Result:
        """Return a Result object for this instance or one passed explicitly.

        Parameters
        ----------
        self : Acid instance, optional
            The Acid instance to get the Result for. If None, must be called on an instance of Acid.

        Returns
        -------
        Result
            The Result object for the given Acid instance.
        """
        if self is None:
            raise ValueError("Must be called on an instance or passed an instance explicitly")
        return Result(self)

# All code below is just to ensure backward compatibility with previous ACID versions
def ACID(*args, **kwargs):
    """Legacy ACID function

    This function runs the legacy ACID code. This is provided for backwards compatibility with previous versions of ACID.
    It is recommended to use the ACID class and its methods for new code. The args and kwargs passing follows the original
    pre-2.0 version of ACID, which can be found in the earlier releases in https://github.com/ldolan05/ACID

    Parameters
    ----------
    *args
        Positional arguments to be passed to the ACID function.
    **kwargs
        Keyword arguments to be passed to the ACID initialisation and function.

    Returns
    -------
    Any
        Returns the outputs of the ACID function (now a Result object).
    """
    # Use old argument names and map to new ones
    LEGACY_ACID_ARGS = [
        "input_wavelengths",
        "input_spectra",
        "input_spectral_errors",
        "line",
        "frame_sns",
        "vgrid",
        "all_frames",
        "poly_or",
        "pix_chunk",
        "dev_perc",
        "n_sig",
        "telluric_lines",
        "order",
    ]
    RENAMED_LEGACY_ARGS = {
        "input_wavelengths": "wavelengths",
        "input_spectra": "flux",
        "input_spectral_errors": "errors",
        "frame_sns": "sn",
        "vgrid": "velocities",
        "line": "linelist",
        "poly_or": "poly_ord",
        "all_frames": "_all_frames",
    }

    # Split args and kwargs into init and run kwargs using helper function
    init_kwargs, run_kwargs = _get_init_and_run_kwargs(LEGACY_ACID_ARGS, RENAMED_LEGACY_ARGS, *args, **kwargs)

    acid = Acid(**init_kwargs)
    return acid.ACID(**run_kwargs)

def ACID_HARPS(*args, **kwargs):
    """Legacy ACID_HARPS function, deprecated after 1.4.5.
    """
    raise NotImplementedError(f"ACID_HARPS is no longer supported. \n"
        f"Please use the ACID function with the appropriate inputs for HARPS spectra instead. \n"
        f"Future versions of ACID will provide functions to load and configure data from a range of different standard instruments. \n"
        f"If you still really wish to use ACID_HARPS, the last stable version of ACID with the method is 1.4.5. Try: pip install ACID_code==1.4.5")

def _get_init_and_run_kwargs(legacy_args, renamed_args_map, *args, **kwargs):
    """Helper function to split legacy args and kwargs into init and run kwargs given
    legacy argument names and their renamed counterparts.
    """
    legacy_kwargs = {}

    # Check for too many positional arguments
    if len(args) > len(legacy_args):
        raise TypeError(f"Too many positional arguments: {len(args)}")

    # Map positional arguments to their legacy names
    for i, val in enumerate(args):
        legacy_kwargs[legacy_args[i]] = val
    
    # Map legacy argument names to new ones
    translated_legacy = {}
    for key, val in legacy_kwargs.items():
        new_key = renamed_args_map.get(key, key)
        translated_legacy[new_key] = val
    translated_kwargs = {}
    for key, val in kwargs.items():
        new_key = renamed_args_map.get(key, key)
        translated_kwargs[new_key] = val

    # Combine both translated dictionaries
    combined = {**translated_legacy, **translated_kwargs}

    # Determine which arguments are for __init__ and which are for run_ACID_HARPS
    init_params = inspect.signature(Acid.__init__).parameters
    init_keys = set(init_params.keys()) - {"self"}

    # Split kwargs accordingly
    init_kwargs = {key: val for key, val in combined.items() if key in init_keys}
    run_kwargs = {key: val for key, val in combined.items() if key not in init_keys}
    return init_kwargs, run_kwargs