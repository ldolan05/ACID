import warnings
warnings.filterwarnings("ignore")
import sys, emcee, os, time, inspect, inspect
import numpy as np
from math import log10, floor
from scipy.interpolate import interp1d
import multiprocessing as mp
from beartype import beartype
from . import utils
from .lsd import LSD
from . import mcmc
from .result import Result
from .data import Data, Config, MaskingLines, LineList
from .data import DataList
from .utils import IntLike, Scalar, Array1D, Array2D

@beartype
class Acid:
    """Accurate Continuum fItting and Deconvolution (ACID) class. This class contains the ACID method 
    which fits the continuum of spectra and performs Least Squares Deconvolution (LSD) to obtain
    LSD profiles for each spectrum. It also contains many internal methods used within the main ACID 
    function. See Dolan et al (2024) for more details on the ACID method and its applications."""

    def __init__(
        self,
        velocities      : Array1D|None                                  = None,   # Data
        linelist_path   : Array2D|str|LineList|dict                     = None,   # Data
        linelist_wl     : Array1D|None                                  = None,   # Data
        linelist_depths : Array1D|None                                  = None,   # Data
        order           : IntLike                                       = None,   # Config
        order_range     : Array1D                                       = None,   # Config
        verbose         : IntLike|bool|str                              = None,   # Config
        telluric_lines  : Array1D|Array2D|dict|MaskingLines|list[tuple] = None,   # Config
        telluric_widths : Scalar                                        = None,   # Config
        hydrogen_lines  : Array1D|Array2D|dict|MaskingLines|list[tuple] = None,   # Config
        hydrogen_widths : Scalar                                        = None,   # Config
        seed            : IntLike                                       = None,   # Config
        data            : Data|DataList                                 = None,   # Data
        config          : Config                                        = None,   # Config
        ):
        """Initialises the Acid class with inputted parameters. The class keeps calculations stored in the Data class and run configurations
        in the config class (stored in Data for convenience). Both Data and the Result class (passed after run_ACID) have save and load 
        methods which can save the result of any calculations, with the Result class naturally saving the Data class together. ACID is designed
        now to be run on only one order at a time, for running and keeping track of multiple orders, please see the DataList class for a natural
        implementation of running ACID on multiple orders and keeping track of which orders have been run and which haven't, as well as storing 
        the results for each order. The DataList instance has been designed with parallelization on HPC's in mind, allowing orders (which are
        independent) to be run by different jobs. See also the multiprocessing section the readthedocs
        (https://acid-v2.readthedocs.io/en/latest/using_ACID.html#multiprocessing).

        Important note: All defaults in the code are None, meaning if any values are input, they will override the default Config and/or Data values or
        any values that have already been input. The defaults within the config are written below. The config defaults can also be accessed by:
        ACID_code_v2.Config.defaults (returning a dictionary of defaults for both initialisation and run_acid)

        Parameters
        ----------
        velocities : np.ndarray | None, optional
            Velocity grid for LSD profiles (in km/s). For example, use: np.arange(-25, 25, 0.82) to create. If None, a default grid
            from -25 to 25 km/s with a spacing calculated by calc_deltav. It is highly recommended to choose your own velocity grid, 
            by default None
        linelist_path : str | None, optional
            Can be a path to linelist in string format, a dictionary with keys "wavelengths" and "depths", a LineList class, or a 
            list/array indexed such that 0 is the wavelengths and 1 is the depths. If None, you can directly provide linelist_wl
            and linelist_depths instead. At least one of linelist_path or linelist_wl and linelist_depths must be provided. By default None.
        linelist_wl : np.ndarray | list | None, optional
            Wavelengths of lines in linelist (in Angstroms). Only necessary if linelist_path is not provided. 
            Must be same length as linelist_depths. If None, linelist_path must be provided., by default None
        linelist_depths : np.ndarray | list | None, optional
            Depths of lines in linelist (between 0 and 1). Only necessary if linelist_path is not provided. 
            Must be same length as linelist_wl. If None, linelist_path must be provided., by default None
        order : int, optional
            If this ACID instance is intended as a run on a specific order, then you can designate this instance to that order. This will allow
            the resulting Data instance to track which order the profiles correspond to. Note that orders can be indexed by the correct indexing
            of the spectrograph (ie. some spectrographs start at order ~20). By default 0.
        order_range : np.ndarray | list, optional
            Optionally also give ACID the full order range of the spectograph for the observation. ACID only ever runs on one order at a time,
            but this will allows ACID and eventually the DataList to keep track of which orders have been run and which haven't, and will be 
            used in the future for plotting and saving results. As with order, the orders can be indexed to the spectrograph orders.
        verbose : bool | int | None, optional
            An integer between 0 and 3. If 0, nothing is printed. If 2, prints out useful progress information, as well as ACID warnings 
            about any potential issues with the input data or autocorrelation warnings. If True, defaults to 2. If False, defaults to 0.
            If you want to ignore the warnings but still keep progress information, set verbose to 1. A verbosity of 3 will produce 
            additional plots, such as the result of the continuum fit. By default None, which defaults to 2 (True). If set, overrides 
            any verbose setting in the dataclass.
        telluric_lines : np.ndarray | list | None, optional
            List of wavelengths (in Angstroms) of telluric lines to be masked. This can also include problematic
            lines/features that should be masked also. For each wavelengths in the list ~3Å eith side of the line is masked.
            By default None. You can also put a MaskingLines class or dictionary with keys "lines" and "widths" for the telluric lines,
            where "lines" (required) is the same list of wavelengths above and "widths" (optional, default None) is a list of the widths of said lines.
            They can be telluric, or a list of strong Hydrogen lines (as included in the default telluric_lines). If widths are not provided,
            a default width of 21 km/s is used for all lines, which is the typical width of telluric lines. If widths are provided, the width of
            each line is taken to be the inputted value in km/s. When masking H lines, ACID will instead use a default width of 1000 km/s, so if you
            want to use your own list, make sure to input wider widths for the H lines.
        telluric_widths : Scalar, optional
            The default telluric width if any widths are missing from the above inputs. For each inputted telluric line, if a width is not provided, 
            this width is used. The default is 21 km/s, which is the typical width of telluric lines. If you are masking H lines, it is recommended 
            you set them in the above telluric_lines input.
        seed : int | None, optional
            Random seed for reproducibility, set it to None to be a random seed, by default 42 (the answer to life,
            the universe and everything)
        data : Data|DataList|None, optional
            An optional backend Data object to use for storing data. Allows previously calculated results to be skipped.
            If None, a new Data object is created. Please note that if the Data class already has a saved ACID config
            class, then those config values will overwrite the inputted values in initialisation or ACID method. If a 
            DataList instance is inputted, the data instance corresponding to the inputted order is used.
        config : Config, optional
            An optional Config object to use for storing configuration. Allows you to override the config values stored in the Data object,
            otherwise, inputs to the init here and the ACID method will overwrite these config values again (if entered).
        """
        # TODO write docstring for all defaults, and all new features docstring them please
        # TODO: update readthedocs with examples for using own linelists and using own masking lines, and also eventually using the Datalist object
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

        # Set linelist in the Data class, the property setter handles input validation
        self.data.set_linelist(linelist_path, linelist_wl, linelist_depths)

        # Set the lines to mask, the telluric_lines and hydrogen_lines property setters in the config class handle input validation and None check
        self.config.telluric_lines = telluric_lines
        self.config.telluric_widths = telluric_widths if telluric_widths is not None else self.config.telluric_widths
        self.config.hydrogen_lines = hydrogen_lines
        self.config.hydrogen_widths = hydrogen_widths if hydrogen_widths is not None else self.config.hydrogen_widths

        # Set seed if not already done in config, in this way, seed is only explicitly set once
        if getattr(self.config, "seed", None) is None:
            self.config.seed = seed
            if self.config.seed is not None:
                np.random.seed(self.config.seed) # In principle this is only ever called once
            # else: user may define a seed at the top of their seed, so can use that
        # else: seed already in config, so seed would already have been set when put in
        # I may make seed a property of the config class in the future

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
    _INIT_KEYS = set(inspect.signature(__init__).parameters) - {"self"}

    def ACID(
        self,
        wavelengths           : Array1D|Array2D        = None, # Data
        flux                  : Array1D|Array2D        = None, # Data
        errors                : Array1D|Array2D        = None, # Data
        sn                    : Array1D|Array2D|Scalar = None, # Data
        deterministic_profile : bool                   = None, # Config
        poly_ord              : IntLike                = None, # Config
        continuum_percentile  : IntLike                = None, # Config
        bin_size              : IntLike                = None, # Config
        pix_chunk             : IntLike                = None, # Config
        dev_perc              : IntLike                = None, # Config
        n_sig                 : IntLike                = None, # Config
        skips                 : IntLike                = None, # Config
        parallel              : bool                   = None, # Config
        cores                 : IntLike                = None, # Config
        nsteps                : IntLike                = None, # Config as the initial steps, Data.nsteps is the true count of steps taken, which can be higher
        max_steps             : IntLike                = None, # Config
        check_interval        : IntLike                = None, # Config
        min_checks            : IntLike                = None, # Config
        min_tau_factor        : IntLike                = None, # Config
        tau_tol               : float                  = None, # Config
        moves                 : list                   = None, # Config
        run_mcmc              : bool                   = True, # Config
        _all_frames                                    = None, # To work with legacy code, not to be used, silently ignored
        **kwargs,
        ):
        """Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra,
        returning an LSD profile for each spectrum given. Spectra must cover a similiar wavelength range.

        Important note: All defaults in the code are None, meaning if any values are input, they will override the default Config and/or Data values or
        any values that have already been input. The defaults within the config are written below. The config defaults can also be accessed by:
        ACID_code_v2.Config.defaults (returning a dictionary of defaults for both initialisation and run_acid)

        Parameters
        ----------
        wavelengths : np.ndarray | list | None, optional
            An array of wavelengths for each frame (in Angstroms). For multiple frames this should be a 2-d array such that
            wavelengths[i] corresponds to the wavelengths for the ith frame. Can only be None if a data instance was 
            provided in initialisation. If a 2D array is provided, they are treated as multiple frames (not orders), by default None
        flux : np.ndarray | list | None, optional
            An array of spectral frames (in flux). For multiple frames this should be a 2-d array such that 
            flux[i] corresponds to the spectral fluxes for the ith frame. Can only be None if a data instance was 
            provided in initialisation. If a 2D array is provided, they are treated as multiple frames (not orders), by default None
        errors : np.ndarray | list | None, optional
            Errors for each frame (in flux). For multiple frames this should be a 2-d array such that
            errors[i] corresponds to the spectral errors for the ith frame. Can only be None if a data instance was 
            provided in initialisation. If a 2D array is provided, they are treated as multiple frames (not orders), by default None
        sn : int | np.ndarray | list | None, optional
            Average signal-to-noise ratio for each frame (used to calculate minimum line depth to consider from line list).
            Each frame should have only one S/N value, so for multiple frames this should be a 1-d array such that
            sn[i] corresponds to the S/N for the ith frame. If None, the S/N will be estimated from the input
            spectra, by default None
        deterministic_profile : bool, optional
            If True, fits both the continuum and the LSD profile simultaneously. If False, only fits the continuum in mcmc, the
            profile is inferred from the continuum fit. Setting this to False can significantly speed up compution time, 
            depending on the machine used as it is not as easy to parallelise. It may decrease accuracy, and is not fully tested
            as of yet, by default True.
        poly_ord : int, optional
            Order of polynomial to fit as the continuum, by default 3
        continuum_percentile : int, optional
            The percentile to use when fitting the continuum, by default 90. For example, if 90, the continuum fit will be performed
            on the points in the spectra that are above the 90th percentile in flux.
        bin_size : int, optional
            The size of bins to use when performing the continuum fit. The spectra are split into bins of this size, and the continuum
            is fit to the median wavelength and the specified percentile of flux in each bin. By default 100 pixels.
        pix_chunk : int, optional
            Size of 'bad' regions in pixels. 'bad' areas are identified by the residuals between an inital model
            and the data. If a residual deviates by a specified percentage (dev_perc) for a specified number of pixels,
            by default 20
        dev_perc : int, optional
            Allowed deviation percentage. 'bad' areas are identified by the residuals between an inital model
            and the data. If a residual deviates by a specified percentage (dev_perc) for a specified number of pixels,
            by default 25
        n_sig : int, optional
            Number of sigma to clip in sigma clipping. Ill fitting lines are identified by sigma-clipping the
            residuals between an inital model and the data. The regions that are clipped from the residuals will
            be masked in the spectra. This masking is only applied to find the continuum fit and is removed when
            LSD is applied to obtain the final profiles, by default 1
        skips : int, optional
            An option to only select one in every n pixels, where n is the integer argument. This is only useful for
            testing to get a quick result, by default 1
        parallel : bool, optional
            If True uses multiprocessing to calculate the profiles for each frame in parallel, by default True
        cores : int, optional
            Number of cores to use if parallel=True. If None, all available cores will be used, by default None
        nsteps : int, optional
            nsteps (int, optional): Number of steps for the MCMC to run, try increasing if it doesn't converge,
            by default 10000
        max_steps : int | None, optional
            If set, the sampler will run until max_steps or convergence is reached by estimation using the emcee autocorrelation 
            time (tau). The sampler will check for convergence every 'check_interval' steps, and will require a minimum number 
            of checks ('min_checks') and a minimum tau factor ('min_tau_factor') before it can stop. The stopping criterion 
            is met when the change in tau is less than 'tau_tol' for all parameters. By default None, which means no maximum. 
            If a value is inputted, the nsteps parameter is ignored. The continue_sampling method in Result or Acid can still
            be used normally to continue after either stopping criterion is reached.
        check_interval : int, optional
            Interval (in steps) at which to check for MCMC convergence if max_steps is set, by default 1000. 
            Only used if max_steps is set.
        min_checks : int, optional
            Minimum number of checks before MCMC can be stopped, by default 3. Only used if max_steps is set.
        min_tau_factor : int, optional
            Minimum tau factor for MCMC stopping criterion, by default 50. Only used if max_steps is set.
        tau_tol : float, optional
            Tolerance for tau convergence in MCMC stopping criterion, by default 0.01. Only used if max_steps is set.
        moves : list[tuple], optional
            A list of tuples specifying the moves. Each tuple should be in the format:
            (move_name:str, fraction:float, move_kwargs:Optional[dict]).
                - move_name: The name of the emcee move, the only possible variants are currently as follows:
                "RedBlueMove", "StretchMove", "WalkMove", "KDEMove", "DEMove", "DESnookerMove", 
                "MHMove", "GaussianMove". Refer to the emcee documentation for more details on each move type. The move
                names that get input are checked against the emcee.moves module for if they exist, so any move in that 
                module can be used, but not all of them are tested with ACID.
                - fraction: The fraction of walkers to which this move should be applied.
                - move_kwargs: A dictionary of keyword arguments to pass to the move class initialisation.
        run_mcmc : bool, optional
            If True, runs the MCMC to fit the model, by default True. Can be set to False to perform all of the preparation
            for MCMC without actually running it. The ACID function will still update the class and data attributes.
        **kwargs : dict, optional
            Additional keyword arguments. kwargs are passed to the Result class when returning the Result object,
            see Result class for more details on what kwargs can be passed. Note that any kwargs that are also 
            in the class initialisation will be ignored, and the inputted value will not be used. This is to 
            avoid confusion for users who may accidentally input an argument that is meant for the class 
            initialisation rather than the ACID function, which takes different arguments. The ignored 
            kwargs are checked for and printed at the start of the function.
        Returns
        -------
        Result
            Result object containing the LSD profiles and associated data. See Result class for methods and attributes.
            If run_mcmc is False, None is returned, but the class attributes are still updated (so that acid.data can be 
            used for example).

        Raises
        ------
        TypeError
            If the input types are not as expected.
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

        # Raise an error if the kwargs are not part of either the ACID init or the Result init, so that the error happens
        # now rather than during the Result initialisation, which would waste the user's time
        valid_result_keys = set(inspect.signature(Result.__init__).parameters) - {"self"}
        invalid_keys = set(kwargs.keys()) - self._INIT_KEYS - valid_result_keys
        if invalid_keys and self.config.verbose > 0:
            raise ValueError(f"The following kwargs are not valid for either the ACID initialisation or the Result "
                             f"initialisation: {', '.join(invalid_keys)}. Please check your input arguments.")

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
        # Inputs: self.x, self.y, self.yerr, self.data.model_inputs, self.poly
        # Sets: self.c_factor, self.residual_masks
        # Modifies: self.yerr, and as of 1.5, 
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
            self.residual_mask() # will eventually add options for this

        ## Setting number of walkers and their start values(pos)
        self.data.ndim = len(self.data.model_inputs)
        factor = 3 # emcee recommendation
        self.data.nwalkers = self.data.ndim * factor
        rng = np.random.default_rng(self.config.seed)

        ### starting values of walkers with independent variation
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

        if self.config.deterministic_profile is True:
            self.data.ndim = self.config.poly_ord + 1
            self.data.nwalkers = self.data.ndim * factor
            initial_state = np.array(initial_state)[-self.data.ndim:, :self.data.nwalkers]

        # Transpose initial state to have shape (nwalkers, ndim) for emcee
        initial_state = np.transpose(np.array(initial_state))
        self.data.initial_state = initial_state # Saved for debugging if needed, otherwise class variable not used for now

        ### ACID initialialised ###
        self.data.initialisation_time = time.time() - init_t0
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
            self.data.mcmc_time = time.time() - init_t0 - self.data.initialisation_time
            return Result(self)

        else:
            if self.config.verbose > 0:
                print("MCMC not run, returning None. Class attributes have been updated.")
            return None

    def ACID_HARPS(self, *args, **kwargs):
        raise NotImplementedError(f"ACID_HARPS is no longer supported in ACID v2. \n"
        f"Please use the ACID function with the appropriate inputs for HARPS spectra instead. \n"
        f"Future versions of ACID will provide functions to load and configure data from a range of different standard instruments. \n"
        f"If you still really wish to use ACID_HARPS, the last stable version of ACID with the method is 1.4.5. Try: pip install ACID_code_v2==1.4.5")

    def combine_spec(
        self,
        frame_wavelengths: Array1D|Array2D|None = None,
        frame_flux:        Array1D|Array2D|None = None,
        frame_errors:      Array1D|Array2D|None = None,
        frame_sns:         Array1D|Array2D|None = None,
        output:            bool                 = True
        ):
        """Combines multiple spectral frames into one spectrum

        Parameters
        ----------
        frame_wavelengths : array, optional
            Wavelengths for the spectral frames, by default None
        frame_flux : array, optional
            Fluxes for the spectral frames, by default None
        frame_errors : array, optional
            Errors for the spectral frames, by default None
        frame_sns : array, optional
            Signal-to-noise ratio for the spectral frames, by default None
        output : bool, optional
            Whether to output the combined spectrum, by default True

        Returns
        -------
        tuple, if output is True, containing:
            combined_wavelengths : array
                Wavelengths for the combined spectrum
            combined_spectrum : array
                Fluxes for the combined spectrum
            combined_errors : array
                Errors for the combined spectrum
            combined_sn : float
                Signal-to-noise ratio for the combined spectrum
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
        fluxes     : Array1D,
        wavelengths: Array1D,
        errors     : Array1D,
        poly_ord   : IntLike = 3,
        plot_result: bool    = False,
        plot_type : str     = "initial"
        ):
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
        plot_type : str, optional
            The type of plot to generate, either "initial" or "masked", by default "initial"

        Returns
        -------
        tuple
            A tuple containing the polynomial coefficients, the normalized flux, and the normalized errors.
        """
        a, b = utils.get_normalisation_coeffs(wavelengths)
        unnormalized_wavelengths = np.copy(wavelengths)
        wavelengths = (wavelengths*a)+b

        idx = np.argsort(wavelengths)
        w = wavelengths[idx]
        f = fluxes[idx]
        e = errors[idx]

        binsize = self.config.bin_size
        n = len(w) // binsize  # full bins only

        w2 = w[:n*binsize].reshape(n, binsize)
        f2 = f[:n*binsize].reshape(n, binsize)
        e2 = e[:n*binsize].reshape(n, binsize)

        clipped_flux = np.nanpercentile(f2, self.config.continuum_percentile, axis=1)
        clipped_waves = np.nanmedian(w2, axis=1)
        clipped_errs = np.nanmedian(e2, axis=1)

        good = (
            np.isfinite(clipped_waves)
            & np.isfinite(clipped_flux)
            & np.isfinite(clipped_errs)
            & (clipped_errs > 0)
            & (clipped_errs < 1e11) # 1e12 is the default mask error value, which can be picked up in the median error binning
        )

        coeffs = np.polyfit(clipped_waves[good], clipped_flux[good], poly_ord, w=1/clipped_errs[good])
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)
        flux_obs = fluxes / fit
        new_errors = errors / fit
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
            raise ValueError("Continuum fit resulted in non-positive flux or errors, which is not physical.\n " \
            "Consider adjusting the polynomial order or continuum percentile. Use verbose=3 to see the plot of the continuum fit.\n " \
            "Note that this will only work for interactive terminals or displays which work with plt.show()")

        return poly_coeffs, flux_obs, new_errors

    def residual_mask(
        self,
        ):
        """Masks regions of the spectrum based on residuals from an initial model fit.
        """

        ## iterative residual masking - mask continuous areas first - then possibly progress to masking the narrow lines

        # Set standard variables
        x = self.data.wavelengths["combined"]
        y = self.data.flux["combined"]
        yerr = self.data.errors["combined"]
        sn = self.data.sn["combined"]
        forward, _ = mcmc.MCMC(x, y, yerr, self.data.alpha).full_model(self.data.model_inputs)

        # data_normalised = (y - np.min(y)) / (np.max(y) - np.min(y))
        # forward_normalised = (forward - np.min(forward)) / (np.max(forward) - np.min(forward))
        # residuals = data_normalised - forward_normalised
        residuals = (y - forward) / forward

        ### finds consectuative sections where at least pix_chunk points have residuals greater than 0.25 - these are masked
        # idx = (abs(residuals) > self.config.dev_perc / 100)
        # flag_min = 0
        # flag_max = 0
        # for value in range(len(idx)):
        #     if idx[value] == True and flag_min <= value:
        #         flag_min = value
        #         flag_max = value
        #     elif idx[value] == True and flag_max < value:
        #         flag_max = value
        #     elif idx[value] == False and flag_max - flag_min >= self.config.pix_chunk:
        #         yerr[flag_min:flag_max] = 1e12
        #         flag_min = value
        #         flag_max = value
        bad_idx = np.abs(residuals) > (self.config.dev_perc / 100)

        padded = np.concatenate(([False], bad_idx, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:])
        pix_mask = np.zeros_like(residuals, dtype=bool)

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

        ###############################################
        #         TELLURIC AND HYDROGEN LINES         #   
        ###############################################

        telluric_mask = self.config.telluric_lines.get_mask(x)
        hydrogen_mask = self.config.hydrogen_lines.get_mask(x)
        yerr[telluric_mask | hydrogen_mask] = 1e12

        # Note that this is used to keep track of the residual masks for later use in _get_profiles
        self.data.residual_masks = tuple([yerr >= 1e12])

        ###################################
        ###      sigma clip masking     ###
        ###################################

        m = np.median(residuals)
        sigma = np.std(residuals)
        clip = self.config.n_sig * sigma

        lower_clip = m - clip
        upper_clip = m + clip
        mask = (residuals <= lower_clip) | (residuals >= upper_clip)
        yerr[mask] = 1e12

        # Now do final LSD call
        poly_inputs, fitted_flux, fitted_errors  = self.continuumfit(y, x, yerr, self.config.poly_ord,
                                                   plot_result=self.config.verbose > 2,
                                                   plot_type="masked")

        LSD_masking = LSD(self.data)
        # Since the above ONLY modifies yerr, and the alpha matrix is independent of yerr, we can input previous 
        # alpha since it wil be the same. We still run LSD to get c_factor and the profile
        LSD_masking.run_LSD(x, fitted_flux, fitted_errors, sn, alpha=self.data.alpha)

        ### Update and set new variables
        self.data.c_factor = LSD_masking.c_factor
        self.data.initial_model_inputs = np.copy(self.data.model_inputs) # Save the initial model inputs before masking for later use if needed
        self.data.model_inputs = np.concatenate((LSD_masking.profile, poly_inputs))

        self.data.wavelengths["masked"] = x
        self.data.flux["masked"]        = y # x and y dont change in this func
        self.data.errors["masked"]      = yerr # yerr is modified in this func
        self.data.sn["masked"]          = np.copy(self.data.sn["combined"]) # SN is not changed in this func
        # self.alpha is also modified in this func to get new alpha with masked residuals using pix chunk and dev perc

        # Set required variables for plotting
        if "residual_masking" not in self.data.plotting_variables:
            self.data.plotting_variables["residual_masking"] = {}
        self.data.plotting_variables["residual_masking"]["mask"] = mask
        self.data.plotting_variables["residual_masking"]["residuals"] = residuals
        self.data.plotting_variables["residual_masking"]["upper_clip"] = upper_clip
        self.data.plotting_variables["residual_masking"]["lower_clip"] = lower_clip
        self.data.plotting_variables["residual_masking"]["telluric_mask"] = telluric_mask
        self.data.plotting_variables["residual_masking"]["pix_mask"] = pix_mask
        self.data.plotting_variables["residual_masking"]["profile_F"] = LSD_masking.profile_F
        if self.config.verbose > 2:
            self.data.plot_residual_masking()

        return

    def run_mcmc(
        self,
        nsteps,
        state = None,        
        ):

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
                self.sampler = emcee.EnsembleSampler(**sampler_kwargs, pool=pool, log_prob_fn=mcmc._mp_log_probability)
                self.sampler.run_mcmc(**mcmc_kwargs)

        else:
            MCMC = mcmc.MCMC(self.data)
            self.sampler = emcee.EnsembleSampler(**sampler_kwargs, log_prob_fn=MCMC)
            self.sampler.run_mcmc(**mcmc_kwargs)

    def run_mcmc_until_converged(
        self,
        max_steps      : IntLike,
        state=None,
        ):

        sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps=self.config.check_interval, state=state)
        stopping_criterion_args = (self.config.min_checks, self.config.min_tau_factor, self.config.tau_tol)

        step_number = 0
        tau_list = []
        max_samples = max_steps // self.config.check_interval
        last_tolerance = np.inf
        last_neff = 0

        if self.config.parallel:

            utils.configure_mp_environ(os) # Raises error is not configured correctly, otherwise does nothing
            
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=self.config.cores, initializer=mcmc._mp_init_worker, initargs=(self.data,)) as pool:
                self.sampler = emcee.EnsembleSampler(**sampler_kwargs, pool=pool, log_prob_fn=mcmc._mp_log_probability)
                for i in range(max_samples):
                    tol_str, neff_str = mcmc.MCMC.get_tqdm_desc(last_tolerance, last_neff, self.config)
                    desc_dict = {"desc": f"Iteration {i+1}/{max_samples}, last tolerance: {tol_str}, neff: {neff_str}"}
                    mcmc_kwargs["progress_kwargs"] = desc_dict
                    self.sampler.run_mcmc(**mcmc_kwargs, skip_initial_state_check=True)
                    
                    mcmc_kwargs["initial_state"] = None # only use initial state for first run
                    step_number += self.config.check_interval

                    try:
                        # We want to keep the time for get_autocorr_time to run constant, so thin accordingly 
                        tau = self.sampler.get_autocorr_time(tol=0, thin=step_number//self.config.check_interval)
                    except emcee.autocorr.AutocorrError:
                        continue

                    tau_list.append(tau)
                    condition, last_tolerance, last_neff = mcmc.MCMC.get_mcmc_stopping_criterion(tau_list, step_number, *stopping_criterion_args)
                    if condition is True and self.config.verbose > 1:
                        print(f"Converged at step {step_number}. Final tolerance: {last_tolerance:.4f}, final effective sample size: {last_neff:.2f}.")
                        break
                if self.config.verbose > 1 and condition is False:
                    print(f"Not converged after reaching max steps of {step_number}. Final effective sample size: {last_neff:.2f}, final tolerance: {last_tolerance:.4f}.\n"
                          f"Consider increasing max_steps.")
        else:
            MCMC = mcmc.MCMC(self.data)
            self.sampler = emcee.EnsembleSampler(**sampler_kwargs, log_prob_fn=MCMC)

            for i in range(max_samples):
                tol_str, neff_str = mcmc.MCMC.get_tqdm_desc(last_tolerance, last_neff, self.config)
                desc_dict = {"desc": f"Iteration {i+1}/{max_samples}, last tolerance: {tol_str}, neff: {neff_str}"}
                mcmc_kwargs["progress_kwargs"] = desc_dict
                self.sampler.run_mcmc(**mcmc_kwargs, skip_initial_state_check=True)
                mcmc_kwargs["initial_state"] = None # only use initial state for first run
                step_number += self.config.check_interval

                try:
                    tau = self.sampler.get_autocorr_time(tol=0)
                except emcee.autocorr.AutocorrError:
                    continue

                tau_list.append(tau)
                condition, last_tolerance, last_neff = MCMC.get_mcmc_stopping_criterion(tau_list, step_number, *stopping_criterion_args)
                if condition is True and self.config.verbose > 1:
                    print(f"Converged at step {step_number}. Final tolerance: {last_tolerance:.4f}, final effective sample size: {last_neff:.2f}.")
                    break
            if self.config.verbose > 1 and condition is False:
                    print(f"Not converged after reaching max steps of {step_number}. Final effective sample size: {last_neff:.2f}, final tolerance: {last_tolerance:.4f}.\n"
                          f"Consider increasing max_steps.")
        
        self.step_number = step_number

    def _get_sampler_kwargs(self, nsteps, state=None):

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

        # Configure moves based on config
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
        max_steps_kwards : dict|None = None
        ):
        """Continue MCMC sampling for additional steps. This should be called in Result class by the user.
        This necessarily requires a Data instance to have been put into the ACID init.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            The existing MCMC sampler to continue sampling from.
        nsteps : int
            Number of additional steps to run the MCMC for.
        max_steps : int, optional
            Maximum number of steps to run the MCMC for in total (including previous steps).
            If specified, the MCMC will stop if this number of steps is reached even if convergence has not been reached, by default None.
            If input, nsteps is ignored.
        max_steps_kwards : dict, optional
            Additional keyword arguments to be passed to the run_mcmc_until_converged function if max_steps is specified, by default None.
            The kwargs description can be found in Acid.ACID(), they are the 4 kwargs appearing after max_steps. Typos for kwargs are silently
            ignored.

        Returns
        -------
        emcee.EnsembleSampler
            The MCMC sampler after running for the additional steps.
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

        return self.sampler

    def get_result(
        self=None,
        ):
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

    @staticmethod
    def combineprofiles(
        spectra: Array2D,
        errors: Array2D,
        ):
        spectra = np.asarray(spectra)
        errors = np.asarray(errors)
        if spectra.shape != errors.shape:
            raise ValueError(f"Spectra and errors must have the same shape. Got {spectra.shape} and {errors.shape}.")

        # idx = np.isnan(spectra)
        # shape_og = spectra.shape
        # if len(spectra[idx])>0:
        #     spectra = spectra.reshape((len(spectra)*len(spectra[0]), ))
        #     for n in range(len(spectra)):
        #         if spectra[n] == np.nan:
        #             spectra[n] = (spectra[n+1]+spectra[n-1])/2
        #             if spectra[n] == np.nan:
        #                 spectra[n] = 0.
        # spectra = spectra.reshape(shape_og)
        # errors = np.array(errors)

        spectra_to_combine = []
        weights=[]
        for n in range(0, len(spectra)):
            if np.sum(spectra[n])!=0:
                spectra_to_combine.append(list(spectra[n]))
                temp_err = np.array(errors[n, :])
                weight = (1/temp_err**2)
                weights.append(np.mean(weight))
        weights = np.array(weights/sum(weights))

        spectra_to_combine = np.array(spectra_to_combine)

        length, width = np.shape(spectra_to_combine)
        spectrum = np.zeros((1,width))
        spec_errors = np.zeros((1,width))

        for n in range(0, width):
            temp_spec = spectra_to_combine[:, n]
            spectrum[0,n]=sum(weights*temp_spec)/sum(weights)
            spec_errors[0,n] = (np.std(temp_spec, ddof=1)**2) * np.sqrt(sum(weights**2))

        spectrum = list(np.reshape(spectrum, (width,)))
        spec_errors = list(np.reshape(spec_errors, (width,)))

        return spectrum, spec_errors

# All code below is just to ensure backward compatibility with previous ACID versions
def ACID(*args, **kwargs):
    """Legacy ACID function

    This function runs the legacy ACID code. This is provided for backwards compatibility with previous versions of ACID.
    It is recommended to use the ACID class and its methods for new code. The args and kwargs passing follows the original
    v1 version of ACID, which can be found in https://github.com/ldolan05/ACID

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
        "line": "linelist_path",
        "poly_or": "poly_ord",
        "all_frames": "_all_frames",
    }

    # Split args and kwargs into init and run kwargs using helper function
    init_kwargs, run_kwargs = _get_init_and_run_kwargs(LEGACY_ACID_ARGS, RENAMED_LEGACY_ARGS, *args, **kwargs)

    acid = Acid(**init_kwargs)
    return acid.ACID(**run_kwargs)

def ACID_HARPS(*args, **kwargs):
    """Legacy ACID_HARPS function, depracated after 1.4.5.
    """
    raise NotImplementedError(f"ACID_HARPS is no longer supported in ACID v2. \n"
        f"Please use the ACID function with the appropriate inputs for HARPS spectra instead. \n"
        f"Future versions of ACID will provide functions to load and configure data from a range of different standard instruments. \n"
        f"If you still really wish to use ACID_HARPS, the last stable version of ACID with the method is 1.4.5. Try: pip install ACID_code_v2==1.4.5")

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