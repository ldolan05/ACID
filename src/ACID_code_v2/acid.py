import warnings
warnings.filterwarnings("ignore")
import sys, emcee, os, time, inspect, inspect
import numpy as np
from math import log10, floor
from astropy.io import fits
from scipy.interpolate import interp1d
import multiprocessing as mp
from beartype import beartype
from numpy import integer as npint
import matplotlib.pyplot as plt
import scipy.constants as const
from . import utils
from .lsd import LSD
from . import mcmc
from .result import Result
from .data import Data

c_kms = float(const.c/1e3)

@beartype
class Acid:
    """Accurate Continuum fItting and Deconvolution (ACID) class. This class contains the ACID method 
    which fits the continuum of spectra and performs Least Squares Deconvolution (LSD) to obtain
    LSD profiles for each spectrum. It also contains many internal methods used within the main ACID 
    function."""

    def __init__(
        self,
        velocities     :np.ndarray|None      = None,
        linelist_path                        = None,
        linelist_wl    :np.ndarray|list|None = None,
        linelist_depths:np.ndarray|list|None = None,
        verbose        :int|npint|bool|None  = 2,
        telluric_lines :np.ndarray|list|None = None,
        name           :str                  = 'ACID',
        seed           :int|npint|None       = None,
        data                                 = None,
        ):
        """Initialises the Acid class with inputted parameters. The parameters set here arre independent
        of the choice of the ACID and ACID_HARPS functions, which take different formats for inputted spectra.

        Parameters
        ----------
        velocities : np.ndarray | None, optional
            Velocity grid for LSD profiles (in km/s). For example, use: np.arange(-25, 25, 0.82) to create. If None, a default grid
            from -25 to 25 km/s with a spacing calculated by calc_deltav. It is highly recommended to choose your own velocity grid, 
            by default None
        linelist_path : str | None, optional
            Can be a path to linelist in string format, a dictionary with keys "wavelengths" and "depths", a Linelist class, or a 
            list/array indexed such that 0 is the wavelengths and 1 is the depths. If None, you can directly provide linelist_wl
            and linelist_depths instead. At least one of linelist_path or linelist_wl and linelist_depths must be provided. By default None.
        linelist_wl : np.ndarray | list | None, optional
            Wavelengths of lines in linelist (in Angstroms). Only necessary if linelist_path is not provided. 
            Must be same length as linelist_depths. If None, linelist_path must be provided., by default None
        linelist_depths : np.ndarray | list | None, optional
            Depths of lines in linelist (between 0 and 1). Only necessary if linelist_path is not provided. 
            Must be same length as linelist_wl. If None, linelist_path must be provided., by default None
        verbose : bool | int | None, optional
            An integer between 0 and 3. If 0, nothing is printed. If 2, prints out useful progress information, as well as ACID warnings 
            about any potential issues with the input data or autocorrelation warnings. If True, defaults to 2. If False, defaults to 0.
            If you want to ignore the warnings but still keep progress information, set verbose to 1. A verbosity of 3 will produce 
            additional plots, such as the result of the continuum fit. By default None, which defaults to 2 (True). If set, overrides 
            any verbose setting in the dataclass.
        telluric_lines : np.ndarray | list | None, optional
            List of wavelengths (in Angstroms) of telluric lines to be masked. This can also include problematic
            lines/features that should be masked also. For each wavelengths in the list ~3Å eith side of the line is masked. By default None
        name : str, optional
            Name to call any saved files, by default 'ACID'
        seed : int | None, optional
            Random seed for reproducibility, set it to None to be a random seed, by default 42 (the answer to life,
            the universe and everything)
        data : Data|None, optional
            An optional backend Data object to use for storing data. Allows previously calculated results to be skipped.
            If None, a new Data object is created, by default None. Please note that if the Data class already has a saved ACID config
            class, then those config values will overwrite the inputted values in initialisation or ACID method.
            
        """
        # Initialise the data class to store calculations in ACID
        if data is not None:
            self.data = data
        else:
            self.data = Data()
        
        # Set config if old one exists
        self.config = self.data.config # Make config the same as old config, or generates a new empty one (handled in Data)

        # Validate velocities input, if None, this is handled in ACID function later when a input spectrum is provided
        if velocities is not None:
            if velocities.ndim != 1:
                raise ValueError("'velocities' must be a one-dimensional array")
        # data.velocities defaults to None in Data class, can be set in ACID function
        self.data.velocities = self.data.velocities if self.data.velocities is not None else velocities

        # Verbosity validation handled in config property setter
        self.config.verbose = verbose

        # Set linelist in the Data class, the property setter handles input validation
        self.data.set_linelist(linelist_path, linelist_wl, linelist_depths)

        self.config.telluric_lines = telluric_lines

        # Set seed if not already done in config, in this way, seed is only explicitly set once
        if getattr(self.config, "seed", None) is None:
            self.config.seed = seed
            if self.config.seed is not None:
                np.random.seed(self.config.seed) # In principle this is only ever called once
            # else: user may define a seed at the top of their seed, so can use that
        # else: seed already in config, so seed would already have been set when put in
        # I may make seed a property of the config class in the future

        # Name is also added to config
        self.config.name = name if getattr(self.config, "name", None) is None else self.config.name

        # Default order range for ACID, can be updated in ACID_HARPS. Eventually will add option to add this to inputs
        self.config.order_range = [1]

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
        input_wavelengths                     = None,
        input_flux                            = None,
        input_errors                          = None,
        input_sn                              = None,
        all_frames                            = None,
        deterministic_profile :bool           = False,
        poly_ord              :int|npint      = 3,
        pix_chunk             :int|npint      = 20,
        dev_perc              :int|npint      = 25,
        n_sig                 :int|npint      = 1,
        order                 :int|npint      = 0,
        skips                 :int|npint      = 1,
        parallel              :bool           = True,
        cores                 :int|npint|None = None,
        nsteps                :int|npint      = 10000,
        max_steps             :int|npint|None = None,
        check_interval        :int|npint      = 1000,
        min_checks            :int|npint      = 1,
        min_tau_factor        :int|npint      = 50,
        tau_tol               :float          = 0.05,
        run_mcmc              :bool           = True,
        **kwargs,
        ):
        """Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra,
        returning an LSD profile for each spectrum given. Spectra must cover a similiar wavelength range.

        Parameters
        ----------
        input_wavelengths : np.ndarray | list | None, optional
            An array of wavelengths for each frame (in Angstroms). For multiple frames this should be a 2-d array such that
            input_wavelengths[i] corresponds to the wavelengths for the ith frame. Can only be None if a data instance was 
            provided in initialisation.
        input_flux : np.ndarray | list | None, optional
            An array of spectral frames (in flux). For multiple frames this should be a 2-d array such that 
            input_flux[i] corresponds to the spectral fluxes for the ith frame. Can only be None if a data instance was 
            provided in initialisation., by default None
        input_errors : np.ndarray | list | None, optional
            Errors for each frame (in flux). For multiple frames this should be a 2-d array such that
            input_errors[i] corresponds to the spectral errors for the ith frame. Can only be None if a data instance was 
            provided in initialisation., by default None
        input_sn : int | np.ndarray | list | None, optional
            Average signal-to-noise ratio for each frame (used to calculate minimum line depth to consider from line list).
            Each frame should have only one S/N value, so for multiple frames this should be a 1-d array such that
            input_sn[i] corresponds to the S/N for the ith frame. If None, the S/N will be estimated from the input
            spectra, by default None
        all_frames : str | np.ndarray | None, optional
            Output array for resulting profiles. Only neccessary if looping ACID function over many wavelength
            regions or order (in the case of echelle spectra). General shape needs to be
            (no. of frames, no. of orders, 2, no. of velocity pixels). If not provided, one is created with that shape.
             The only allowed string is "default" due to legacy behaviour, which now acts the same as None, by default None
        deterministic_profile : bool, optional
            If True, fits both the continuum and the LSD profile simultaneously. If False, only fits the continuum in mcmc, the
            profile is inferred from the continuum fit. Setting this to False can significantly speed up compution time, 
            depending on the machine used as it is not as easy to parallelise. It may decrease accuracy, and is not fully tested
            as of yet, by default True.
        poly_ord : int, optional
            Order of polynomial to fit as the continuum, by default 3
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
        order : int, optional
            Only applicable if an all_frames output array has been provided as this is the order position in that
            array where the result should be input. i.e. if order = 5 the output profile and errors would be inserted in
            all_frames[:, 5]., by default 0
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
            Interval (in steps) at which to check for MCMC convergence if max_steps is set, by default 100. 
            Only used if max_steps is set.
        min_checks : int, optional
            Minimum number of checks before MCMC can be stopped, by default 3. Only used if max_steps is set.
        min_tau_factor : int, optional
            Minimum tau factor for MCMC stopping criterion, by default 50. Only used if max_steps is set.
        tau_tol : float, optional
            Tolerance for tau convergence in MCMC stopping criterion, by default 0.01. Only used if max_steps is set.
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
        self.data.set_inputs(input_wavelengths, input_flux, input_errors, input_sn, skips)

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
            "order"                 : order,
            "pix_chunk"             : pix_chunk,
            "dev_perc"              : dev_perc,
            "n_sig"                 : n_sig,
            "parallel"              : parallel,
            "cores"                 : cores,
            "deterministic_profile" : deterministic_profile,
            "max_steps"             : max_steps,
            "check_interval"        : check_interval,
            "min_checks"            : min_checks,
            "min_tau_factor"        : min_tau_factor,
            "tau_tol"               : tau_tol,
            "run_mcmc"              : run_mcmc,
        }
        # TODO: make all input defaults None and overwrite config if input, with config handling problems caused therein

        # Update config if any of the above config settings are new
        self.config.update_lowpri(**ACID_config) # self.config overwrites ACID_config if overlapping
        self.data.config = self.config # update dataclass config as well

        if self.config.parallel and sys.platform == "win32":
            if self.config.verbose > 0:
                # This doesn't work, needs serious modifications to make work, so just run serially for now
                print("Parallel MCMC on Windows is not currently supported. Running MCMC serially.")
            self.config.parallel = False

        # Now that the data is set, we can check if the velocities were set in the initialisation or not, and if not,
        # calculate a default velocity grid using the input wavelengths.
        if self.data.velocities is None:
            if self.config.verbose > 0:
                print("Velocity grid not input, using a grid calculated from input wavelengths with default range of -25 to 25 km/s. " \
                "It is recommended to input your own velocity grid, especially if you have a different wavelength range or resolution.")
            deltav = utils.calc_deltav(self.data.wavelengths["input"][0])
            self.data.velocities = np.arange(-25, 25 + deltav, deltav) # default velocity grid from -25 to 25 km/s with spacing calculated from input wavelengths

        # Initiates all_frames variable, which is used to store the results of the MCMC sampling.
        # If an all_frames array is provided, this is used, otherwise a new one is created with the correct shape.
        self.data.initiate_all_frames(all_frames)

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
            hasattr(self.data.sn, "combined")
        )):
            if self.config.verbose > 2:
                print("Combined spectra already exists, skipping combination step.")
        else:
            if self.config.verbose > 2:
                print("Combining spectra...")
            self.combine_spec(output=False)

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
                self.data.wavelengths["combined_normalized"],
                self.data.errors["combined"]
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
        # Modifies: self.alpha, self.yerr
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
                    print("Running MCMC for %s steps..."%nsteps)
                self.run_mcmc(nsteps, initial_state)
                self.data.nsteps += nsteps
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

    def ACID_HARPS(
        self,
        filelist    : list,
        order_range : list|np.ndarray|None = None,
        save_path   : str                  = './',
        file_type   : str                  = 'e2ds',
        **kwargs,
        ):
        """ACID for HARPS e2ds and s1d spectra (DRS pipeline 3.5)

        Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra,
        returning an LSD profile for each file given. Files must all be kept in the same folder as well
        as their corresponding blaze files. If 's1d' are being used their e2ds equivalents must also be
        in this folder. Result files containing profiles and associated errors for each order (or
        corresponding wavelength range in the case of 's1d' files) will be created and saved to a
        specified folder. It is recommended that this folder is seperate to the input files.

        Parameters
        ----------
        filelist : list of strings
            List of files. Files must come from the same observation night as continuum is fit for a combined
            spectrum of all frames. A profile and associated errors will be produced for each file specified.
        order_range : array, optional
            Orders to be included in the final profiles. If s1d files are input, the corresponding wavelengths 
            will be considered, by default None.
        save_path : str, optional
            Path to the directory where output files will be saved, by default './'
        file_type : str, optional
            Type of the input files, either "e2ds" or "s1d", by default 'e2ds'
        **kwargs
            Additional arguments to be passed to the ACID function. See ACID function for details.

        Returns
        -------
        Object
            Result object containing the LSD profiles and associated data. ACID_HARPS=True flag is set to allow
            legacy subscripting and iteration if needed. The legacy subscript and iteration methods will access the
            following attributes:
            list
                Barycentric Julian Date for files
            list
                Profiles (in normalised flux)
            list
                Errors on profiles (in normalised flux)
            It can be accessed for example by:
            >>> result = Acid.ACID_HARPS(...)
            >>> BJDs = result.BJDs
            >>> profiles = result.profiles
            >>> errors = result.errors
            or
            >>> BJDs, profiles, errors = result
        """

        file_type = file_type.lower()
        if file_type not in ['e2ds', 's1d']:
            raise ValueError("file_type must be either 'e2ds' or 's1d'")

        # Handle order_range input
        if order_range is None:
            # Be default, class is initialised with order_range = [1] for HARPS, this part forces
            # order range to np.arange(10, 70) if not specified for the ACID HARPS function.
            order_range = np.arange(10, 70)

        self.config.order_range = np.array(order_range) # Makes sure order range is an array regardless of input type
        self.file_type = file_type
        self.filelist = filelist

        for order in self.config.order_range:
            if self.config.verbose > 1:
                print('Running for order %s/%s...'%(order-min(self.config.order_range)+1, max(self.config.order_range)-min(self.config.order_range)+1))

            frame_wavelengths, frame_flux, frame_errors, sns = self.read_in_frames(order, self.filelist, self.file_type)

            # Updates recursively the all_frames array with the profiles for each order
            self.ACID(
                frame_wavelengths,
                frame_flux,
                frame_errors,
                sns,
                order         = order-min(self.config.order_range),
                **kwargs
            )

        # adding into fits files for each frame
        BJDs = []
        profiles = []
        errors = []
        for frame_no in range(0, len(frame_flux)):
            file = filelist[frame_no]
            fits_file = fits.open(file)
            hdu = fits.HDUList()
            hdr = fits.Header()
            
            for order in self.config.order_range:
                hdr['ORDER'] = order
                hdr['BJD'] = fits_file[0].header['ESO DRS BJD']
                if order == self.config.order_range[0]:
                    BJDs.append(fits_file[0].header['ESO DRS BJD'])
                hdr['CRVAL1'] = np.min(self.data.velocities)
                hdr['CDELT1'] = self.data.velocities[1] - self.data.velocities[0]

                profile = self.data.all_frames[frame_no, order-min(self.config.order_range), 0]
                profile_err = self.data.all_frames[frame_no, order-min(self.config.order_range), 1]

                hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
                if save_path != 'no save':
                    month = 'August2007'
                    hdu.writeto('%s%s_%s_%s.fits'%(save_path, month, frame_no, self.config.name), output_verify='fix', overwrite='True')

            result1, result2 = self.combineprofiles(self.data.all_frames[frame_no, :, 0], self.data.all_frames[frame_no, :, 1])
            profiles.append(result1)
            errors.append(result2)

        self.BJDs = BJDs
        self.profiles = profiles
        self.errors = errors
        # Return Result class with ACID_HARPS=True flag to allow legacy subscripting and iteration if needed.
        return Result(self, ACID_HARPS=True)

    def combine_spec(
        self,
        frame_wavelengths: np.ndarray | None = None,
        frame_flux:        np.ndarray | None = None,
        frame_errors:      np.ndarray | None = None,
        frame_sns:         np.ndarray | None = None,
        output:            bool              = True
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
        Tuple, if output is True, containing:
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
            self.wavelengths["input"] = np.copy(frame_wavelengths)
            self.flux["input"]        = np.copy(frame_flux)
            self.errors["input"]      = np.copy(frame_errors)
            self.sn["input"]          = np.copy(frame_sns)

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
        fluxes     : np.ndarray,
        wavelengths: np.ndarray,
        errors     : np.ndarray,
        poly_ord   : int|npint = 3,
        plot_result: bool      = False
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
        plot_result : bool, optional
            If True, plots the original spectrum and the fitted continuum, by default False

        Returns
        -------
        tuple
            A tuple containing the polynomial coefficients, the normalized flux, and the normalized errors.
        """

        idx = wavelengths.argsort()
        wavelength = wavelengths[idx]
        fluxe = fluxes[idx]
        clipped_flux = []
        clipped_waves = []
        binsize = 100
        for i in range(0, len(wavelength), binsize):
            waves = wavelength[i:i+binsize]
            flux = fluxe[i:i+binsize]
            indicies = flux.argsort()
            flux = flux[indicies]
            waves = waves[indicies]
            clipped_flux.append(flux[len(flux)-1])
            clipped_waves.append(waves[len(waves)-1])
        coeffs = np.polyfit(clipped_waves, clipped_flux, poly_ord)
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)
        flux_obs = fluxes / fit
        new_errors = errors / fit
        poly_coeffs = np.flip(coeffs)

        if self.config.verbose > 2 or plot_result is True:
            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, fluxes, label='Original Spectrum')
            plt.plot(wavelengths, fit, label='Fitted Continuum', color='orange')
            plt.title('Continuum Fit')
            plt.legend()
            plt.show()
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
        x_norm = self.data.wavelengths["combined_normalized"]

        forward, _ = mcmc.MCMC(x, y, yerr, self.data.alpha).full_model(self.data.model_inputs)

        data_normalised = (y - np.min(y)) / (np.max(y) - np.min(y))
        forward_normalised = (forward - np.min(forward)) / (np.max(forward) - np.min(forward))
        residuals = data_normalised - forward_normalised
        
        ### finds consectuative sections where at least pix_chunk points have residuals greater than 0.25 - these are masked
        idx = (abs(residuals) > self.config.dev_perc / 100)

        flag_min = 0
        flag_max = 0
        for value in range(len(idx)):
            if idx[value] == True and flag_min <= value:
                flag_min = value
                flag_max = value
            elif idx[value] == True and flag_max < value:
                flag_max = value
            elif idx[value] == False and flag_max - flag_min >= self.config.pix_chunk:
                yerr[flag_min:flag_max] = 1e12
                flag_min = value
                flag_max = value

        ##############################################
        #                  TELLURICS                 #   
        ##############################################

        # self.yerr_compare = self.yerr.copy()

        ## masking tellurics
        for line in self.config.telluric_lines:
            limit = (21/c_kms)*line +3
            idx = np.logical_and((line-limit) <= x, x <= (limit+line))
            yerr[idx] = 1e12

        # Note that this is used to keep track of the residual masks for later use in _get_profiles
        self.data.residual_masks = tuple([yerr >= 1e12])

        ###################################
        ###      sigma clip masking     ###
        ###################################

        m = np.median(residuals)
        sigma = np.std(residuals)

        upper_clip = m + self.config.n_sig * sigma
        lower_clip = m - self.config.n_sig * sigma

        rcopy = residuals.copy()

        idx1 = tuple([rcopy <= lower_clip])
        idx2 = tuple([rcopy >= upper_clip])

        yerr[idx1] = 1e12
        yerr[idx2] = 1e12

        a, b = utils.get_normalisation_coeffs(x)
        poly_inputs, _bin, bye = self.continuumfit(y, (x*a)+b, yerr, self.config.poly_ord, plot_result=False)

        LSD_masking = LSD(self.data)
        LSD_masking.run_LSD(x, _bin, bye, sn=100)
        # profile = LSD_masking.profile
        self.data.alpha = LSD_masking.alpha
        self.data.c_factor = LSD_masking.c_factor

        if self.config.verbose > 2:
            nremoved = np.sum(idx1)+np.sum(idx2)
            print(f"Residal masking has removed {nremoved}/{len(residuals)} points.")

            plt.figure(figsize=(10, 6))
            plt.plot(x, residuals, label='Residuals', color='blue')
            plt.axhline(upper_clip, color='red', linestyle='--', label='Upper Clip Threshold')
            plt.axhline(lower_clip, color='green', linestyle='--', label='Lower Clip Threshold')
            plt.fill_between(x, upper_clip, lower_clip, color='gray', alpha=0.3, label='Clipped Region')
            for i, line in enumerate(self.config.telluric_lines):
                limit = (21/c_kms)*line + 3
                label = 'Telluric Masking Region' if i==0 else None
                plt.axvspan(line-limit, line+limit, color='orange', alpha=0.5, label=label)
            plt.xlim(np.min(x), np.max(x))
            plt.title('Residuals with Sigma Clipping Thresholds')
            plt.xlabel('Wavelength')
            plt.ylabel('Residuals')
            plt.legend(loc="lower right")
            plt.show()

            plt.figure(figsize=(10, 6))
            plt.plot(self.data.velocities, LSD_masking.profile_F, label='LSD Profile after Masking and before sampling', color='red')
            plt.title('LSD Profile after Residual Masking')
            plt.xlabel('Velocity (km/s)')
            plt.ylabel('LSD Profile')
            plt.legend()
            plt.show()

        self.data.wavelengths["masked"] = x
        self.data.flux["masked"]        = y # x and y dont change in this func
        self.data.errors["masked"]      = yerr # yerr is modified in this func
        self.data.sn["masked"]          = np.copy(self.data.sn["combined"]) # SN is not changed in this func
        # self.alpha is also modified in this func to get new alpha with masked residuals using pix chunk and dev perc

        return

    def run_mcmc(
        self,
        nsteps,
        state = None,        
        ):

        sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps, state)

        if self.config.parallel:
            os.environ["OMP_NUM_THREADS"] = "1" # emcee recommendation for multiprocessing

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
        max_steps      : int|npint,
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
            os.environ["OMP_NUM_THREADS"] = "1" # emcee recommendation for multiprocessing
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=self.config.cores, initializer=mcmc._mp_init_worker, initargs=(self.data,)) as pool:
                self.sampler = emcee.EnsembleSampler(**sampler_kwargs, pool=pool, log_prob_fn=mcmc._mp_log_probability)
                for i in range(max_samples):
                    tol_str, neff_str = mcmc.MCMC.get_tqdm_desc(last_tolerance, last_neff, self.config)
                    desc_dict = {"desc": f"Iteration {i+1}/{max_samples}, last tolerance: {tol_str}, neff: {neff_str}"}
                    mcmc_kwargs["progress_kwargs"] = desc_dict
                    self.sampler.run_mcmc(**mcmc_kwargs)
                    
                    mcmc_kwargs["initial_state"] = None # only use initial state for first run
                    step_number += self.config.check_interval

                    try:
                        tau = self.sampler.get_autocorr_time(tol=0)
                    except emcee.autocorr.AutocorrError:
                        continue

                    tau_list.append(tau)
                    condition, last_tolerance, last_neff = mcmc.MCMC.get_mcmc_stopping_criterion(tau_list, step_number, *stopping_criterion_args)
                    if condition is True and self.config.verbose > 1:
                        print(f"Converged at step {step_number}. Effective sample size: {last_neff:.2f}, tolerance: {last_tolerance:.4f}.")
                        break
                if self.config.verbose > 1 and condition is False:
                    print(f"Not converged after reaching max steps of {step_number}. Effective sample size: {last_neff:.2f}, tolerance: {last_tolerance:.4f}.\n"
                          f"Consider increasing max_steps.")
        else:
            MCMC = mcmc.MCMC(self.data)
            self.sampler = emcee.EnsembleSampler(**sampler_kwargs, log_prob_fn=MCMC)

            for i in range(max_samples):
                tol_str, neff_str = mcmc.MCMC.get_tqdm_desc(last_tolerance, last_neff, self.config)
                desc_dict = {"desc": f"Iteration {i+1}/{max_samples}, last tolerance: {tol_str}, neff: {neff_str}"}
                mcmc_kwargs["progress_kwargs"] = desc_dict
                self.sampler.run_mcmc(**mcmc_kwargs)
                mcmc_kwargs["initial_state"] = None # only use initial state for first run
                step_number += self.config.check_interval

                try:
                    tau = self.sampler.get_autocorr_time(tol=0)
                except emcee.autocorr.AutocorrError:
                    continue

                tau_list.append(tau)
                condition, last_tolerance, last_neff = MCMC.get_mcmc_stopping_criterion(tau_list, step_number, *stopping_criterion_args)
                if condition is True and self.config.verbose > 1:
                    print(f"Converged at step {step_number}. Effective sample size: {last_neff:.2f}, tolerance: {last_tolerance:.4f}.")
                    break
            if self.config.verbose > 1 and condition is False:
                    print(f"Not converged after reaching max steps of {step_number}. Effective sample size: {last_neff:.2f}, tolerance: {last_tolerance:.4f}.\n"
                          f"Consider increasing max_steps.")
        
        self.step_number = step_number

    def _get_sampler_kwargs(self, nsteps, state=None):

        state = self.data.initial_state if state is None else state
        if state is None:
            raise ValueError("No initial state provided and no initial state found in data. " \
            "Please provide an initial state or run 'ACID' first to generate one.")

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
        # TODO Make moves a ACID input
        moves = [
            (emcee.moves.StretchMove(), 0.20),
            (emcee.moves.DESnookerMove(), 0.1),
            (emcee.moves.DEMove(), 0.6),
            (emcee.moves.DEMove(gamma0=1.0), 0.1)
        ]

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
        nsteps        : int|npint,
        ):
        """Continue MCMC sampling for additional steps. This should be called in Result class by the user.
        This necessarily requires a Data instance to have been put into the ACID init.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            The existing MCMC sampler to continue sampling from.
        nsteps : int
            Number of additional steps to run the MCMC for.
        
        Returns
        -------
        emcee.EnsembleSampler
            The MCMC sampler after running for the additional steps.
        """
        assert self.data.alpha is not None, "Data instance must have alpha matrix calculated to continue sampling."

        self.sampler = sampler
        self.config = self.data.config
        self.run_mcmc(nsteps, state=None) # continue from current state

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

    def read_in_frames(
        self,
        order,
        filelist,
        file_type,
        directory=None,
        ):
        # read in first frame
        fluxes, wavelengths, flux_error_order, sn = LSD().blaze_correct(
            file_type, 'order', order, filelist[0], directory, 'unmasked', self.config.name, 'y')
        # fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct(
        #     file_type, 'order', order, filelist[0], directory, 'unmasked', self.config.name, 'y')

        frames = np.zeros((len(filelist), len(wavelengths)))
        errors = np.zeros((len(filelist), len(wavelengths)))
        frame_wavelengths = np.zeros((len(filelist), len(wavelengths)))
        sns = np.zeros((len(filelist), ))

        frames[0] = fluxes
        errors[0] = flux_error_order
        frame_wavelengths[0] = wavelengths
        sns[0] = sn

        def task_frames(frames, errors, frame_wavelengths, sns, i):
            file = filelist[i]
            frames[i], frame_wavelengths[i], errors[i], sns[i] = LSD().blaze_correct(
                file_type, 'order', order, file, directory, 'unmasked', self.config.name, 'y')

            return frames, frame_wavelengths, errors, sns
        
        ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
        for i in range(len(filelist[1:])+1):

            frames, frame_wavelengths, errors, sns = task_frames(frames, errors, frame_wavelengths, sns, i)
            
        ### finding highest S/N frame, saves this as reference frame

        idx = (sns==np.max(sns))
        # global reference_wave
        reference_wave = frame_wavelengths[idx][0]
        reference_frame = frames[idx][0]
        reference_frame[reference_frame == 0] = 0.001
        reference_error = errors[idx][0]
        reference_error[reference_frame == 0] = 1e12

        # global frames_unadjusted
        frames_unadjusted = frames
        # global frame_errors_unadjusted
        frame_errors_unadjusted = errors

        ### each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
        for n in range(len(frames)):
            f2 = interp1d(frame_wavelengths[n], frames[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
            div_frame = f2(reference_wave)/reference_frame

            idx_ref = (reference_frame<=0)
            div_frame[idx_ref]=1

            binned = []
            binned_waves = []
            binsize = int(round(len(div_frame)/5, 1))
            for i in range(0, len(div_frame), binsize):
                if i+binsize<len(reference_wave):
                    waves = reference_wave[i:i+binsize]
                    flux = div_frame[i:i+binsize]
                    waves = waves[abs(flux-np.median(flux))<0.1]
                    flux = flux[abs(flux-np.median(flux))<0.1]
                    binned.append(np.median(flux))
                    binned_waves.append(np.median(waves))

            binned = np.array(binned)
            binned_waves = np.array(binned_waves)
        
            ### fitting polynomial to div_frame
            try:coeffs = np.polyfit(binned_waves, binned, 4)
            except:coeffs = np.polyfit(binned_waves, binned, 2)
            poly = np.poly1d(coeffs)
            fit = poly(frame_wavelengths[n])
            frames[n] = frames[n]/fit
            errors[n] = errors[n]/fit
            idx = (frames[n] == 0)
            frames[n][idx] = 0.00001
            errors[n][idx] = 1e12

        return frame_wavelengths, frames, errors, sns

    def combineprofiles(
        self,
        spectra,
        errors,
        ):
        spectra = np.array(spectra)
        idx = np.isnan(spectra)
        shape_og = spectra.shape
        if len(spectra[idx])>0:
            spectra = spectra.reshape((len(spectra)*len(spectra[0]), ))
            for n in range(len(spectra)):
                if spectra[n] == np.nan:
                    spectra[n] = (spectra[n+1]+spectra[n-1])/2
                    if spectra[n] == np.nan:
                        spectra[n] = 0.
        spectra = spectra.reshape(shape_og)
        errors = np.array(errors)
        
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
        "input_wavelengths": "input_wavelengths",
        "input_spectra": "input_flux",
        "input_spectral_errors": "input_errors",
        "frame_sns": "input_sn",
        "vgrid": "velocities",
        "line": "linelist_path",
        "poly_or": "poly_ord",
    }

    # Split args and kwargs into init and run kwargs using helper function
    init_kwargs, run_kwargs = _get_init_and_run_kwargs(LEGACY_ACID_ARGS, RENAMED_LEGACY_ARGS, *args, **kwargs)

    acid = Acid(**init_kwargs)
    return acid.ACID(**run_kwargs)

def ACID_HARPS(*args, **kwargs):
    """Legacy ACID_HARPS function

    This function runs the legacy ACID_HARPS code. This is provided for backwards compatibility with previous versions of ACID.
    It is recommended to use the ACID class and its methods for new code. The args and kwargs passing follows the original
    v1 version of ACID_HARPS, which can be found in https://github.com/ldolan05/ACID

    Parameters
    ----------
    *args
        Positional arguments to be passed to the run_ACID_HARPS function.
    **kwargs
        Keyword arguments to be passed to the ACID initialisation and run_ACID_HARPS function.

    Returns
    -------
    Any
        Returns the outputs of the run_ACID_HARPS function (now a Result object).
    """

    # Use old argument names and map to new ones
    LEGACY_HARPS_ARGS = [
        "filelist",
        "line",
        "vgrid",
        "poly_or",
        "order_range",
        "save_path",
        "file_type",
        "pix_chunk",
        "dev_perc",
        "n_sig",
        "telluric_lines",
    ]
    RENAMED_LEGACY_ARGS = {
        "input_wavelengths": "input_wavelengths",
        "input_spectra": "input_flux",
        "input_spectral_errors": "input_errors",
        "frame_sns": "input_sn",
        "vgrid": "velocities",
        "line": "linelist_path",
        "poly_or": "poly_ord",
    }

    # Split args and kwargs into init and run kwargs using helper function
    init_kwargs, run_kwargs = _get_init_and_run_kwargs(LEGACY_HARPS_ARGS, RENAMED_LEGACY_ARGS, *args, **kwargs)

    acid = Acid(**init_kwargs)
    return acid.ACID_HARPS(**run_kwargs)

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