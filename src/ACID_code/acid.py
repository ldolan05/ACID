from __future__ import annotations
import traceback, warnings
import sys, emcee, os, time, contextlib
from emcee import EnsembleSampler
import numpy as np
import multiprocessing as mp
from beartype import beartype
from contextlib import nullcontext
from . import utils, mcmc
from .lsd import LSD
from .result import Result
from .data import Data, Config, MaskingLines, LineList
from .errors import ContinuumError
from .utils import IntLike, Scalar, Array1D, Array2D
from astropy.stats.sigma_clipping import sigma_clip

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
        data : Data | None = None,
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
        :py:attr:`ACID_code.Config.defaults` (returning a dictionary of defaults for both initialisation and the ACID method).

        All parameters below and in the ACID method are stored in the :py:class:`Config` instance, unless explicitly stated to be in the :py:class:`Data` instance.
        The :py:class:`Config` instance is for runtime settings and the :py:class:`Data` instance is for storing data and any calculations. 

        Parameters
        ----------
        data : :py:type:`Data | None`, optional
            A :py:class:`Data` instance to store the data and calculations in. 
        **kwargs : :py:type:`dict`, optional
            Any kwargs that can be passed to :py:func:`Acid.ACID` can also be passed here.

        Raises
        ------
        BeartypeError
            See :ref:`type_validation` to understand input validation errors.
        """
        # Initialise the data class to store calculations in ACID
        if data is not None:
            self.data = data
        else:
            self.data = Data() # generates also a config on initilisation

        config = kwargs.pop("config", None)
        if config is not None:
            self.data.config = config

        # data.config was either empty (on Data initialisation above) or had a previous config stored in the input
        self.config = self.data.config
        self.config.update_hipri(**{name: kwargs[name] for name in Config.defaults if name in kwargs})

        # Store the init kwargs to be handled on ACID call
        self.init_kwargs = kwargs

    def ACID(
        self,
        wavelengths           : Array1D|Array2D|None           = None, # Data
        flux                  : Array1D|Array2D|None           = None, # Data
        errors                : Array1D|Array2D|None           = None, # Data
        sn                    : Array1D|Array2D|Scalar|None    = None, # Data
        velocities            : Array1D|None                   = None, # Data
        linelist              : Array2D|str|LineList|dict|None = None, # Data
        order                 : IntLike|None                   = None, # Config
        order_range           : Array1D|None                   = None, # Config
        verbose               : IntLike|bool|str|None          = None, # Config
        sampler_progress      : bool|None                      = None, # Config
        masking_lines         : dict|MaskingLines|None         = None, # Config
        seed                  : IntLike|None                   = None, # Config
        dir                   : str|None                       = None, # Config
        save_path             : str|None                       = None, # Config
        sampler_path          : str|None                       = None, # Config
        figure_dir            : str|None                       = None, # Config
        deterministic_profile : bool|None                      = None, # Config
        poly_ord              : IntLike|None                   = None, # Config
        continuum_percentile  : IntLike|None                   = None, # Config
        n_bins                : IntLike|None                   = None, # Config
        bin_size              : IntLike|None                   = None, # Config
        pix_chunk             : IntLike|None                   = None, # Config
        dev_perc              : IntLike|None                   = None, # Config
        sigma_lower           : Scalar|None                    = None, # Config
        sigma_upper           : Scalar|None                    = None, # Config
        skips                 : IntLike|None                   = None, # Config
        od                    : bool|None                      = None, # Config
        sparse                : bool|None                      = None, # Config
        profile_groups        : Array1D|Array2D|None           = None, # Config, then Data after LSD clipping/grouping
        depth_group_rules     : dict|None                      = None, # Config
        sampler_type          : str|None                       = None, # Config
        parallel              : bool|None                      = None, # Config
        cores                 : IntLike|None                   = None, # Config
        nwalkers              : IntLike|None                   = None, # Config, then Data just before MCMC
        nsteps                : IntLike|None                   = None, # Config as the initial steps, Data.nsteps is the true count of steps taken, which can be higher
        max_steps             : IntLike|None                   = None, # Config
        check_interval        : IntLike|None                   = None, # Config
        min_checks            : IntLike|None                   = None, # Config
        min_tau_factor        : IntLike|None                   = None, # Config
        tau_tol               : float|None                     = None, # Config
        moves                 : list|None                      = None, # Config
        continuum_method      : str|None                       = None, # Config
        run_mcmc              : bool|None                      = None, # Config
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
        velocities : :py:type:`Array1D`, optional
            Velocity grid for LSD profiles (in km/s). For example, use: np.arange(-25, 25, 0.82) to create one. If None, a default grid
            from -25 to 25 km/s is used with a spacing calculated by calc_deltav after the wavelengths are provided. It is highly recommended to 
            choose your own velocity grid, by default None, stored in the Data instance.
        linelist : :py:type:`Array2D | str` | :py:class:`LineList` | dict`, optional
            The linelist to use for LSD. The linelist should have wavelengths in angstroms and relative depths between 0 and 1.
            This is a required parameter. It can be of the forms:
            - String: A path to a VALD linelist in string format. Support for other linelists may be added in the future or on request.
            - :py:type:`Array2D`: A 2D array-like object indexed such that 0 is wavelengths and 1 is depths.
            - dict: A dictionary with keys "wavelengths" and "depths", each containing array-like objects for the wavelengths and depths respectively.
            - :py:class:`LineList`: The :py:class:`LineList` class is used to expose the linelist for masking or getting/plotting the linelist. You can input an instance if you have one.
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
            4: Debugging mode, printing all information and saving all internal variables to the Data instance for debugging. Not recommended; this will take more memory and space.
            The possible input types are described below:
            - Integer: Must be between 0 and 4, corresponding to the verbosities described above.
            - Boolean: If True, defaults to 2. If False, defaults to 0.
            - String: Can be one of ["none", "low", "medium", "high", "debug"] or their common variants.
            By default 2 (medium).
        sampler_progress : :py:type:`bool`, optional
            A verbosity override for just the MCMC sampling progress.
            By default None which does not override, but if True/False, it will overwrite with that value, and use/don't use a tqdm output for the sampler.
        masking_lines : :py:type:`dict` | :py:class:`MaskingLines`, optional
            Telluric lines (in angstroms) and widths in (km/s) to mask from the wavelength regions from. Unless you'd like to change the default masking
            lines, we recommend just using the defaults (leaving this as None), which are based on telluric lines and strong hydrogen/metal lines in the 
            optical and near infrared. For a guide on using your own/modifying the defaults, see :ref:`masking_lines`. By default None, stored in the Config instance.
        seed : :py:type:`IntLike`, optional
            Random seed for reproducibility, leave it on None for a random seed, by default None.
        dir : :py:type:`str`, optional
            Sets the save_path to dir/data.pkl, the sampler_path to dir/sampler.h5, and figure_dir to dir/figures/.
            Any inputted paths for save_path, sampler_path, or figure_dir will override this input.
            If None, the save_path, sampler_path, and figure_dir are not set. By default None.
            If the directory does not exist, only its final component is created; its parent must already exist.
        save_path : :py:type:`str`, optional
            The path to save the data instance (containing the results) to. If None, results are not saved to disk, by default None.
            If a string is input, the data instance will be saved to this path as a .pkl file when the results are finished.
            Should be a valid file path that ends with ".pkl". If its direct parent does not exist, that final directory is created.
            The parent's parent must already exist.
            If a file already exists at this path, it will be overwritten on Acid initialization.
            Note that we separate the save and sampler paths, as the sampler can be very large and may not be desired to be saved.
        sampler_path : :py:type:`str`, optional
            The path to save the sampler HDF5 backend file to.
            If None, the sampler is not saved and only stored in memory. By default None.
            Note that if your path points to an existing file, it will be overwritten on Acid initialization.
            If existing, we use the emcee HDF5 backend to store and load the sampler.
            Should be a valid file path that ends with ".h5". If its direct parent does not exist, that final directory is created.
            The parent's parent must already exist.
            Note that if you later try and save the sampler through the data class, it is converted to a HDF5 backend.
        figure_dir : :py:type:`str`, optional
            A directory to save the figures to.
            If None, figures are not saved to disk and figures are instead shown (if asked) with plt.show(), by default None.
            If the directory does not exist, only its final component is created; its parent must already exist.
        deterministic_profile : bool, optional
            If True, fits both the continuum and the LSD profile simultaneously. If False, only fits the continuum in mcmc, the
            profile is inferred from the continuum fit. This is a new feature that has been set to the default as it significantly
            decrease convergence time and computation time per step, while fully maintaining accuracy. Setting this to False will 
            match legacy behaviour, by default True.
        poly_ord : :py:type:`IntLike`, optional
            Order of polynomial to fit as the continuum, by default 3
        continuum_percentile : :py:type:`IntLike`, optional
            The percentile to use when fitting the continuum, by default 99. For example, if 99, the continuum fit will be performed
            on the points in the spectra that are above the 99th percentile in flux in each spectral bin (determined by n_bins/bin_size below).
        n_bins : :py:type:`IntLike`, optional
            The number of bins to use when performing the continuum fit. The spectra are evenly split into this many bins and the 
            continuum is fit to the median wavelength and the specified percentile (continuum_percentile) of flux in each bin.
            By default 20.
        bin_size : :py:type:`IntLike`, optional
            Instead of specifying the total number of bins in your spectrum (nbins), specify the number of pixels to go in each bin.
            The spectra are split into bins with this number of pixels, and the continuum is fit to the median wavelength 
            and the specified percentile of flux in each bin. If a value is input it will override n_bins. By default None.
        pix_chunk : :py:type:`IntLike`, optional
            Size of 'bad' regions in pixels. 'bad' areas are identified by the residuals between an inital model
            and the data. If the residuals deviate by a specified percentage (see dev_perc below) for this number (pix_chunk) of pixels,
            then this chunk of pixels are masked in the spectra. By default 20.
        dev_perc : :py:type:`IntLike`, optional
            Allowed deviation percentage. 'bad' areas are identified by the residuals between an inital model
            and the data. If a residual deviates by this percentage for a specified number of pixels,
            then this chunk of pixels are masked in the spectra. By default 25
        sigma_lower : :py:type:`IntLike`, optional
            Number of sigma to keep in sigma clipping. Ill fitting lines are identified by sigma-clipping the
            residuals between an inital model and the data (with masking_lines already removed).
            Regions that lie outside the median - sigma_lower STDEVs are clipped.
            The clipped regions will be masked in the spectra.
            This masking is only applied to find the MCMC continuum fit and is removed when
            LSD is applied to obtain the final profiles, by default 3
        sigma_upper : :py:type:`IntLike`, optional
            Number of sigma to keep in sigma clipping. Ill fitting lines are identified by sigma-clipping the
            residuals between an inital model and the data (with masking_lines already removed).
            Regions that lie outside the median + sigma_upper STDEVs are clipped.
            The clipped regions will be masked in the spectra.
            This masking is only applied to find the MCMC continuum fit and is removed when
            LSD is applied to obtain the final profiles, by default 5.
            The default is higher to allow ACID to find a better continuum fit in MCMC fitting (try reducing sigma_lower yourself!).
        skips : :py:type:`IntLike`, optional
            An option to only run acid on one in every n pixels, where n is the integer argument. This is only useful for
            testing to get a quicker result especially for larger wavelength ranges or datasets, by default 1 (no skipping)
        od : :py:type:`bool`, optional
            If True, runs ACID in optical depth, otherwise, the LSD methods and ACID fitting is performed in flux. By default None which defaults to True.
            Note that the whole point of ACID is to run LSD in OD, we highly recommend leaving this unless you specifically want to compare.
        sparse : bool, optional
            Whether to use "sparse" matrix calculations for the alpha matrix in the LSD class, by default True. If you set it to false it will use 
            the legacy method for alpha calculation which is slower and more memory intensive to no real benefit.
            It is kept mainly for testing. Note also that we are not acutally calculating a sparse matrix, instead,
            we are only calculating the contributions of the nearest neighbour velocity bins and setting the rest to 0.
            For sparse=False, it calculates the entire alpha matrix with an efficient numpy method.
        profile_groups : :py:type:`Array1D | None`, optional
            A mask for the linelist elements indicating which group they belong to. Each group is fitted with their own profiles.
            If provided, the resulting profiles will be a 2D array in with the same order as the index of the group.
            The groups should be 0 indexed, e.g. [0,0,2,0,1,0,3,...]. The shape must match the inputted linelist.
            If None, then just one profile is generated (as if the mask was [0,0,0,0,0,...]). By default None.
        depth_group_rules : :py:type:`dict`, optional
            A way to automatically generate the profile_groups based on a set of rules.
            This was set up as a way to seperate the the linelist into groups of similar depth.
            The dictionary must have the following two keys:

            - "n_groups" (int), total number of depth groups
            - "min_lines" (int), the minimum number of lines that are required in each group, we recommend at least 20 for a sensible profile.
            
            If just those two keys were put, then the program will fill the 4 groups with the same number of lines. However, you can additionally
            specify the depth of lines for each group starting from the deepest group. Here is an example:
            
            depth_group_rules = {
                "n_groups": 4,
                "min_lines": 20,
                "0": 0.8,
                "1": 0.5,
            }

            This will mean that the deepest group, 0, will contain all lines from 1-0.8, or lower if that range does not contain min_lines.
            The next group will contain all lines from 0.8-0.5, or lower if not filled with min_lines. The rest of the unspecified groups
            are filled evenly with the remaining lines.
            Any additional dictionary keys should corresond to integer indices and float absorption depths as shown in the example above.
            The groups will only be generated with this ruleset after the linelist wavelength ranges and S/N cuts have been applied.
            Note that providing profile_groups will override this parameter, causing it to have no effect.
        sampler_type : :py:type:`str`, optional
            If you really try to wish to use the dynesty nested sampler, you can set this to "dynesty". It is almost entirely unsupported
            by the rest of the code other than to just get a finished result object, and much slower. We highly recommend using None or "emcee" (default).
            The only reason I added this was to get the Bayesian evidence for model comparison.
            If "dynesty" is chosen, the dynesty package needs to be installed, and the nsteps parameter is treated as "nlive" to be passed to the NestedSampler.
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
        continuum_method : :py:type:`str`, optional
            The method to use for fitting and evaluating the continuum. Options are "polyval" or "chebval". Default is None,
            which uses polyval for poly_ord <= 5 and chebval for poly_ord > 5.
            This is because chebyshev polynomials are more numerically stable for higher order polynomials.
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

        # Part 1: Setup and validation
        # ============================
        # region setup and validation
        # region verbose
        # Check if verbose was put in the init
        init_verbose = self.init_kwargs.pop("verbose", None)
        # Set verbosity first with validation handled in config property setter
        self.config.verbose = verbose if verbose is not None else init_verbose

        # Suppress warnings generally, but high verbosity will show them
        if self.config.verbose <= 2:
            warnings.filterwarnings("ignore")

        # Print initialisation status
        init_t0 = time.time()
        if self.config.verbose >= 2:
            print('Initialising...')
        # endregion verbose


        # Intercept legacy kwargs and invalid inputs
        # ------------------------------------------
        # region input kwargs
        # Add init_kwargs to kwargs, with kwargs overwriting
        kwargs = {**self.init_kwargs, **kwargs}

        # Catch for the linelist_path, linelist_wl, or linelist_depths arguments, which was old way to input a linelist
        if "linelist_path" in kwargs:
            legacy_linelist = kwargs.pop("linelist_path")
            if linelist is None and "linelist" not in kwargs:
                linelist = legacy_linelist
            if self.config.verbose >= 1:
                print("Warning: 'linelist_path' is a legacy argument for inputting a linelist, " \
                f"please use 'linelist' instead.\n The 'linelist_path' argument does not support full input validation.")
        if "linelist_wl" in kwargs or "linelist_depths" in kwargs:
            raise ValueError("The 'linelist_wl' and 'linelist_depths' arguments are legacy linelist arguments, use 'linelist' instead.\n" \
                             "If your linelist wl and depths are two 1D arrays, you can use linelist=np.array([wl, depths]) for the correct format.")

        # Check for old n_sig input
        if "n_sig" in kwargs:
            legacy_n_sig = kwargs.pop("n_sig")
            if sigma_lower is None and "sigma_lower" not in kwargs:
                sigma_lower = legacy_n_sig
            if self.config.verbose >= 1:
                print("Warning: 'n_sig' is a legacy argument for inputting sigma_lower.\n" \
                f"Please use 'sigma_lower' and 'sigma_upper' to configure the sigma range instead.")

        # Check for old _all_frames input
        if "_all_frames" in kwargs:
            _all_frames = kwargs.pop("_all_frames")
            if self.config.verbose >= 0:
                print("Warning: 'all_frames' is a legacy argument and is now unused. See 'DataList' in the documentation for running multiple orders.")

        # Check for telluric_lines old input
        if "telluric_lines" in kwargs:
            telluric_lines = kwargs.pop("telluric_lines")
            if self.config.verbose >= 0:
                print("Warning: 'telluric_lines' is a legacy argument and now forms part of the broader 'masking_lines' argument.\n" \
                "See 'MaskingLines' in the documentation for more information.\n"
                "The telluric_lines will be ignored and default masking_lines will be used instead.")

        # Check data or config wasnt passed
        if "data" in kwargs:
            raise ValueError("The 'data' kwarg should be passed in initialisation, please remove it from the ACID method call.")
        if "config" in kwargs:
            raise ValueError("The 'config' kwarg is stored in Data.config.\n" \
                                "Set Data.config to your desired config instance and pass it in Acid initialisation.")
        # The remaining kwargs are either valid and passed in init, or invalid in either init or ACID kwargs
        # endregion input kwargs


        # Validating config inputs
        # ------------------------
        # region config validation
        local_kwargs = locals().copy()

        # Assign inputted configuration to config dictionary, preferring ACID inputs over init inputs
        config_kwargs = {}
        for name in Config.defaults:
            if name in local_kwargs:
                init_input = kwargs.pop(name, None)
                config_kwargs[name] = local_kwargs[name] if local_kwargs[name] is not None else init_input

        old_profile_groups = self.config.profile_groups

        # Update config if any of the above config settings are new
        self.config.update_hipri(**config_kwargs) # self.config overwrites config_kwargs if overlapping
        if config_kwargs.get("profile_groups") is not None and (
            old_profile_groups is None
            or not np.array_equal(old_profile_groups, self.config.profile_groups)
        ):
            self.data.reset()

        # Then also remove the valid data kwargs, preferring ACID inputs over init inputs
        valid_data_kwargs = ["wavelengths", "flux", "errors", "sn", "velocities", "linelist"]
        data_kwargs = {}
        for name in valid_data_kwargs:
            init_input = kwargs.pop(name, None)
            data_kwargs[name] = local_kwargs[name] if local_kwargs[name] is not None else init_input
        # Then set to locals to be used downstream
        wavelengths, flux, errors, sn, velocities, linelist = (data_kwargs[name] for name in valid_data_kwargs)

        # Finally those that remain in kwargs are invalid and raise an error
        if kwargs:
            raise ValueError(f"Unexpected keyword argument(s) for Acid.ACID: {', '.join(sorted(kwargs))}")
        self.init_kwargs = {}

        # TODO: Keep for now, move to config if no mp fix, otherwise try spawn mp context
        if self.config.parallel and sys.platform == "win32":
            if self.config.verbose >= 1:
                # This doesn't work, needs serious modifications to make work, so just run serially for now
                print("Parallel MCMC on Windows is not currently supported. Running MCMC serially.")
            self.config.parallel = False

        # TODO: Apply seed here (only if complete=False, or run_mcmc=False), maybe even save the generator state before mcmc
        # endregion config validation
        # endregion setup and validation

        # Part 2: Preprocessing
        # ---------------------
        # Setup and data validation done in data class and applies skips, also combines frames if multiple frames were input
        # Sets the "input" and "combined" keys in the data instance for wavelengths, flux, errors, and sn
        self.data.set_inputs(wavelengths, flux, errors, sn)

        # Let the respective properties in Data handle the validation and setting, this is set after set_inputs so velocities 
        # can be guessed from them if not input
        self.data.linelist = linelist
        self.data.velocities = velocities

        # Get the line masking before initial fit to avoid ill-fitting lines biasing the continuum fit
        self.data.line_mask = self.config.masking_lines.get_1d_mask_on_grid(self.data.wavelengths["combined"])

        # Prepare the "initial" keys, this is just the combined key, except the errors have masked out the masking lines.
        # They are also used in the final step as these are the only regions masked in the final step
        self.data.errors["initial"] = np.where(self.data.line_mask, 1e12, self.data.errors["combined"])
        self.data.wavelengths["initial"] = self.data.wavelengths["combined"]
        self.data.flux["initial"] = self.data.flux["combined"]
        self.data.sn["initial"] = self.data.sn["combined"]

        # Check if the initial continuum fit and LSD run has been performed
        if all((
            # We only bother to check for one of these keys generated in the scipy_continuum_fit and LSD runs
            "initial" in self.data.poly_coeffs,
            "initial" in self.data.alpha,
        )):
            if self.config.verbose >= 3:
                print("Initial fit and LSD run already exists, skipping this step.")
        else:
            if self.config.verbose >= 2:
                print("Performing initial fit and LSD...")

            # Uses all information stored in data, accessing and storing the data attributed with the key
            self.scipy_continuum_fit(self.data, key="initial")
            _lsd = LSD.runlsd_and_store(self.data, key="initial", return_cls=True)

            # Save lsd dict if debugging mode
            if self.config.verbose == 4:
                self.data.debug["lsd_initial"] = _lsd.__dict__

            _lsd = None # discard to save memory

        # Masking based off residuals
        if all((
            # Again we only need to check if some of the keys have been made, not all of them
            "masked" in self.data.wavelengths,
            "mcmc" in self.data.c_factor,
        )):
            if self.config.verbose >= 2:
                print("Residual masks already exists, skipping residual masking step.")
        else:
            if self.config.verbose >= 2:
                print('Residual masking...')

            # Use the initial LSD run to get the scaled residuals
            residuals = self.data.residuals["initial"]
            
            # Masking pixel chunks based on deviation from residuals
            # -----------------------------------------------
            # Get bad pixels that deviate by a percentage greater than dev_perc on the full residuals
            bad_idx = np.zeros_like(residuals, dtype=bool)
            unmasked = ~self.data.line_mask
            bad_idx[unmasked] = (np.abs(residuals[unmasked]) > (self.config.dev_perc / 100))

            # A trick to get the mask for continuous regions of bad pixels, by padding the bad_idx 
            # with False on both sides and finding the start and end indices of the True regions
            padded = np.concatenate(([False], bad_idx, [False]))
            starts = np.flatnonzero(~padded[:-1] & padded[1:])
            ends = np.flatnonzero(padded[:-1] & ~padded[1:])
            pix_mask = np.zeros_like(residuals, dtype=bool)

            # Then make pix_mask for regions that are greater than pix_chunk in length
            for start, end in zip(starts, ends):
                if (end - start) >= self.config.pix_chunk:
                    pix_mask[start:end] = True
            self.data.pix_mask = pix_mask # Save the pix_mask for later use in plotting and analysis

            # Sigma Clipping
            # --------------
            # Use astropy's iterative sigma clipping, only sigma clip residuals that are not already line masked
            masked_residuals = residuals[~self.data.line_mask] # so that we can get the std on the masked residuals

            # Use the iterative sigma clipping in astropy, returning a masked array of clipped residuals
            result, lower_clip, upper_clip = sigma_clip(
                masked_residuals,
                sigma_lower=self.config.sigma_lower,
                sigma_upper=self.config.sigma_upper,
                return_bounds=True
            )

            # Put the sigma mask back onto the full pixel grid
            sigma_mask = np.zeros_like(residuals, dtype=bool)
            sigma_mask[unmasked] = np.ma.getmaskarray(result)

            self.data.sigma_mask = sigma_mask

            # Combine all masks
            self.data.full_mask = pix_mask | sigma_mask | self.data.line_mask
            
            # Warn if more than 50% of spectrum is masked this way
            if np.sum(self.data.full_mask) > 0.5 * len(self.data.full_mask):
                if self.config.verbose >= 1:
                    print(f"Warning: More than 50% of the spectrum is masked. \n" \
                    "Please check your initial continuum fit and masking (by using verbose=3 when initialising). \n" \
                    "If you are aware that you have bad spectra, then this can be ignored.")

            # Apply a error mask onto just y for the continuum fit and LSD call, later we fully remove them with the full mask for fitting
            self.data.errors["masked"]      = np.where(self.data.full_mask, 1e12, self.data.errors["combined"])
            self.data.wavelengths["masked"] = self.data.wavelengths["combined"]
            self.data.flux["masked"]        = self.data.flux["combined"]
            self.data.sn["masked"]          = self.data.sn["combined"]

            # We can also skip alpha recalculation as it is unchanged
            self.data.alpha["masked"] = self.data.alpha["initial"]


            # Second Continuum Fit and LSD run with new masked errors
            # -------------------------------------------------------
            # Now do another continuum fit with masked yerr, continuumfit removes high error points from the fit
            self.scipy_continuum_fit(self.data, key="masked")
            lsd = LSD.runlsd_and_store(self.data, key="masked", return_cls=True)

            # Save lsd dict if debugging mode
            if self.config.verbose == 4:
                self.data.debug["lsd_masked"] = lsd.__dict__


            # Applying Residual Masks to the Data for Fitting
            #------------------------------------------------
            # First apply to the flattened alpha, and then bin the lsd class to save memory.
            # The flatted alpha mechanic is important for multi-profile LSD, otherwise alpha_flat is the same as alpha
            # Slicing alpha like this avoids a recalculation because we know which wavelengths are masked
            self.data.alpha["mcmc"] = lsd.alpha_flat[~self.data.full_mask, :]

            lsd = None # Discard to save memory once the alpha is sliced

            # Apply to the rest of the data
            self.data.wavelengths["mcmc"] = self.data.wavelengths["combined"][~self.data.full_mask]
            self.data.flux["mcmc"]        = self.data.flux["combined"][~self.data.full_mask]
            self.data.errors["mcmc"]      = self.data.errors["combined"][~self.data.full_mask]
            self.data.sn["mcmc"]          = self.data.sn["combined"] # no change as its single valued per frame
            # Normalisation occurs on the full grid, then select masked wavelengths
            self.data.norm_wavelengths["mcmc"] = utils.normalize_wavelengths(self.data.wavelengths["combined"])[~self.data.full_mask]

            # For the Cholesky factor, we need to recalculate them on the new wavelength grid, and convert to OD if needed
            _, errors = utils.flux_to_od(self.data.flux["mcmc"], self.data.errors["mcmc"], od=self.config.od) # only need errors for c_factor
            self.data.c_factor["mcmc"] = LSD.calc_cholesky(alpha=self.data.alpha["mcmc"], error=errors)

            # Save extra variables for plotting in the Data class
            if "masked" not in self.data.plotting_variables:
                self.data.plotting_variables["masked"] = {}
            self.data.plotting_variables["masked"]["residuals"]        = residuals
            self.data.plotting_variables["masked"]["masked_residuals"] = masked_residuals
            self.data.plotting_variables["masked"]["lower_clip"]       = lower_clip
            self.data.plotting_variables["masked"]["upper_clip"]       = upper_clip
            if self.config.verbose >= 3: # Plot now if verbose enough
                self.data.plot_residual_masking()

        # ACID Initialialised
        # -------------------
        self.data.setup_time += time.time() - init_t0
        mcmc_t0 = time.time()
        if self.config.verbose >= 2:
            print('Initialised in %ss'%round((self.data.setup_time), 3))
        if self.config.verbose >= 3:
            print('State of Data before MCMC run:')
            print(self.data) # use the __repr__ method of the Data class to print a nice summary

        # Prepare and Run MCMC
        #----------------------
        # Get the initial state from all of the above calculated data
        self.data.initial_state = self.get_initial_state()

        # Run MCMC if requested
        if self.config.run_mcmc is True:
            # Default run for just nsteps steps
            if self.config.max_steps is None:
                if self.config.verbose >= 2:
                    print("Running MCMC for %s steps..."%self.config.nsteps)
                self.run_mcmc(self.config.nsteps, self.data.initial_state)
                if self.config.sampler_type == "emcee":
                    self.data.nsteps = self.sampler.backend.iteration
                else:
                    self.data.nsteps = self.config.nsteps

            # Else use max_steps path
            else:
                if self.config.verbose >= 2:
                    print(f"Running MCMC with a maximum of {self.config.max_steps} steps or until convergence is reached...")

                self.run_mcmc_until_converged(self.config.max_steps, state=self.data.initial_state)
                self.data.nsteps = self.sampler.backend.iteration

            self.data.mcmc_time += time.time() - mcmc_t0

            if self.config.verbose >= 2:
                print('MCMC finished after %ss'%(round(self.data.mcmc_time, 3)))

            return Result(self)

        else:
            if self.config.verbose >= 1:
                print("MCMC not run, returning None. Class attributes have been updated.")
            return None

    def ACID_HARPS(self, *args, **kwargs):
        """
        This method is no longer supported in ACID. Please use the ACID function with the appropriate inputs for HARPS spectra instead. 
        Future versions of ACID may provide functions to load and configure data from a range of different standard instruments. 
        """
        # TODO: ACID HARPS raises NotImplementedError
        raise NotImplementedError(f"ACID_HARPS is no longer supported in ACID. \n"
        f"Please use the ACID function with the appropriate inputs for HARPS spectra instead. \n"
        f"Future versions of ACID may provide functions to load and configure data from a range of different standard instruments.")

    @staticmethod
    def scipy_continuum_fit(data:Data, key:str) -> None:
        """
        Fits the continuum of a spectrum using the specified order and method.
        """
        config = data.config
        wavelengths = data.wavelengths[key]
        fluxes = data.flux[key]
        errors = data.errors[key]

        # Normalise wavelengths
        unnormalized_wavelengths = wavelengths
        norm_wavelengths = utils.normalize_wavelengths(wavelengths)
        data.norm_wavelengths[key] = norm_wavelengths

        # Sort to ensure smooth binning and fitting
        idx = np.argsort(norm_wavelengths)
        w = norm_wavelengths[idx]
        f = fluxes[idx]
        e = errors[idx]

        # Get bin size. Explicit bin_size overrides n_bins.
        if config.bin_size is not None:
            binsize = config.bin_size
        else:
            binsize = max(1, len(w) // config.n_bins)

        # Get binsize, reshape into 2D array of bins
        n = len(w) // binsize  # full bins only
        w2 = w[:n*binsize].reshape(n, binsize)
        f2 = f[:n*binsize].reshape(n, binsize)
        e2 = e[:n*binsize].reshape(n, binsize)

        # Get the median wavelength, specified percentile flux, and median error in each bin
        clipped_flux = np.nanpercentile(f2, config.continuum_percentile, axis=1)
        clipped_waves = np.nanmedian(w2, axis=1)
        clipped_errs = np.nanmedian(e2, axis=1)

        # # Also add as a safeguard an extra point for high orders at start and end of spectrum to avoid edge effects in high order fits
        if config.poly_ord > 5 and config.continuum_method == "polyval":
            max_wl_idx = np.nanargmax(w)
            min_wl_idx = np.nanargmin(w)
            clipped_waves = np.concatenate(([w[min_wl_idx]], clipped_waves, [w[max_wl_idx]]))
            clipped_flux = np.concatenate(([f[min_wl_idx]], clipped_flux, [f[max_wl_idx]]))
            clipped_errs = np.concatenate(([e[min_wl_idx]], clipped_errs, [e[max_wl_idx]]))

        # Remove bad points for the polynomial fit, defined as non-finite values or errors that are non-positive or above 1e11
        good = (
            np.isfinite(clipped_waves)
            & np.isfinite(clipped_flux)
            & np.isfinite(clipped_errs)
            & (clipped_errs > 0)
            & (clipped_errs < 1e11) # 1e12 is the default mask error value, which can be picked up in the median error binning
        )

        # Check if there are enough good points for the polynomial fit
        if np.sum(good) < config.poly_ord + 1:
            raise ValueError("Insufficient good points for polynomial fit. "
                             "Consider reducing the polynomial order or adjusting the masking.")

        # Fit with MCMC to get the polynomial coefficients and evaluate the continuum fit
        # The fitting and had different methods added, so they've been moved to their own function
        data.poly_coeffs[key] = utils.fit_continuum(clipped_waves[good], clipped_flux[good], config.poly_ord, method=config.continuum_method, w=1/clipped_errs[good])
        data.continuum[key] = utils.eval_continuum(norm_wavelengths, data.poly_coeffs[key], method=config.continuum_method)

        # Get the model fitted flux and errors from the fit
        data.fitted_flux[key] = fluxes / data.continuum[key]
        data.fitted_errors[key] = errors / data.continuum[key]

        # Save to Data the required variables for the plot
        if key not in data.plotting_variables:
            data.plotting_variables[key] = {}
        data.plotting_variables[key]["clipped_waves"]            = clipped_waves
        data.plotting_variables[key]["clipped_flux"]             = clipped_flux
        data.plotting_variables[key]["good"]                     = good
        if config.verbose >= 3:
                data.plot_continuum_fit(key=key)

        if np.any(data.fitted_flux[key][~data.line_mask] <= 0) or np.any(data.fitted_errors[key][~data.line_mask] <= 0):
            error = ContinuumError("Continuum fit resulted in non-positive flux or errors, which is not physical.\n " \
            "Consider adjusting the polynomial order. Use verbose=3 to see the plot of the continuum fit.\n " \
            "Note that this will only work for interactive terminals or displays which work with plt.show()")
            data.exception = error
            data.traceback = traceback.format_stack()
            raise error

        return

    def get_initial_state(self) -> np.ndarray|None:
    
        # Set rng seed off of config seed if desired, otherwise default config seed is None and rng will be random
        rng = np.random.default_rng(self.config.seed)

        n_profile_params = self.data.alpha["mcmc"].shape[1]

        # Set the number of dimensions, add the no. of velocity points if also fitting the profile
        self.data.ndim = self.config.poly_ord + 1
        if not self.config.deterministic_profile:
            self.data.ndim += n_profile_params

        # emcee recommendation is 3 times the number of dimensions (we add a +3 buffer as well), but can be overridden by user input
        self.data.nwalkers = 3 + self.data.ndim * 3 if self.config.nwalkers is None else self.config.nwalkers

        if self.config.sampler_type == "emcee":
            theta0 = self.data.poly_coeffs["masked"]

            if not self.config.deterministic_profile:
                profile0 = np.asarray(self.data.profile["masked"][0]).reshape(-1)
                theta0 = np.concatenate((profile0, theta0))

            # Test to see if the starting position is valid, up to a max of 1000 attempts
            test_mcmc = mcmc.MCMC(self.data)
            width = np.maximum(0.5 * np.abs(theta0), 1e-3)
            walkers = []
            max_attempts = 1000
            n_attempt = 0
            while len(walkers) < self.data.nwalkers:
                theta = rng.normal(theta0, width)
                if np.isfinite(test_mcmc.log_probability(theta)):
                    walkers.append(theta)

                n_attempt += 1
                if n_attempt == max_attempts:
                    raise RuntimeError("Reached the max number of attempts for finding an initial state for MCMC walkers.")

            initial_state = np.array(walkers)
        else:
            initial_state = None

        return initial_state

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
        if self.config.sampler_type == "emcee":
            sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps, state)
        sampler_verbosity = True if self.config.verbose >= 2 else False
        sampler_verbosity = self.config.sampler_progress if self.config.sampler_progress is not None else sampler_verbosity
        pool_context = nullcontext(None)

        if self.config.parallel:
            utils.configure_mp_environ(os) # Raises error is not configured correctly, otherwise does nothing

            if self.config.verbose >= 2:
                print(f"Using {self.config.cores} cores for MCMC")
            
            ctx = mp.get_context("fork")
            pool_context = ctx.Pool(processes=self.config.cores, initializer=mcmc._mp_init_worker, initargs=(self.data,))
            log_prob = mcmc._mp_log_probability if self.config.sampler_type == "emcee" else mcmc._mp_log_likelihood
            ptform = mcmc._mp_ptform
            queue_size = os.cpu_count()
        else:
            MCMC = mcmc.MCMC(self.data)
            log_prob = MCMC if self.config.sampler_type == "emcee" else MCMC.dynesty_logprob
            ptform = MCMC.ptform
            queue_size = None

        with pool_context as pool:
            if self.config.sampler_type == "emcee":
                self.sampler = EnsembleSampler(log_prob_fn=log_prob, pool=pool, **sampler_kwargs)
                self.sampler.run_mcmc(**mcmc_kwargs)
            else:
                import dynesty # we have already checked if the user can import dynesty in the config property setter
                if self.config.parallel:
                    pool.size = self.config.cores
                self.sampler = dynesty.NestedSampler(log_prob, ptform, self.data.ndim, self.config.nsteps, pool=pool, queue_size=queue_size)
                self.sampler.run_nested(print_progress=sampler_verbosity)

    def run_mcmc_until_converged(self, max_steps:IntLike, state=None) -> None:
        """
        Runs MCMC until convergence is reached. A purely class method that I do not recommend you use directly. Use
        Acid.ACID(run_mcmc=True) to run MCMC for the first pass if not already done, which will skip already performed calculations.
        Otherwise, use Acid.continue_sampling or Result.continue_sampling if you have already run MCMC and want to continue.
        """
        # Get sampler and stopping criterion kwargs for the first run based on initial state, then update nsteps in mcmc_kwargs for subsequent runs
        sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps=self.config.check_interval, state=state)
        stopping_criterion_args = (self.config.min_checks, self.config.min_tau_factor, self.config.tau_tol)

        # Set variables to be updated within the convergence loop
        step_number = 0
        tau_list = []
        max_samples = int(np.ceil(max_steps / self.config.check_interval))
        last_tolerance = np.inf
        last_neff = 0
        condition = False
        pool_context = nullcontext(None)

        if self.config.parallel:
            utils.configure_mp_environ(os)

            if self.config.verbose >= 2:
                print(f"Using {self.config.cores} cores for MCMC")

            ctx = mp.get_context("fork")
            pool_context = ctx.Pool(processes=self.config.cores, initializer=mcmc._mp_init_worker, initargs=(self.data,))
            log_prob_fn = mcmc._mp_log_probability
        else:
            log_prob_fn = mcmc.MCMC(self.data)

        with pool_context as pool:
            self.sampler = EnsembleSampler(**sampler_kwargs, pool=pool, log_prob_fn=log_prob_fn)

            for i in range(max_samples):
                steps_this_run = min(self.config.check_interval, max_steps-step_number)
                mcmc_kwargs["nsteps"] = steps_this_run

                tol_str, neff_str = mcmc.MCMC._get_tqdm_desc(last_tolerance, last_neff, self.config)
                mcmc_kwargs["progress_kwargs"] = {"desc": f"Iteration {i+1}/{max_samples}, last tolerance: {tol_str}, neff: {neff_str}"}
                self.sampler.run_mcmc(**mcmc_kwargs, skip_initial_state_check=True)
                mcmc_kwargs["initial_state"] = None

                step_number += steps_this_run

                # We want to keep the time for get_autocorr_time to run constant, so thin accordingly
                # It scales with the number of steps, so thin by the number of steps taken divided by 
                # the check interval to keep the same number of samples for get_autocorr_time to process.
                try:
                    with open(os.devnull, "w") as devnull, \
                        contextlib.redirect_stdout(devnull), \
                        contextlib.redirect_stderr(devnull):
                        tau = self.sampler.get_autocorr_time(tol=0, thin=max(1, self.sampler.backend.iteration//self.config.check_interval))
                except emcee.autocorr.AutocorrError:
                    continue
                tau_list.append(tau)

                # The stopping criterion function below handles the logic for determining stopping condition
                total_step_number = self.sampler.backend.iteration
                condition, last_tolerance, last_neff = mcmc.MCMC._get_mcmc_stopping_criterion(tau_list, total_step_number, *stopping_criterion_args)

                if condition is True:
                    if self.config.verbose >= 2:
                        print(f"Converged at step {total_step_number}. Final tolerance: {last_tolerance:.4f}, final effective sample size: {last_neff:.2f}.")
                    break

        # Warn if convergence not reached after either parallel or non-parallel version
        if self.config.verbose >= 2 and condition is False:
            print(f"Not converged after reaching max steps of {step_number}. Final effective sample size: {last_neff:.2f}, final tolerance: {last_tolerance:.4f}.\n"
                  f"Consider increasing max_steps.")

        self.step_number = step_number # Update step number once mcmc has finished
        return

    def _get_sampler_kwargs(self, nsteps, state=None):
        # Gets sampler kwargs for the emcee EnsembleSampler and run_mcmc functions based on the current state of the
        # ACID instance and the inputted nsteps and state.

        # Set verbosity of the sampler with sampler_progress override if specified
        sampler_verbosity = True if self.config.verbose >= 2 else False
        sampler_verbosity = self.config.sampler_progress if self.config.sampler_progress is not None else sampler_verbosity
        
        backend = None
        if state is None:
            if self.sampler is None:
                raise ValueError(f"Either a state or an existing sampler must be provided to initiate the sampler. \n" \
                                 "This has most likely happened because you ran continue_sampling without first running ACID or using run_mcmc=False.")
            backend = self.sampler.backend # This includes previous seed
        
        # Now that the backend has been set depending on an existing state, if the backend is stil None, we choose depending on sampler_path
        if backend is None and self.config.sampler_path is not None:
            utils.ensure_directory(os.path.dirname(self.config.sampler_path), "sampler directory")
            backend = emcee.backends.HDFBackend(self.config.sampler_path)
            if state is not None:
                backend.reset(self.data.nwalkers, self.data.ndim)
            if self.config.verbose >= 2:
                print(f"Using sampler backend at {self.config.sampler_path}")

        elif self.sampler is not None:
            # And, if it is still none, we check if self.sampler already exists in memory, 
            # and if so, we reuse its backend. We started with the path backend as that supersedes
            backend = self.sampler.backend
        # else: leave none and a normal in-memory sampler backend is used

        if self.config.cores is None:
            if "SLURM_JOB_ID" in os.environ:
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
        nsteps           : IntLike|None = None,
        max_steps        : IntLike|None = None,
        max_steps_kwargs : dict|None    = None,
        parallel         : bool         = None,
        cores            : int          = None,
        moves            : dict         = None,
        return_sampler   : bool         = False,
        ) -> EnsembleSampler | None:
        """
        Continue MCMC sampling for additional steps. This should be called in Result class by the user.
        This necessarily requires a Data instance to have been put into the ACID init.

        Parameters
        ----------
        nsteps : :py:type:`IntLike`, optional
            Number of additional steps to run the MCMC for.
        max_steps : :py:type:`IntLike`, optional
            Maximum number of steps to run the MCMC for in total (including previous steps).
            If specified, the MCMC will stop if this number of steps is reached even if convergence has not been reached, by default None.
            If input, nsteps is ignored.
        max_steps_kwargs : dict, optional
            Additional keyword arguments to be passed to the run_mcmc_until_converged function if max_steps is specified, by default None.
            The kwargs description can be found in Acid.ACID(), they are the 4 kwargs appearing after max_steps. Typos for kwargs are silently
            ignored.
        parallel : bool, optional
            Overwrites config with whether to run the MCMC in parallel. If None, uses already existing configuration. Default is None.
        cores : int, optional
            Overwrites config with the number of cores to use for parallel MCMC. If None, uses already existing configuration. Default is None.
        moves : dict, optional
            Overwrites config with the dictionary specifying the moves to use for MCMC sampling. If None, uses already existing configuration. 
            Default is None. See :py:function:`Acid.ACID` for format.
        return_sampler : bool, optional
            Whether to return the sampler after continuing sampling. Default is True.

        Returns
        -------
        emcee.EnsembleSampler | None
            The MCMC sampler after running for the additional steps, or None if return_sampler is False.
        """
        # Update config with any new parallel, cores, or moves settings for the continued sampling
        self.config.update_hipri(parallel=parallel, cores=cores, moves=moves)

        if max_steps is not None:
            if max_steps_kwargs is not None:
                self.config.update_hipri(**max_steps_kwargs)
            remaining_steps = max_steps - self.sampler.backend.iteration
            if remaining_steps > 0:
                self.run_mcmc_until_converged(remaining_steps, state=None)
                self.data.complete = False

        else:
            if nsteps is None:
                raise ValueError("Either nsteps or max_steps must be provided.")

            self.run_mcmc(nsteps, state=None)
            self.data.complete = False

        self.data.nsteps = self.sampler.backend.iteration

        if return_sampler:
            return self.sampler

    @property
    def result(self) -> Result:
        """Return a Result object for this instance or one passed explicitly.

        Returns
        -------
        Result
            The Result object for the given Acid instance.
        """
        if not self.data.complete:
            raise ValueError("ACID has not been run yet. Cannot create a Result instance.")
        return Result(self)

    @property
    def sampler(self):
        """Returns the sampler stored in the Data class."""
        return self.data.sampler

    @sampler.setter
    def sampler(self, value):
        self.data.sampler = value
 
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
        "n_sig": "sigma_lower",
    }

    # Translate legacy args and kwargs to the current ACID inputs
    run_kwargs = _get_run_kwargs(LEGACY_ACID_ARGS, RENAMED_LEGACY_ARGS, *args, **kwargs)

    data = run_kwargs.pop("data", None)
    return Acid(data=data).ACID(**run_kwargs)

def ACID_HARPS(*args, **kwargs):
    """Legacy ACID_HARPS function, deprecated after 1.4.5.
    """
    raise NotImplementedError(f"ACID_HARPS is no longer supported. \n"
        f"Please use the ACID function with the appropriate inputs for HARPS spectra instead. \n"
        f"Future versions of ACID will provide functions to load and configure data from a range of different standard instruments. \n"
        f"If you still really wish to use ACID_HARPS, the last stable version of ACID with the method is 1.4.5. Try: pip install ACID_code==1.4.5")

def _get_run_kwargs(legacy_args, renamed_args_map, *args, **kwargs):
    """Helper function to translate legacy args and kwargs to their current names.
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

    return combined
