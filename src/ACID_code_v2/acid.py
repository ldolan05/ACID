import sys, emcee, warnings, os, time, inspect
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
from . import mcmc_utils
from .result import Result

warnings.filterwarnings("ignore")

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
        linelist_path  :str|None             = None,
        linelist_wl    :np.ndarray|list|None = None,
        linelist_depths:np.ndarray|list|None = None,
        verbose        :int|npint|bool       = 2,
        telluric_lines :np.ndarray|list|None = None,
        name           :str                  = 'ACID',
        seed           :int|npint|None       = 42,
        ):
        """Initialises the Acid class with inputted parameters. The parameters set here arre independent
        of the choice of the ACID and ACID_HARPS functions, which take different formats for inputted spectra.

        Parameters
        ----------
        linelist_path : str | None, optional
            Path to linelist. Takes VALD linelist in long or short format as input. Minimum line depth input into VALD must
            be less than 1/(3*SN) where SN is the highest signal-to-noise ratio of the spectra. If None, you can directly provide linelist_wl
            and linelist_depths instead. At least one of linelist_path or linelist_wl and linelist_depths must be provided., by default None
        velocities : np.ndarray | None, optional
            Velocity grid for LSD profiles (in km/s). For example, use: np.arange(-25, 25, 0.82) to create. If None, a default grid
            from -25 to 25 km/s with a spacing calculated by calc_deltav. It is highly recommended to choose your own velocity grid, by default None
        linelist_wl : np.ndarray | list | None, optional
            Wavelengths of lines in linelist (in Angstroms). Only necessary if linelist_path is not provided. 
            Must be same length as linelist_depths. If None, linelist_path must be provided., by default None
        linelist_depths : np.ndarray | list | None, optional
            Depths of lines in linelist (between 0 and 1). Only necessary if linelist_path is not provided. 
            Must be same length as linelist_wl. If None, linelist_path must be provided., by default None
        verbose : bool | int, optional
            An integer between 0 and 3. If 0, nothing is printed. If 2, prints out useful progress information, as well as ACID warnings 
            about any potential issues with the input data or autocorrelation warnings. If True, defaults to 2. If False, defaults to 0.
            If you want to ignore the warnings but still keep progress information, set verbose to 1. A verbosity of 3 will produce 
            additional plots, such as the result of the continuum fit. By default 2
        telluric_lines : np.ndarray | list | None, optional
            List of wavelengths (in Angstroms) of telluric lines to be masked. This can also include problematic
            lines/features that should be masked also. For each wavelengths in the list ~3Å eith side of the line is masked., by default None
        name : str, optional
            Name to call any saved files, by default 'ACID'
        seed : int | None, optional
            Random seed for reproducibility, set it to None to be a random seed, by default 42 (the answer to life,
            the universe and everything)
        """

        # Sets self.velocities, self.linelist_path, self.telluric_lines, self.name, given the inputs
        # Validate velocities input, if None, this is handled in ACID function later when a input spectrum is provided
        if velocities is not None:
            if velocities.ndim != 1:
                raise ValueError("'velocities' must be a one-dimensional array or list")

        # Validate linelist inputs
        if (linelist_wl is None and linelist_depths is None) and linelist_path is None:
            raise ValueError("One of ('linelist_wl' and 'linelist_depths') or 'linelist_path' must be provided.")
        if linelist_path is None and (linelist_wl is None or linelist_depths is None):
            raise ValueError("If 'linelist_path' is not provided, both 'linelist_wl' and 'linelist_depths' must be provided.")
        if linelist_wl is not None:
            # In this case both linelist_wl and linelist_depths were provided as checked in the two above statements
            linelist_wl     = np.array(linelist_wl)
            linelist_depths = np.array(linelist_depths)
            if linelist_wl.ndim != 1 or linelist_depths.ndim != 1:
                raise ValueError("'linelist_wl' and 'linelist_depths' must be a one-dimensional array or list")
            if linelist_wl.shape != linelist_depths.shape:
                raise ValueError("'linelist_wl' and 'linelist_depths' must have the same length and shape")
        # To keep linelist_path as the main input to LSD, if linelist_wl and linelist_depths are provided,
        # make linelist_path a dictionary to pass to LSD, which contains wavelengths and depths to be read by LSD
        if linelist_path is None:
            linelist_path = {"wavelengths": linelist_wl, "depths": linelist_depths}

        # Define telluric_lines with defaults if not input, check type if it is
        if telluric_lines is None:
            telluric_lines = [3820.33, 3933.66, 3968.47, 4327.74, 4307.90, 4383.55, 4861.34,
                              5183.62, 5270.39, 5889.95, 5895.92, 6562.81, 7593.70, 8226.96]
        if not isinstance(telluric_lines, (list, np.ndarray)):
            raise TypeError("telluric_lines must be a list or numpy array of telluric lines to" \
            "mask in angstroms (could be empty or single-valued)")
        telluric_lines = np.array(telluric_lines)
        if telluric_lines.ndim != 1 or telluric_lines.size == 0:
            raise ValueError("telluric_lines must be a one-dimensional array or list")
        
        # Make verbosity always an int regardless of input type, and check correct range
        if verbose is True:
            verbose = 2
        elif verbose is False:
            verbose = 0
        elif isinstance(verbose, int):
            if verbose < 0 or verbose > 3:
                raise ValueError("verbose must be an integer between 0 and 3 (or True/False)")

        # Set the above class attributes
        self.velocities     = velocities
        self.linelist_path  = linelist_path
        self.telluric_lines = telluric_lines
        self.name           = name
        self.verbose        = verbose
        self.seed           = seed

        # Define default order range, can be overwritten in ACID_HARPS
        self.order_range = [1]

        # Set an initial all_frames to None, which is smartly handled in ACID (by input or defaults) and ACID_HARPS
        self.all_frames = None

        # Determine if running in SLURM environment
        self.slurm = "SLURM_JOB_ID" in os.environ

        # Set a random seed
        np.random.seed(self.seed)
        
        return

    def ACID(
        self,
        input_wavelengths,
        input_spectra,
        input_spectral_errors,
        frame_sns                      = None,
        all_frames                     = None,
        poly_ord       :int|npint      = 3,
        pix_chunk      :int|npint      = 20,
        dev_perc       :int|npint      = 25,
        n_sig          :int|npint      = 1,
        order          :int|npint      = 0,
        parallel       :bool           = True,
        cores          :int|npint|None = None,
        nsteps         :int|npint      = 10000,
        return_result  :bool           = True,
        production_run :bool           = False,

        # Testing and internal kwargs
        _input_data                    = None,
        **kwargs,
        ):
        """Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra,
        returning an LSD profile for each spectrum given. Spectra must cover a similiar wavelength range.

        Parameters
        ----------
        input_wavelengths : np.ndarray | list
            An array of wavelengths for each frame (in Angstroms). For multiple frames this should be a 2-d array such that
            input_wavelengths[i] corresponds to the wavelengths for the ith frame.
        input_spectra : np.ndarray | list
            An array of spectral frames (in flux). For multiple frames this should be a 2-d array such that 
            input_spectra[i] corresponds to the spectral fluxes for the ith frame.
        input_spectral_errors : np.ndarray | list
            Errors for each frame (in flux). For multiple frames this should be a 2-d array such that
            input_spectral_errors[i] corresponds to the spectral errors for the ith frame.
        frame_sns : int | np.ndarray | list | None, optional
            Average signal-to-noise ratio for each frame (used to calculate minimum line depth to consider from line list).
            Each frame should have only one S/N value, so for multiple frames this should be a 1-d array such that
            frame_sns[i] corresponds to the S/N for the ith frame. If None, the S/N will be estimated from the input
            spectra (this will make the code work, but is unlikely to produce the desired result)., by default None
        all_frames : str | np.ndarray | None, optional
            Output array for resulting profiles. Only neccessary if looping ACID function over many wavelength
            regions or order (in the case of echelle spectra). General shape needs to be
            (no. of frames, no. of orders, 2, no. of velocity pixels). If not provided, one is created with that shape.
             The only allowed string is "default" due to legacy behaviour, which now acts the same as None, by default None
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
        parallel : bool, optional
            If True uses multiprocessing to calculate the profiles for each frame in parallel, by default True
        cores : int, optional
            Number of cores to use if parallel=True. If None, all available cores will be used, by default None
        nsteps : int, optional
            nsteps (int, optional): Number of steps for the MCMC to run, try increasing if it doesn't converge,
            by default 10000
        return_result : bool, optional
            If True, returns the Result object with the resulting profiles. Otherwise, you will need to handle
            the output manually from acid.all_frames and acid.sampler, by default True
        production_run : bool, optional
            If True, skips the final process_results step and returns a Result object directly. This allows for
            faster chain analysis and want to increase the number of steps with result.continue_sampling(steps).
            If true, some methods in Result will be desabled, by default False
        **kwargs : dict, optional
            Additional keyword arguments. For the moment, these are not used. They are included to allow for
            future expansion of the function without breaking existing code.
        Returns
        -------
        Result
            Result object containing the LSD profiles and associated data. See Result class for methods and attributes.

        Raises
        ------
        TypeError
            If the input types are not as expected.
        """
        ### Setup, validation and input conversion

        # Validate input arrays using the validate_args function within utils.py, ensuring inputs are correct shape, or to
        # best guess the user's intentions. See the utils.validate_args function for more details. This also converts
        # inputs to numpy arrays.
        input_wavelengths, input_spectra, input_spectral_errors = [
            utils.validate_args(arg, i) for i, arg in enumerate((input_wavelengths, input_spectra, input_spectral_errors))]
        frame_sns = utils.validate_args(frame_sns, 3, sn=True, allow_none=True)

        # Check all inputs have the same shape
        if not input_wavelengths.shape == input_spectra.shape == input_spectral_errors.shape:
            raise ValueError("Input wavelengths, spectra and spectral errors must all have the same shape.")

        # Attempt to convert input spectra to be within 0 and 1 if they are not already and warning if this is the case
        if np.any(input_spectra <= 0):
            if self.verbose > 1:
                print("Input spectra contain values <= 0. ACID will attempt to rescale inputs, and mask " \
                "negative values. However, it is recommended to input spectra that are already normalised and positive. " \
                "Please check your data. You can check acid.scale_spectra for more information on how this is done.")
            input_wavelengths, input_spectra, input_spectral_errors = utils.scale_spectra(
                input_wavelengths, input_spectra, input_spectral_errors)

        # Validated frame_sns input
        # If frame_sns is not provided, estimate using specutils, this is a very rudimentary guess and get around for not
        # providing a SNS which should normally come from fits files.
        if frame_sns is None:
            frame_sns = utils.guess_SNR(input_wavelengths, input_spectra, input_spectral_errors)
            assert frame_sns.shape == input_spectra.shape, \
            "frame_sns.shape and input_spectra.shape do not match"
        if np.asarray(frame_sns).shape == np.asarray(input_spectra).shape:
            raise ValueError("frame_sns must be a single-valued list/array with the average S/N for each frame, " \
            "not an array of S/N values for each pixel.")

        # Assign all inputs to class variables (except all frames, handled below)
        self.wavelengths    = {"input": input_wavelengths}
        self.flux           = {"input": input_spectra}
        self.errors         = {"input": input_spectral_errors}
        self.sn             = {"input": frame_sns}
        self.poly_ord       = poly_ord
        self.nsteps         = nsteps
        self.order          = order
        self.pix_chunk      = pix_chunk
        self.dev_perc       = dev_perc
        self.n_sig          = n_sig
        self.return_result  = return_result
        self.parallel       = parallel
        self.production_run = production_run
        self.cores          = cores

        if isinstance(all_frames, str):
            if all_frames == "default":
                all_frames = None # legacy behaviour
        if all_frames is None:
            if self.all_frames is None:
                # By default order_range is [1], so len(self.order_range) = 1, which is same as original
                # code behaviour. This change allows self.order_range to be used in ACID_HARPS.
                self.all_frames = np.zeros((len(self.flux["input"]), len(self.order_range), 2, len(self.velocities)))
        else:
            self.all_frames = all_frames
        if isinstance(self.all_frames, Result):
            self.all_frames = self.all_frames.all_frames
        if not isinstance(self.all_frames, np.ndarray):
            raise TypeError("'all_frames' must be a numpy array")
        if not self.all_frames.ndim == 4:
            raise ValueError("'all_frames' must be a 4-dimensional numpy array, see docstring for details")

        ### Begin ACID process

        if self.verbose>0:
            t0 = time.time()
            print('Initialising...')

        # Combines spectra from each frame (weighted based of S/N), returns to S/N of combined spectra.
        # If only one frame, just uses that frame (handled in the function).
        # This function requires assigned values:
        # self.wavelengths["input"], self.flux["input"], self.errors["input"], self.sn["input"]
        # To generate:
        # As of 1.0.4, this generates self.wavelengths["combined"], self.flux["combined"], self.errors["combined"]
        self.combine_spec(output=False)

        # Get the initial polynomial coefficents
        a, b = utils.get_normalisation_coeffs(self.wavelengths["combined"])
        self.wavelengths["combined_normalized"] = (self.wavelengths["combined"]*a)+b

        # Compute an initial continuum fit
        # poly inputs has polynomial coefficients and scale at the end
        self.poly_inputs, self.flux["fitted"], self.errors["fitted"] = self.continuumfit(
            self.flux["combined"], self.wavelengths["combined_normalized"], self.errors["combined"])
        self.wavelengths["fitted"] = np.copy(self.wavelengths["combined"]) # Just to keep track
        self.sn["fitted"]      = np.copy(self.sn["combined"])      # SN is not changed here

        # Get the initial LSD profile using the initial fit
        initial_LSD = LSD(self) # Initialise LSD class with standard Acid attributes
        initial_LSD.run_LSD(self.wavelengths["fitted"], self.flux["fitted"], self.errors["fitted"], self.sn["fitted"])

        # Use alpha matrix and initial profile class variables from initial LSD run
        self.initial_profile = initial_LSD.profile
        self.initial_profile_errors = initial_LSD.profile_errors
        self.alpha = initial_LSD.alpha

        # Set x, y, yerr, and model_inputs for emcee
        self.model_inputs = np.concatenate((self.initial_profile, self.poly_inputs))

        # Masking based off residuals
        if self.verbose>0:
            print('Residual masking...')

        # Inputs:
        # self.x, self.y, self.yerr, self.model_inputs, self.poly
        # Sets:
        # self.model_inputs_resi
        # Modifies:
        # self.alpha, self.yerr
        self.residual_mask() # will eventually add options for this

        ## Setting number of walkers and their start values(pos)
        self.ndim = len(self.model_inputs)
        self.nwalkers = self.ndim * 3
        rng = np.random.default_rng(self.seed)

        ### starting values of walkers with independent variation
        sigma = 0.8 * 0.005
        initial_state = []
        for i in range(0, self.ndim):
            if i < self.ndim - self.poly_ord - 2:
                pos = rng.normal(self.model_inputs[i], sigma, (self.nwalkers, ))
            else:
                x1 = self.model_inputs[i]
                rounded_sigma = round(x1, 1-int(floor(log10(abs(x1))))-1)
                sigma = abs(rounded_sigma) / 10
                pos = rng.normal(self.model_inputs[i], sigma, (self.nwalkers, ))
            initial_state.append(pos)

        initial_state = np.transpose(np.array(initial_state))
        self.initial_state = initial_state # Saved for debugging if needed, otherwise class variable not used for now

        # Setting global data for mcmc_utils initialisation
        self.mcmc_global_data = {
            "x"         : self.wavelengths["masked"],
            "y"         : self.flux["masked"],
            "yerr"      : self.errors["masked"],
            "alpha"     : self.alpha,
            "velocities": self.velocities,
            "seed"      : self.seed
            }

        if self.verbose>0:
            t5 = time.time()
            print('Initialised in %ss'%round((t5-t0), 2))
            print('Fitting the continuum using emcee...')

        _input_data = _input_data if _input_data is not None else {}
        if "sampler" not in _input_data:
            self._run_mcmc(initial_state, self.nsteps)

        else: # Only for testing
            self.sampler = _input_data.get("sampler")

        # At this point, MCMC has been run, and results need to be processed. This is normally done automatically
        # when using ACID function, but if using class directly, user can choose to call this part separately.
        if production_run is False:
            return self.process_results()
        else:
            return Result(self, production_run=True)

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
            self.order_range = np.arange(10, 70)

        self.order_range = np.array(order_range) # Makes sure order range is an array regardless of input type
        self.file_type = file_type
        self.filelist = filelist

        for order in self.order_range:

            print('Running for order %s/%s...'%(order-min(self.order_range)+1, max(self.order_range)-min(self.order_range)+1))

            frame_wavelengths, frame_flux, frame_errors, sns = self.read_in_frames(order, self.filelist, self.file_type)

            # Updates recursively the all_frames array with the profiles for each order
            self.ACID(
                frame_wavelengths,
                frame_flux,
                frame_errors,
                sns,
                order         = order-min(self.order_range),
                return_result = False,
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
            
            for order in self.order_range:
                hdr['ORDER'] = order
                hdr['BJD'] = fits_file[0].header['ESO DRS BJD']
                if order == self.order_range[0]:
                    BJDs.append(fits_file[0].header['ESO DRS BJD'])
                hdr['CRVAL1'] = np.min(self.velocities)
                hdr['CDELT1'] = self.velocities[1] - self.velocities[0]

                profile = self.all_frames[frame_no, order-min(self.order_range), 0]
                profile_err = self.all_frames[frame_no, order-min(self.order_range), 1]

                hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
                if save_path != 'no save':
                    month = 'August2007'
                    hdu.writeto('%s%s_%s_%s.fits'%(save_path, month, frame_no, self.name), output_verify='fix', overwrite='True')

            result1, result2 = self.combineprofiles(self.all_frames[frame_no, :, 0], self.all_frames[frame_no, :, 1])
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

        if frame_wavelengths: # This should only be for testing
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
            self.wavelengths["combined"] = np.copy(self.wavelengths["input"][0])
            self.flux["combined"]        = np.copy(self.flux["input"][0])
            self.errors["combined"]      = np.copy(self.errors["input"][0])
            self.sn["combined"]          = np.copy(self.sn["input"][0])

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

            combined_flux = np.zeros_like(combined_wavelengths)
            combined_errors = np.zeros_like(combined_wavelengths)

            for n in range(len(combined_wavelengths)):
                temp_spec_f = interpolated_flux[:, n]
                temp_err_f = interpolated_errors[:, n]

                weights_f = (1/temp_err_f**2)

                idx = tuple([temp_err_f>=1e12])
                weights_f[idx] = 0.

                if sum(weights_f) > 0:
                    weights_f = weights_f / np.sum(weights_f)

                    combined_flux[n] = sum(weights_f * temp_spec_f)
                    combined_sn = sum(weights_f * sn) / sum(weights_f)

                    combined_errors[n] = 1 / (sum(weights_f ** 2)) # TODO! CHECK this is the right calculation with the square

                else:
                    combined_flux[n] = np.mean(temp_spec_f)
                    combined_errors[n] = 1e12

            # Faster and hopefully correct method once checked with Ernst
            # invvars = 1 / interpolated_errors**2
            # invvars[interpolated_errors >= 1e12] = 0
            # weighted_flux = interpolated_flux * invvars
            # combined_flux = np.sum(weighted_flux, axis=0) / np.sum(invvars, axis=0)
            # combined_errors = 1 / np.sqrt(np.sum(invvars, axis=0))

            self.wavelengths["combined"] = combined_wavelengths
            self.flux["combined"]        = combined_flux
            self.errors["combined"]      = combined_errors
            self.sn["combined"]          = combined_sn

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

        cont_factor = np.percentile(fluxes, 95)
        idx = wavelengths.argsort()
        wavelength = wavelengths[idx]
        fluxe = fluxes[idx] / cont_factor
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
        fit = poly(wavelengths) * cont_factor
        flux_obs = fluxes / fit
        new_errors = errors / fit
        poly_coeffs = np.concatenate((np.flip(coeffs), [cont_factor]))

        if self.verbose > 2 or plot_result is True:
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
        x = self.wavelengths["combined"]
        y = self.flux["combined"]
        yerr = self.errors["combined"]
        x_norm = self.wavelengths["combined_normalized"]

        forward = mcmc_utils.model_func(self.model_inputs, x, alpha=self.alpha)

        mdl1 = 0
        nvel = len(self.velocities)
        for i in range(nvel, len(self.model_inputs) - 1):
            mdl1 = mdl1 + (self.model_inputs[i] * x_norm ** (i - nvel))

        mdl1 = mdl1 * self.model_inputs[-1]

        data_normalised = (y - np.min(y)) / (np.max(y) - np.min(y))
        forward_normalised = (forward - np.min(forward)) / (np.max(forward) - np.min(forward))
        residuals = data_normalised - forward_normalised
        
        ### finds consectuative sections where at least pix_chunk points have residuals greater than 0.25 - these are masked
        idx = (abs(residuals) > self.dev_perc / 100)

        flag_min = 0
        flag_max = 0
        for value in range(len(idx)):
            if idx[value] == True and flag_min <= value:
                flag_min = value
                flag_max = value
            elif idx[value] == True and flag_max < value:
                flag_max = value
            elif idx[value] == False and flag_max - flag_min >= self.pix_chunk:
                yerr[flag_min:flag_max] = 1e12
                # print(f"Masking from {x[flag_min]} to {x[flag_max]} due to residuals.")
                flag_min = value
                flag_max = value

        ##############################################
        #                  TELLURICS                 #   
        ##############################################

        # self.yerr_compare = self.yerr.copy()

        ## masking tellurics
        for line in self.telluric_lines:
            limit = (21/c_kms)*line +3
            idx = np.logical_and((line-limit) <= x, x <= (limit+line))
            yerr[idx] = 1e12

        # Note that this is used to keep track of the residual masks for later use in _get_profiles
        self.residual_masks = tuple([yerr >= 1e12])

        ###################################
        ###      sigma clip masking     ###
        ###################################

        m = np.median(residuals)
        sigma = np.std(residuals)

        # TODO! : check what the hell is gong on here with a_old
        a_old = 1

        upper_clip = m + self.n_sig * sigma
        lower_clip = m - self.n_sig * sigma

        rcopy = residuals.copy()

        idx1 = tuple([rcopy <= lower_clip])
        idx2 = tuple([rcopy >= upper_clip])

        yerr[idx1] = 1e12
        yerr[idx2] = 1e12

        a, b = utils.get_normalisation_coeffs(x)
        poly_inputs, _bin, bye = self.continuumfit(y, (x*a)+b, yerr, self.poly_ord)

        LSD_masking = LSD(self)
        LSD_masking.run_LSD(x, _bin, bye, sn=100)
        # profile = LSD_masking.profile
        self.alpha = LSD_masking.alpha

        if self.verbose > 2:
            nremoved = np.sum(idx1)+np.sum(idx2)
            print(f"Residal masking has removed {nremoved}/{len(residuals)} points.")

            plt.figure(figsize=(10, 6))
            plt.plot(x, residuals, label='Residuals', color='blue')
            plt.axhline(upper_clip, color='red', linestyle='--', label='Upper Clip Threshold')
            plt.axhline(lower_clip, color='green', linestyle='--', label='Lower Clip Threshold')
            plt.fill_between(x, upper_clip, lower_clip, color='gray', alpha=0.3, label='Clipped Region')
            for i, line in enumerate(self.telluric_lines):
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
            plt.plot(self.velocities, LSD_masking.profile, label='LSD Profile after Masking and before sampling', color='red')
            plt.title('LSD Profile after Residual Masking')
            plt.xlabel('Velocity (km/s)')
            plt.ylabel('LSD Profile')
            plt.legend()
            plt.show()

        self.wavelengths["masked"] = x
        self.flux["masked"]        = y # x and y dont change in this func
        self.errors["masked"]      = yerr # yerr is modified in this func
        # self.alpha is also modified in this func to get new alpha with masked residuals using pix chunk and dev perc

        return

    def _run_mcmc(
        self,
        state,
        nsteps,
        ):

        sampler_verbosity = True if self.verbose>0 else False
        backend = None
        if state is None:
            if not hasattr(self, 'sampler'):
                raise ValueError("No existing sampler found. Please run 'ACID' first or provide a state.")
            backend = self.sampler.backend # This includes previous seed

        if self.cores is None:
            if self.slurm:
                self.cores = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
            else:
                self.cores = os.cpu_count()
        
        moves=[ # Use the new moves option in emcee v3 for faster convergence
            (emcee.moves.StretchMove(), 0.6),
            (emcee.moves.DEMove(), 0.3),
            (emcee.moves.DESnookerMove(), 0.1),
        ]
        sampler_kwargs = {
            "nwalkers"   : self.nwalkers,
            "ndim"       : self.ndim,
            "log_prob_fn": mcmc_utils._log_probability,
            "moves"      : moves,
            "backend"    : backend,
        }
        mcmc_kwargs = {
            "initial_state": state,
            "nsteps"       : nsteps,
            "progress"     : sampler_verbosity,
            "store"        : True,
        }

        if self.parallel:
            os.environ["OMP_NUM_THREADS"] = "1" # emcee recommendation for multiprocessing
            if self.verbose>0:
                print(f"Using {self.cores} cores for MCMC")

            # For some reason, unspecified pooling as was before (as in case of windows in the else statement)
            # leds to a hung computer. So specify mp.get_context required, default is spawn, but spawn
            # causes multiple instances of this script to rerun, causing alpha matrix calculation to be redone
            # in each child process. Therefore, fork, which is legacy mp behavior on unix, is used.
            if sys.platform != "win32":
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=self.cores, initializer=mcmc_utils._init_worker, initargs=(self.mcmc_global_data,)) as pool:
                    self.sampler = emcee.EnsembleSampler(**sampler_kwargs, pool=pool)
                    self.sampler.run_mcmc(**mcmc_kwargs)

            else: # This doesn't work, needs serious modifications to make work
                raise NotImplementedError("Parallel MCMC on Windows is not currently supported.")

            os.environ["OMP_NUM_THREADS"] = "None" # reset OMP threads 

        else:
            # log_prob = partial(mcmc_utils._log_probability, global_data=self.mcmc_global_data) # This didn't work initialising global data
            mcmc_utils._init_worker(self.mcmc_global_data)
            self.sampler = emcee.EnsembleSampler(**sampler_kwargs)
            self.sampler.run_mcmc(**mcmc_kwargs)

    def process_results(
        self,
        return_result=True,
        ):

        # Discarding all vales except the last 1000 steps.
        dis_no = self.nsteps-1000

        # Obtain flattened samples
        flat_samples = self.sampler.get_chain(discard=dis_no, flat=True)

        # Getting the final profile and continuum values - median of last 1000 steps
        self.profile      = []
        self.poly_cos     = []
        self.profile_err  = []
        self.poly_cos_err = []

        for i in range(self.ndim):
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            error = np.diff(mcmc)
            if i<len(self.velocities):
                self.profile.append(mcmc[1])
                self.profile_err.append(np.max(error))
            else:
                self.poly_cos.append(mcmc[1])
                self.poly_cos_err.append(np.max(error))
        self.profile = np.array(self.profile)
        self.profile_err = np.array(self.profile_err)
        nvel = len(self.velocities)
        # quartiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        # errors = np.diff(quartiles, axis=0)
        # errors = np.max(errors, axis=0) # why?
        # self.profile       = quartiles[1, :nvel]
        # self.profile_err   = errors[:nvel]
        # self.poly_cos      = quartiles[1, nvel:]
        # self.poly_cos_err  = errors[nvel:]

        if self.verbose > 0:
            print('Getting the final profiles...')

        # Finding error for the continuum fit
        nsamples = 50
        rng = np.random.default_rng(self.seed)
        inds = rng.integers(len(flat_samples), size=nsamples)
        norm_wl = self.wavelengths["combined_normalized"]
        nvel = len(self.velocities)

        # conts1 = []
        # for ind in inds:
        #     sample = flat_samples[ind]
        #     # mdl = mcmc_utils.model_func(sample, self.wavelengths["combined"], alpha=self.alpha)
        #     mdl1_temp = 0
        #     for i in np.arange(len(self.velocities), len(sample)-1):
        #         mdl1_temp = mdl1_temp+sample[i]*(norm_wl**(i-nvel))
        #     mdl1_temp = mdl1_temp*sample[-1]
        #     conts1.append(mdl1_temp)

        samples = flat_samples[inds]
        coeffs = samples[:, nvel:-1]
        ncoeffs = self.poly_ord + 1 # is equivalent to coeffs.shape[1]
        scales = samples[:, -1]
        powers = np.vander(norm_wl, N=ncoeffs, increasing=True)
        conts = (coeffs @ powers.T) * scales[:, None]

        self.continuum_error = np.std(np.array(conts), axis=0)

        self.all_frames = self._get_profiles(self.all_frames)  

        if self.return_result and return_result:
            return Result(self)
        return

    def _get_profiles(
        self,
        all_frames,
        ):

        for counter in range(len(self.flux["input"])):
            flux = self.flux["input"][counter]
            error = self.errors["input"][counter]
            wavelengths = self.wavelengths["input"][counter]
            sn = self.sn["input"][counter]

            a, b = utils.get_normalisation_coeffs(wavelengths)
            norm_wavelengths = (a*wavelengths)+b
            mdl1 =0
            for i in np.arange(0, len(self.poly_cos)-1):
                mdl1 = mdl1+self.poly_cos[i]*norm_wavelengths**(i)
            mdl1 = mdl1*self.poly_cos[-1]

            # mdl1 = np.polynomial.polynomial.polyval(norm_wavelengths, self.poly_cos[:-1]) * self.poly_inputs[-1]

            # Masking based off residuals interpolated onto new wavelength grid
            reference_wave = self.wavelengths["input"][np.argmax(self.sn["input"])]
            mask_pos = np.ones(reference_wave.shape)
            mask_pos[self.residual_masks]=1e12
            f2 = interp1d(reference_wave, mask_pos, bounds_error = False, fill_value = np.nan)
            interp_mask_pos = f2(wavelengths)
            interp_mask_idx = tuple([interp_mask_pos>=1e12])

            error[interp_mask_idx]=1e12

            # corrrecting continuum
            # error = (error/flux) + (self.continuum_error/mdl1)
            # error = np.sqrt((error/flux)**2 + (self.continuum_error/mdl1)**2) # Compare before after
            error = np.sqrt((error/mdl1)**2 + (self.continuum_error/mdl1)**2) # Compare before after

            flux /= mdl1
            # error *= flux

            remove = tuple([flux<0])
            flux[remove] = 1.
            error[remove] = 1e12

            # idx = tuple([flux>0])
            # if len(flux[idx]) == 0:
            #     print('continuing... frame %s'%counter)
            # Removed this if else block as you should be able to assert len(flux[idx])>0 from earlier checks
            # else:

            LSD_profiles = LSD(self)
            LSD_profiles.run_LSD(wavelengths, flux, error, sn=sn)
            profile_OD = LSD_profiles.profile
            profile_errors = LSD_profiles.profile_errors

            # Need to check whats going on here with the -1
            p = np.exp(profile_OD)-1
            profile_f = np.exp(profile_OD)
            profile_errors_f = np.sqrt(profile_errors**2/profile_f**2)
            profile_f = profile_f-1

            all_frames[counter, self.order]=[profile_f, profile_errors_f]
        
        return all_frames

    def read_in_frames(
        self,
        order,
        filelist,
        file_type,
        directory=None,
        ):
        # read in first frame
        fluxes, wavelengths, flux_error_order, sn = LSD().blaze_correct(
            file_type, 'order', order, filelist[0], directory, 'unmasked', self.name, 'y')
        # fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct(
        #     file_type, 'order', order, filelist[0], directory, 'unmasked', self.name, 'y')

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
                file_type, 'order', order, file, directory, 'unmasked', self.name, 'y')
            # print(i, frames)
            return frames, frame_wavelengths, errors, sns
        
        ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
        for i in range(len(filelist[1:])+1):
            # print(i)
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

    def continue_sampling(
        self,
        nsteps        : int|npint,
        production_run: bool = False,
        ):
        """Continue MCMC sampling for additional steps.

        Parameters
        ----------
        nsteps : int
            Number of additional steps to run the MCMC for.
        """
        # Check that sampler exists
        if not hasattr(self, 'sampler'):
            raise AttributeError("No existing sampler found. Please run 'ACID' before continuing.")

        self._run_mcmc(state=None, nsteps=nsteps) # continue from current state
        self.nsteps += nsteps

        if production_run is False:
            return self.process_results()
        else:
            return Result(self, production_run=True)

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