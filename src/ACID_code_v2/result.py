from math import tau
import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os, pickle, warnings, contextlib, functools
from emcee import EnsembleSampler
from beartype import beartype
from scipy.interpolate import interp1d
from numpy import integer as npint
from .lsd import LSD
from . import mcmc
from . import utils
from .data import Data
from .acid import Acid

warnings.filterwarnings("ignore")

__all__ = ['Result']

def _require_all_frames(method):
    # Make sure all results are processed before calling method
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.all_frames is None:
            name = method.__qualname__
            if self.sampler is not None and self.data is not None:
                if self.verbose>0:
                    print(f"Note: The Result object was created without all_frames processed. " \
                        f"Running {name} requires all results to be processed, " \
                        "so process_results() will be called automatically...")
                self.process_results()
            else:
                error = f"Cannot call {name}. The all_frames attribute is not available, and no " \
                "sampler and data objects are available to process results. Please pass an Acid " \
                "object after running ACID."
                raise ValueError(error)
        return method(self, *args, **kwargs)
    return wrapper

def _require_data(method):
    # Make sure Data object is available before calling method
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.data is None:
            name = method.__qualname__
            error = f"Cannot call {name}. The Data object is not available in this " \
            "Result instance. This can occur if Data was set to None to allow for pickling in the " \
            "case that multiple orders or frames were used."
            raise ValueError(error)
        return method(self, *args, **kwargs)
    return wrapper

def _require_sampler(method):
    # Make sure sampler object is available before calling method
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.sampler is None:
            name = method.__qualname__
            error = f"Cannot call {name}. The sampler object is not available in this " \
            "Result instance."
            raise ValueError(error)
        return method(self, *args, **kwargs)
    return wrapper

@beartype
class Result:
    """Class to handle the results from the Acid MCMC sampling, and results processing. Fundamentally, this
    class requires two objects to run, the Sampler object and the Data object, both of which can be obtained
    from the Acid object. If one or the other is not provided, some methods will not work."""

    def __init__(
            self,
            Acid_or_Data_or_Sampler,
            sampler                        = None,
            process_results: bool          = True,
            ACID_HARPS     : bool          = False,
            verbose        : int|bool|None = None,
        ):
        """Initiate Result class

        Parameters
        ----------
        Acid_or_Data_or_Sampler : Acid | Data | emcee.EnsembleSampler
            An Acid object, Data object (contained in Acid class), or sampler object. If an Acid 
            object is provided, all other arguments are taken from there. If a Data object is 
            provided, a sampler can be provided in the second argument. If a sampler object 
            is provided, it will be used as the sampler, but all other attributes will need 
            to be set manually for the Result object to be fully functional.
        sampler : emcee.EnsembleSampler | None, optional
            A sampler object to use if the Data object was provided. If an Acid object 
            was provided, the sampler will be taken from there. If a sampler object was
            provided in the first argument, this will be ignored (with a warning), by default None
        process_results : bool, optional
            Whether to process the results from the Acid object upon initialisation, by default True.
            If False, the all_frames attribute will not be available until Result.process_results() is called.
            The process_results functions does a LSD call, which can be skipped to save time and use
            the Result object for methods that do not require the all_frames attribute, such as 
            continue_sampling() or plot_walkers(). This requires a Data object with the necessary attributes, 
            and a sampler object in the initialisation, or an Acid object with the necessary attributes already set.
            By default, None.
        ACID_HARPS : bool, optional
            Whether the ACID_HARPS function was used, by default False
        verbose : int|bool|None, optional
            Verbosity level, works exactly the same as Acid verbosity, if not provided
            defaults to provided Acid class verbosity otherwise defaults to 2.
        # production_run : bool, optional
        #     Whether Acid was run in production mode, by default False
        """

        self.sampler    = None
        self.data       = None
        self.all_frames = None

        if verbose is True:
            verbose = 2
        elif verbose is False:
            verbose = 0
        elif verbose is None:
            if hasattr(Acid_or_Data_or_Sampler, 'verbose'):
                verbose = Acid_or_Data_or_Sampler.verbose
            else:
                verbose = 2 # the default
        self.verbose = verbose

        # if Acid is None:
        #     self.verbose = verbose if verbose is not None else 2
        #     self.initiate_sampler(sampler) # handles if sampler is None
        #     self.production_run = True # Forces True to activate @_require_all_frames decorator
        #     self.Acid = None
        #     if self.verbose>0:
        #         print("Warning: Acid object not provided. Result object will not be fully functional.")
        #     return
        if isinstance(Acid_or_Data_or_Sampler, Acid):
            self.data = Acid_or_Data_or_Sampler.data
            self.initiate_sampler(Acid_or_Data_or_Sampler.sampler)

        if isinstance(Acid_or_Data_or_Sampler, Data):
            self.data = Acid_or_Data_or_Sampler
            if sampler is not None:
                self.initiate_sampler(sampler)

        if isinstance(Acid_or_Data_or_Sampler, EnsembleSampler):
            self.initiate_sampler(Acid_or_Data_or_Sampler)
            if self.verbose>0:
                print("Warning: Data object not provided. Result object will not be fully functional.")
            return
        
        # From this point, a Data instance is provided and can be drawn from, but sampler may or may not be provided.
        # All frames must be available as a Result class variable due to legacy behaviour. Once created, we can point
        # Data.all_frames to Result.all_frames to keep them in sync.
        if process_results:
            self.process_results() # this function needs to be moved here
            Acid_or_Data_or_Sampler.all_frames = self.all_frames # all frames set in above func
        else:
            if self.verbose>0:
                print("Warning: Results not processed. all_frames attribute will not be available until " \
                "Result.process_results() is called.")

        # Store internal variables
        self.ACID_HARPS = ACID_HARPS

        # Only takes if ACID_HARPS was run, otherwise all None
        self.BJDs = getattr(Acid_or_Data_or_Sampler, 'BJDs', None)
        self.profiles = getattr(Acid_or_Data_or_Sampler, 'profiles', None)
        self.errors = getattr(Acid_or_Data_or_Sampler, 'errors', None)

    def process_results(self):

        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.sampler.get_autocorr_time(quiet=True)

        try:
            burnin = int(2 * np.max(self.tau))
            thin = int(np.min(self.tau)/5)
        except:
            if self.verbose>0:
                print(f"Warning: Could not compute autocorrelation time for burnin and thinning.\n This is likely" \
                f" due to all posterior samples being rejected by prior constraints.\n The resulting profile is likely" \
                f" wrong. Setting burnin=0 and thin=1.")
            burnin = 0
            thin = 1

        # Obtain flattened samples
        flat_samples = self.sampler.get_chain(discard=burnin, thin=thin, flat=True)

        # Getting the final profile and continuum values - median of last 1000 steps
        nvel = len(self.velocities) if self.fit_profile else 0
        quartiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        errors = np.diff(quartiles, axis=0)
        errors = np.max(errors, axis=0) # why?
        self.profile       = quartiles[1, :nvel]
        self.profile_err   = errors[:nvel]
        self.poly_cos      = quartiles[1, nvel:]
        self.poly_cos_err  = errors[nvel:]  

        if self.verbose > 1:
            print('Getting the final profiles...')

        # Finding error for the continuum fit
        norm_wl = self.wavelengths["combined_normalized"]
        coeffs = flat_samples[:, nvel:-1]
        ncoeffs = self.poly_ord + 1 # is equivalent to coeffs.shape[1]
        scales = flat_samples[:, -1]
        powers = np.vander(norm_wl, N=ncoeffs, increasing=True)
        conts = (coeffs @ powers.T) * scales[:, None]

        continuum_error = np.std(np.array(conts), axis=0)  

        for counter in range(len(self.flux["input"])):
            flux = np.copy(self.flux["input"][counter])
            error = np.copy(self.errors["input"][counter])
            wavelengths = np.copy(self.wavelengths["input"][counter])
            sn = np.copy(self.sn["input"][counter])

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
            error = np.sqrt((error/mdl1)**2 + (continuum_error/mdl1)**2) # Compare before after
            flux /= mdl1

            remove = tuple([flux<0])
            flux[remove] = 1.
            error[remove] = 1e12

            LSD_profiles = LSD(self)
            LSD_profiles.run_LSD(wavelengths, flux, error, sn=sn)
            profile_OD = LSD_profiles.profile
            profile_errors = LSD_profiles.profile_errors

            profile_f = np.exp(profile_OD)
            profile_errors_f = profile_errors*profile_f
            profile_f = profile_f-1

            self.all_frames[counter, self.order]=[profile_f, profile_errors_f]

        return

    @_require_all_frames
    def __getitem__(self, item):
        """Allows indexing into the all_frames array directly from the Result object.
        """

        if self.ACID_HARPS:
            return self.BJDs[item], self.profiles[item], self.errors[item]
        else:
            return self.all_frames[item]

    @_require_all_frames
    def __iter__(self):
        """Allows iteration over the BJDs, profiles, and errors if ACID_HARPS was used.
        """
        if self.ACID_HARPS:
            return iter((self.BJDs, self.profiles, self.errors))
        # Acid is not subscriptable normally, only when ACID_HARPS was called 
        raise TypeError("Result is not iterable unless ACID_HARPS=True")

    @_require_data
    def continue_sampling(self, nsteps:int|npint, production_run:bool|None=None):
        """Continue MCMC sampling for additional steps.

        Parameters
        ----------
        nsteps : int
            Number of additional MCMC steps to run.
        production_run : bool, optional
            Whether to set the run as a production run, by default None, keeping the current setting.
            The default is False from Acid.
        """
        if production_run is not None:
            self.production_run = production_run

        self.Acid.continue_sampling(nsteps)

        # Update attributes from Acid object
        self.nsteps = self.Acid.nsteps
        self.sampler = self.Acid.sampler
        if not self.production_run:
            self.all_frames = self.Acid.all_frames

        # Recalculate autocorr time, burnin, thin
        # Suppress output from get_autocorr_time call
        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.Acid.sampler.get_autocorr_time(quiet=True)
        self.burnin = int(2 * np.max(self.tau))
        self.thin = int(np.min(self.tau)/5)
    
    @_require_data
    def process_results(self):
        """Processes the MCMC results to extract the LSD profiles and errors. Can be used
        to convert a production run Result object into one with all results processed.
        """
        self.Acid.process_results(return_result=False)
        self.production_run = False
        self.all_frames = self.Acid.all_frames

    def plot_walkers(self, sampler=None, burnin:int|npint|None=None, thin:int|npint|None=None, return_fig:bool=False):
        """Plots, at maximum, the last 10 MCMC walkers for the LSD profile and continuum 
        polynomial coefficients.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        burnin : int | None, optional
            Optionally define the number of burnin steps, by default None
        thin : int | None, optional
            Optionally define the number of thinning steps, by default None
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False
        """

        self.initiate_sampler(sampler)

        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        naxes = min(10, self.ndim)
        fig, ax = plt.subplots(naxes, 1, figsize=(10, 20), sharex=True)
        samples = self.sampler.get_chain(discard=burnin, thin=int(thin))
        steps = np.arange(samples.shape[0]) * thin + burnin
        for i in range(naxes):
            ax_i = ax[i]
            ax_i.plot(steps, samples[:, :, i], "k", alpha=0.3)
            ax_i.axvspan(0, burnin, color="red", alpha=0.1, label="burn-in")

        ax[-1].legend()
        ax[-1].set_xlabel("Step number")
        ax[-1].set_xlim(0, self.nsteps)
        ax[0].set_title('MCMC Walkers')
        plt.subplots_adjust(hspace=0.05)
        if return_fig:
            return fig, ax
        plt.show()

    def plot_corner(self, sampler=None, return_fig:bool=False, **kwargs):
        """Creates a corner plot for at maximum the last 8 LSD profile and continuum polynomial coefficients.
        
        Parameters
        ----------
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        return_fig : bool, optional
            Whether to return the figure object instead of showing the plot, by default False
        **kwargs:
            Additional keyword arguments to pass to corner.corner().
        """

        self.initiate_sampler(sampler)

        naxes = min(8, self.ndim)
        samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, -naxes:]
        fig = corner.corner(samples, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
        plt.suptitle('MCMC Corner Plot')
        if return_fig:
            return fig
        plt.show()

    @_require_all_frames
    def plot_profiles(
        self,
        grid            :bool      = True,
        labels          :dict|None = None,
        return_fig      :bool      = False,
        subplot_kwargs  :dict|None = None,
        errorbar_kwargs :dict|None = None,
        fig_ax                     = None,
        **kwargs,
        ):
        """Plots the LSD profile result from Acid.

        Parameters
        ----------
        grid : bool, optional
            Show or hide grid, by default True
        labels : dict | None, optional
            Keys: 'xlabel', 'ylabel', and 'title'. Allows label overrides., by default None
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(), by default None
        errorbar_kwargs : dict | None, optional
            Keyword arguments to be passed to ax.errorbar(), by default None
        fig_ax : tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] | None, optional
            Optionally provide an existing fig/axis tuple to plot on, by default None
        """
        # Set default errorbar kwargs
        errorbar_defaults = {
            "fmt"      : ".-",
            "ecolor"   : "red",
            "linewidth": 1,
        }
        errorbar_kwargs = utils.set_dict_defaults(errorbar_kwargs, errorbar_defaults)

        # Set default subplot kwargs
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, {"figsize": (10, 6)})

        # Set default labels
        default_labels = {
            "title" : "Acid Profile",
            "xlabel": "Velocity (km/s)",
            "ylabel": "Normalised Flux"
        }
        labels = utils.set_dict_defaults(labels, default_labels)

        # Set useful variables
        nframes = len(self.all_frames)
        norders = len(self.all_frames[0])
        frames = np.copy(self.all_frames)
        if fig_ax is None:
            fig, ax = plt.subplots(**subplot_kwargs)
        else:
            fig, ax = fig_ax
        if nframes > 1:
            if norders > 1:
                print("Warning: Multiple frames and orders detected. Only plotting the first frame for each order")
                frames = frames[:1, :, :, :]  # Take first frame only
        for f, frame in enumerate(frames):
            for o, order in enumerate(frame):
                x, y, yerr = self.velocities, order[0], order[1]
                # TODO: Make Order a function of self.order_range, which needs to be configured in Acid
                # so that order_range is done automatically if multiple orders are manually put (and not 
                # just using ACID_HARPS)
                label_default = f"Frame {f+1}, Order {o+1}" if nframes > 1 and norders > 1 else None
                errorbar_kwargs = utils.set_dict_defaults(errorbar_kwargs, {"label": label_default})
                ax.errorbar(x, y, yerr=yerr, **errorbar_kwargs)

        ax.set_title(labels["title"])
        ax.set_xlabel(labels["xlabel"])
        ax.set_ylabel(labels["ylabel"])
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.legend()
        ax.grid(grid)
        if return_fig:
            return fig, ax
        else:
            plt.show()

    @_require_all_frames
    def plot_forward_model(
        self,
        input_version   :str       = "masked",
        grid            :bool      = True,
        labels          :dict|None = None,
        return_fig      :bool      = False,
        subplot_kwargs  :dict|None = None,
        **kwargs # for testing
        ):
        """Plots the forward model fit to the observed spectrum.

        Parameters
        ----------
        input_version : str, optional
            Which input spectrum to use: 'combined', 'input', 'masked', by default 'masked'
        grid : bool, optional
            Show or hide grid, by default True
        labels : dict | None, optional
            Keys: 'xlabel', 'ylabel', 'title', and 'residuals_ylabel'. Allows label overrides, by default None
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None
        """

        # Validate all inputs and set defaults
        input_version = input_version.lower()
        if input_version not in self.wavelengths.keys():
            raise ValueError(f"input_version must be one of {list(self.wavelengths.keys())}")
        
        # Set default labels
        default_labels = {
            "title"           : "Forward Model Fit to Observed Spectrum",
            "xlabel"          : "Wavelength (Angstroms)",
            "ylabel"          : "Normalised Flux",
            "residuals_ylabel": "Residuals",
        }
        labels = utils.set_dict_defaults(labels, default_labels)

        # Set default subplot kwargs
        subplot_kwargs = {
            "figsize": (10, 8),
            "sharex": True,
            "gridspec_kw": {'height_ratios': [3, 1]}
        }
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, {"figsize": (10, 8)})

        # Get input data
        input_wavelengths = self.wavelengths[input_version]
        input_flux = self.flux[input_version]
        input_errors = self.errors[input_version]

        # Get model flux
        theta_median = np.median(self.samples, axis=0)
        model_flux, _ = self.model(theta_median)

        # Plotting
        fig, ax = plt.subplots(2, 1, **subplot_kwargs)
        ax[0].plot(input_wavelengths, input_flux, color='black', linewidth=1, label='Observed Spectrum')
        ax[0].plot(input_wavelengths, model_flux, color='C0', linewidth=1, label='Forward Model Fit')
        ax[1].plot(input_wavelengths, input_flux - model_flux, color='C0', linewidth=1, label='Residuals')
        ax[0].set_title(labels["title"])
        ax[1].set_xlabel(labels["xlabel"])
        ax[0].set_ylabel(labels["ylabel"])
        ax[1].set_ylabel(labels["residuals_ylabel"])
        ax[1].axhline(0, color='black', linestyle='--', linewidth=1)
        ax[0].legend()
        ax[1].legend()
        ax[0].grid(grid)
        ax[1].grid(grid)
        plt.subplots_adjust(hspace=0.05)

        if return_fig:
            return fig, ax
        else:
            plt.show()

    def save_result(self, filename:str="result.pkl"):
        """Saves the Result object to a pickle file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the Result object to, by default "result.pkl"
        """
        
        nframes = len(self.all_frames) if self.all_frames is not None else 0
        norders = len(self.all_frames[0]) if self.all_frames is not None else 0

        if nframes > 1 or norders > 1:
            print("Discarding Acid object to allow for pickling of multiple frames/orders.")
            self.Acid = None

        with open(filename, "wb") as f:
            pickle.dump(self, f)
        if self.verbose>0:
            print(f"Result object saved to {filename}")

    def initiate_sampler(self, sampler):
        """Initiates the sampler attribute from an external sampler.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            An emcee EnsembleSampler object to set as the sampler attribute.
        """
        self.sampler = sampler if sampler is not None else self.sampler
        if self.sampler is None:
            raise ValueError("A sampler must be provided in initialisation or in method call")

        self.ndim = self.sampler.ndim
        self.nwalkers = self.sampler.nwalkers
        self.nsteps = self.sampler.get_chain().shape[0]

        # Calculate autocorr time, burnin, thin
        # Suppress output from get_autocorr_time call
        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.sampler.get_autocorr_time(quiet=True)
        
        if self.nsteps < 50 * np.max(self.tau):
            if self.verbose>1:
                print("The number of MCMC steps is less than 50 times the maximum autocorrelation " \
                "time.\n The sampler may not have converged. Consider running more steps or checking " \
                f"the walker plots.\n The max autocorrelation time is {np.max(self.tau):.2f}, therefore " \
                f"the minimum number of steps should be roughly {int(50 * np.max(self.tau))}.")

        try:
            self.burnin = int(2 * np.max(self.tau))
            self.thin = int(np.min(self.tau)/5)
        except:
            if self.verbose>0:
                print(f"Warning: Could not compute autocorrelation time for burnin and thinning.\n This is likely" \
                f" due to all posterior samples being rejected by prior constraints.\n The resulting profile is likely" \
                f" wrong. Setting burnin=0 and thin=1.")
            self.burnin = 0
            self.thin = 1

        # Samples if burnin and thin dont need inputs
        self.samples = self.sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)

    def initiate_data(self, data):
        """Initiates the data attribute from an external Data object.

        Parameters
        ----------
        data : Data
            A Data object to set as the data attribute.
        """
        self.data = data if data is not None else self.data
        if self.data is None:
            raise ValueError("A Data object must be provided in initialisation or in method call")

        self.all_frames = self.data.all_frames

        # For convenience, let the user call the model without needing to input all required args
        MCMC_class = mcmc.MCMC(self.data)
        self.model = MCMC_class.run_model_function

    @classmethod
    def load_result(cls, result_object:str|object="result.pkl"):
        """Loads a Result object from a pickle file.

        Parameters
        ----------
        filename : str | object, optional
            Name of the file to load the Result object from, or a Result object, by default "result.pkl"
        """
        if isinstance(result_object, str):
            with open(result_object, "rb") as f:
                obj = pickle.load(f)
        else:
            obj = result_object
        obj.__class__ = cls
        if obj.verbose>0:
            print("Result object loaded")
        return obj