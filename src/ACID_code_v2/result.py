from time import time
import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os, pickle, warnings, contextlib, functools, inspect
from emcee import EnsembleSampler
import emcee.backends.backend as emceebackend
from beartype import beartype
from scipy.interpolate import interp1d
from numpy import integer as npint
from numpy.polynomial import polynomial as P
from .lsd import LSD
from . import mcmc
from . import utils
from .data import Data
from .data import Config

warnings.filterwarnings("ignore")

def _require_all_frames(method):
    # Make sure all results are processed before calling method
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.all_frames is None:
            name = method.__qualname__
            if self.sampler is not None and self.data is not None:
                if self.config.verbose>0:
                    print(f"Note: The Result object was created without all_frames processed. " \
                        f"Running {name} requires all results to be processed, " \
                        "so process_results() will be called automatically...")
                self.process_results()
            else:
                error = f"Cannot call {name}. The all_frames attribute is not available, and no " \
                "sampler and data objects are available to process results. Please pass an Acid " \
                "object after running ACID to the results init."
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
    sig = inspect.signature(method)
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind_partial(self, *args, **kwargs)

        # A specific carvout for the save function
        store_sampler_in_args = "store_sampler" in sig.parameters
        if store_sampler_in_args is True:
            if bound.arguments.get("store_sampler", True) is False:
                return method(self, *args, **kwargs)

        sampler_in_args = "sampler" in sig.parameters
        inputted_sampler = bound.arguments.get("sampler", None)

        self.initiate_sampler(inputted_sampler if sampler_in_args else None)

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
        obj = Acid_or_Data_or_Sampler
        self.sampler    = None
        self.data       = None
        self.all_frames = None
        self.config     = Config() # default config, will be updated if Acid or Data object is provided

        self.config.verbose = verbose

        if hasattr(obj, "data") and hasattr(obj, "config") and hasattr(obj, "sampler"):
            # The above line is only all true if an Acid object (as they are set in initialisation), 
            # the sampler and data classes do not store all 3
            acid = obj
            self.initiate_data(acid.data)
            self.config = acid.config
            self.initiate_sampler(acid.sampler)
        elif isinstance(obj, Data):
            data = obj
            self.initiate_data(data)
            self.config = data.config
            if sampler is not None:
                self.initiate_sampler(sampler)
        elif isinstance(obj, EnsembleSampler):
            self.initiate_sampler(obj)
            if self.config.verbose>0:
                print("Warning: Data object not provided. Result object will not be fully functional.")
            return
        else:
            raise ValueError("First argument must be an Acid object, Data object, or emcee.EnsembleSampler object. "
                             f"Got {type(obj)} instead.")

        # From this point, a Data instance is provided and can be drawn from, but sampler may or may not be provided.
        # All frames must be available as a Result class variable due to legacy behaviour. Once created, we can point
        # Data.all_frames to Result.all_frames to keep them in sync.
        if process_results:
            self.process_results() # sets self.data.all_frames, and points self.all_frames to self.data.all_frames
        else:
            if self.config.verbose>0:
                print("Warning: Results not processed. all_frames attribute will not be available until " \
                "Result.process_results() is called.")

        # Store internal variables
        self.ACID_HARPS = ACID_HARPS

        # Only takes if ACID_HARPS was run, otherwise all None
        self.BJDs = getattr(obj, 'BJDs', None)
        self.profiles = getattr(obj, 'profiles', None)
        self.errors = getattr(obj, 'errors', None)

    @_require_data
    @_require_sampler
    def process_results(self):
        t0 = time()

        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.sampler.get_autocorr_time(quiet=True)

        # Obtain flattened samples
        flat_samples = self.sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)

        # Getting the final profile and continuum values - median of last 1000 steps
        nvel = len(self.data.velocities) if self.config.deterministic_profile is False else 0
        quartiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        errors = np.diff(quartiles, axis=0)
        errors = np.max(errors, axis=0) # why?
        self.profile       = quartiles[1, :nvel]
        self.profile_err   = errors[:nvel]
        self.poly_cos      = quartiles[1, nvel:]
        self.poly_cos_err  = errors[nvel:]  

        if self.config.verbose > 1:
            print('Getting the final profiles...')

        # Finding error for the continuum fit
        norm_wl = self.data.wavelengths["combined_normalized"]
        coeffs = flat_samples[:, nvel:]
        ncoeffs = coeffs.shape[1]
        powers = np.vander(norm_wl, N=ncoeffs, increasing=True)
        conts = (coeffs @ powers.T)

        continuum_error = np.std(np.array(conts), axis=0)  

        for counter in range(len(self.data.flux["input"])):
            flux = np.copy(self.data.flux["input"][counter])
            error = np.copy(self.data.errors["input"][counter])
            wavelengths = np.copy(self.data.wavelengths["input"][counter])
            sn = np.copy(self.data.sn["input"][counter])

            flux = flux[self.data.nanmask]
            error = error[self.data.nanmask]
            wavelengths = wavelengths[self.data.nanmask]

            # Build continuum model
            a, b = utils.get_normalisation_coeffs(wavelengths)
            norm_wavelengths = (a*wavelengths)+b
            mdl1 = P.polyval(norm_wavelengths, self.poly_cos)

            # Masking based off residuals interpolated onto new wavelength grid
            reference_wave = self.data.wavelengths["input"][np.nanargmax(self.data.sn["input"])]
            reference_wave = reference_wave[self.data.nanmask]
            mask_pos = np.ones(reference_wave.shape)
            mask_pos[self.data.residual_masks]=1e12
            f2 = interp1d(reference_wave, mask_pos, bounds_error = False, fill_value = np.nan)
            interp_mask_pos = f2(wavelengths)
            interp_mask_idx = tuple([interp_mask_pos>=1e12])

            error[interp_mask_idx]=1e12

            # correcting continuum
            error = np.sqrt((error/mdl1)**2 + (continuum_error/mdl1)**2)
            flux /= mdl1

            remove = tuple([flux<0])
            flux[remove] = 1.
            error[remove] = 1e12

            LSD_profiles = LSD(self.data)
            LSD_profiles.run_LSD(wavelengths, flux, error, sn=sn)

            profile_f = LSD_profiles.profile_F
            profile_errors_f = LSD_profiles.profile_errors_F
            # profile_f = profile_f-1

            self.all_frames[counter, self.config.order]=[profile_f, profile_errors_f]

        self.data.all_frames = self.all_frames # point Data.all_frames to Result.all_frames to keep them in sync
        self.data.get_profiles_time = time() - t0
        self.data.full_run_time = self.data.initialisation_time + self.data.mcmc_time + self.data.get_profiles_time

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

    def __repr__(self):
        # Only print out the sampler and data attributes, and whether all_frames is available, to avoid printing large arrays
        return f"Result object with sampler={self.sampler}, data={self.data}, all_frames={'available' if self.all_frames is not None else 'not available'}"

    @_require_data
    @_require_sampler
    def continue_sampling(self, nsteps:int|npint, process_results:bool|None=True, sampler=None):
        """Continue MCMC sampling for additional steps.

        Parameters
        ----------
        nsteps : int
            Number of additional MCMC steps to run.
        process_results : bool, optional
            Whether to process the results after continuing sampling, by default True.
            If False, the all_frames attribute will not be updated until Result.process_results() is called.
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to continue sampling from, otherwise,
            takes the sampler from the Result object, by default None
        """
        from .acid import Acid
        acid = Acid(data=self.data) # includes config data
        self.sampler = acid.continue_sampling(self.sampler, nsteps)

        self.initiate_sampler(self.sampler) # update internal variables to match new sampler
        
        # Data nsteps should always match the nsteps chain length, but they can mismatch if the chain at
        # some point was discarded or thinned. We assume data.nsteps to be the most accurate
        self.data.nsteps += nsteps

        if process_results:
            self.process_results() # update all_frames
        else:
            if self.config.verbose>0:
                print("Warning: Results not processed. all_frames attribute will not be available until " \
                "Result.process_results() is called.")

    @_require_sampler
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

        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin
        samples = self.sampler.get_chain(thin=int(thin))
        steps = np.arange(samples.shape[0]) * thin

        naxes = len(self.default_params)

        fig, ax = plt.subplots(naxes, 1, figsize=(10, 20), sharex=True)
        for i in range(naxes):
            ax[i].plot(steps, samples[:, :, self.default_params[i]], "k", alpha=0.3)
            ax[i].axvspan(0, burnin, color="red", alpha=0.1, label="burn-in")
            ax[i].set_ylabel(self.default_param_labels[i])
        ax[-1].legend()
        ax[-1].set_xlabel("Step number")
        ax[-1].set_xlim(0, self.nsteps)
        ax[0].set_title('MCMC Walkers')
        plt.subplots_adjust(hspace=0.05)
        if return_fig:
            return fig, ax
        plt.show()

    @_require_sampler
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

        samples = self.sampler.get_chain()

        samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, self.default_params]

        fig = corner.corner(samples, labels=self.default_param_labels, show_title=True, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
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
                if self.verbose > 0:
                    print("Warning: Multiple frames and orders detected. Only plotting the first frame for each order")
                frames = frames[:1, :, :, :]  # Take first frame only
        for f, frame in enumerate(frames):
            for o, order in enumerate(frame):
                x, y, yerr = self.data.velocities, order[0], order[1]
                # TODO: Make Order a function of self.order_range, which needs to be configured in Acid
                # so that order_range is done automatically if multiple orders are manually put (and not 
                # just using ACID_HARPS)
                label_default = f"Frame {f+1}, Order {o+1}" if nframes > 1 and norders > 1 else None
                errorbar_kwargs = utils.set_dict_defaults(errorbar_kwargs, {"label": label_default})
                ax.errorbar(x, y-1, yerr=yerr, **errorbar_kwargs)

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
        if input_version not in self.data.wavelengths.keys():
            raise ValueError(f"input_version must be one of {list(self.data.wavelengths.keys())}")
        
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
        input_wavelengths = self.data.wavelengths[input_version]
        input_flux = self.data.flux[input_version]
        input_errors = self.data.errors[input_version]

        # Get model flux
        samples = self.sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)
        theta_median = np.median(samples, axis=0)
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

    @_require_sampler
    def plot_autocorrelation(
        self,
        sampler=None,
        burnin: int | None = None,
        thin: int | None = None,
        n_grid: int = 12,
        c: float = 5.0,
        return_fig: bool = False,
        subplot_kwargs: dict | None = None,
        min_steps: int = 100
    ):
        """
        Plot estimated integrated autocorrelation time as a function of chain length.

        From the emcee docs:
        - For several prefixes of the chain, estimate tau with Sokal windowing.
        - Plot tau(N) and the reference line tau = N/50.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        burnin, thin : int | None, optional
            Optional overrides. Defaults to self.burnin/self.thin from the sampler.
        n_grid : int, optional
            Number of N values (prefix lengths) to evaluate, by default 12.
        c : float, optional
            Sokal window constant (usually 5), by default 5.0.
        return_fig : bool, optional
            Whether to return the figure and axes objects, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None
        min_steps : int, optional
            Minimum number of post-burnin samples required to attempt autocorrelation estimation, by default 100
            If you decrease this, you may get unreliable estimates or errors from the autocorrelation time estimation.

        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing 
        the plot. Otherwise, displays the plot and returns None.
        """

        chain = self.sampler.get_chain()  # (nsteps, nwalkers, ndim)
        nsteps, nwalkers, ndim = chain.shape

        if nsteps < min_steps:
            raise ValueError("Not enough post-burnin samples to estimate autocorrelation reliably.")
        
        Ns = np.unique(np.exp(np.linspace(np.log(min_steps), np.log(nsteps), n_grid)).astype(int))
        Ns = Ns[Ns >= min_steps]  # Ensure we only consider N >= min_steps

        tau_estimates = {p: np.full(len(Ns), np.nan, dtype=float) for p in self.default_params}

        # Estimate taus
        for i, n in enumerate(Ns):
            for p in self.default_params:
                y = chain[:n, :, p].T
                tau_estimates[p][i] = utils.autocorr_new(y, c=c)

        subplot_kwargs = {} if subplot_kwargs is None else dict(subplot_kwargs)
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, {"figsize": (10, 6)})

        fig, ax = plt.subplots(**subplot_kwargs)

        for label, p in zip(self.default_param_labels, self.default_params):
            ax.loglog(Ns, tau_estimates[p], "o-", label=f"{label}")

        # Reference line tau = N/50
        ax.loglog(Ns, Ns / 50.0, "--", label=r"$\tau = N/50$")

        ax.set_xlabel("number of post-burnin samples per walker (N)")
        ax.set_ylabel(r"estimated integrated autocorrelation time $\tau$")
        ax.set_title("Autocorrelation time estimates vs chain length")
        ax.legend()
        ax.grid(True, which="both")

        if return_fig:
            return fig, ax
        plt.show()

        return

    @_require_sampler
    def plot_acf(
        self,
        sampler=None,
        max_lag: int | None = None,
        return_fig: bool = False,
        subplot_kwargs: dict | None = None,
    ):
        chain = self.sampler.get_chain() 
        nsteps, nwalkers, ndim = chain.shape

        subplot_kwargs = {} if subplot_kwargs is None else dict(subplot_kwargs)
        subplot_kwargs = utils.set_dict_defaults(subplot_kwargs, {"figsize": (10, 5)})

        fig, ax = plt.subplots(**subplot_kwargs)

        for param, label in zip(self.default_params, self.default_param_labels):

            y = chain[:, :, param].T  # (nwalkers, nsteps)

            # Mean ACF across walkers
            f = np.zeros(nsteps)
            for w in range(nwalkers):
                f += utils.autocorr_func_1d(y[w], norm=True)
            f /= nwalkers

            if max_lag is None:
                max_lag = min(5_000, nsteps - 1)
            max_lag = int(max_lag)

            ax.plot(np.arange(max_lag + 1), f[: max_lag + 1], label=f"{label}")
        
        ax.set_xlabel("Lag (steps)")
        ax.set_ylabel("Autocorrelation")
        ax.set_title(f"Mean ACF across walkers")
        ax.set_xscale("log")
        ax.grid(True)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.legend()

        if return_fig:
            return fig, ax
        plt.show()

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
        if sampler is None:
            return # sampler already initiated from initialisation, so skip the rest of the method

        self.ndim = self.sampler.ndim
        self.nwalkers = self.sampler.nwalkers
        self.nsteps = self.sampler.get_chain().shape[0]

        # Calculate autocorr time, burnin, thin
        # Suppress output from get_autocorr_time call
        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.sampler.get_autocorr_time(quiet=True)
        
        self.converged = True
        if self.nsteps < 50 * np.max(self.tau):
            self.converged = False
            if self.config.verbose>1:
                print("The number of MCMC steps is less than 50 times the maximum autocorrelation " \
                "time.\n The sampler may not have converged. Consider running more steps or checking " \
                f"the walker plots.\n The max autocorrelation time is {np.max(self.tau):.2f}, therefore " \
                f"the minimum number of steps should be roughly {int(50 * np.max(self.tau))}.\n Disabling burnin " \
                f"from autocorrelation time, instead using burnin=steps-1000")

        try:
            self.thin = int(np.min(self.tau)/5)
            if self.converged:
                self.burnin = int(2 * np.max(self.tau))
            else:
                self.burnin = self.nsteps - 1000 # just the last 1000 steps
        except:
            if self.config.verbose>0:
                print(f"Warning: Could not compute autocorrelation time for burnin and thinning.\n This is likely" \
                f" due to all posterior samples being rejected by prior constraints.\n The resulting profile is likely" \
                f" wrong. Setting burnin=nsteps-1000, and thin=1.")
            self.burnin = self.nsteps - 1000 # just the last 1000 steps
            self.thin = 1

        if self.config is not None:
            deterministic = self.config.deterministic_profile
            n_poly_params = self.data.config.poly_ord + 1
        else: # Make a best guess
            if self.ndim > 6: # ie we assume a poly order of 5 is the highest anyone would ever want to go
                deterministic = False
                n_poly_params = 4
            else:
                deterministic = True
                n_poly_params = self.ndim

        poly_params = np.arange(-1, -n_poly_params-1, -1).tolist()
        a=ord('a')
        alph=[chr(i) for i in range(a,a+26)]
        poly_labels = [alph[i] for i in range(n_poly_params)]
        
        samples = self.sampler.get_chain(thin=int(self.thin), discard=int(self.burnin))
        if not deterministic:
            max_profile_idx = np.argmax(samples[:,:,:-n_poly_params].mean(axis=(0,1)))
            poly_params.extend([-5, max_profile_idx, 1])
            poly_labels.extend(["$Z_{-1}$", "$Z_{max}$", "$Z_0$"])
        
        self.default_params = poly_params
        self.default_param_labels = poly_labels

    def initiate_data(self, data):
        """Initiates the data attribute from an external Data object.

        Parameters
        ----------
        data : Data
            A Data object to set as the data attribute.
        """
        self.data = data if data is not None else getattr(self, "data", None)
        if self.data is None:
            raise ValueError("A Data object must be provided in initialisation or in method call")
        if data is None:
            return # data already initiated from initialisation, so skip the rest of the method

        self.all_frames = self.data.all_frames
        self.nsteps     = self.data.nsteps

        # For convenience, let the user call the model without needing to input all required args
        MCMC_class = mcmc.MCMC(self.data)
        self.model = MCMC_class.run_model_function

    @_require_data
    @_require_sampler
    def save_result(self, filename:str="result.pkl", store_sampler:bool=True):
        """Saves the Result object to a pickle file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the Result object to, by default "result.pkl"
        """
        state = dict(self.__dict__)

        state["data"] = self.data.to_dict()
        state["backend"] = dict(self.sampler.backend.__dict__) if store_sampler else None
        state["model"] = None
        state["sampler"] = None

        with open(filename, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        if getattr(self, "config", None) is not None and self.config.verbose > 1:
            print(f"Result object saved to {filename}")

    @classmethod
    def load_result(cls, result_object: str | object = "result.pkl"):
        if isinstance(result_object, str):
            with open(result_object, "rb") as f:
                obj = pickle.load(f)
        else:
            obj = dict(result_object.__dict__)

        res = cls.__new__(cls) 
        res.__dict__.update(obj)

        # reconstruct data
        data = Data()
        data.from_dict(res.data)
        res.data = data

        # reconstruct backend
        if res.backend is not None:
            backend = emceebackend.Backend(dtype=np.float64)
            backend.__dict__.update(res.backend)

            # reconstruct sampler from backend
            shape = backend.shape
            log_prob = mcmc.MCMC(res.data)
            res.sampler = EnsembleSampler(*shape, log_prob, backend=backend) # dummy sampler to hold the backend
            res.backend = None # backend is now stored in the sampler, so remove it from the Result object to avoid confusion

        # rebuild convenience things that shouldn’t be pickled
        cls.initiate_data(res, res.data) # sets all_frames and nsteps

        if getattr(res, "sampler", None) is not None:
            cls.initiate_sampler(res, res.sampler) # sets burnin, thin, and default params/labels
        
        if getattr(res, "config", None) is None:
            res.config = Config()
        
        if res.config.verbose > 1:
            print("Result object loaded")

        return res