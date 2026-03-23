from __future__ import annotations
from time import time
import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os, pickle, warnings, contextlib, functools, inspect, psutil
from emcee import EnsembleSampler
import emcee.backends.backend as emceebackend
from beartype import beartype
from scipy.interpolate import interp1d
from numpy.polynomial import polynomial as P
from .lsd import LSD
from . import mcmc
from . import utils
from .data import Data
from .data import Config
from .utils import FloatLike, IntLike, Scalar, Array1D, Array2D, ArrayAnyD

warnings.filterwarnings("ignore")

def _require_profiles(method):
    # Make sure all results are processed before calling method
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.profiles is None:
            name = method.__qualname__
            if self.sampler is not None and self.data is not None:
                if self.config.verbose>0:
                    print(f"Note: The Result object was created without the profiles processed. " \
                        f"Running {name} requires all results to be processed, " \
                        "so process_results() will be called automatically...")
                self.process_results()
            else:
                error = f"Cannot call {name}. The profiles attribute is not available, and no " \
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
            "Result instance."
            raise ValueError(error)
        return method(self, *args, **kwargs)
    return wrapper

def _require_sampler(method):
    # Make sure sampler object is available before calling method
    # TODO: Make save for a thinned chain, discard sampler unless store_sampler is True
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
            sampler                 : EnsembleSampler|None  = None,
            process_results         : bool                  = True,
            verbose                 : IntLike|bool|str|None = None,
        ) -> None:
        """Initiate the Result class

        Parameters
        ----------
        Acid_or_Data_or_Sampler : Acid | Data | emcee.EnsembleSampler
            An Acid object, Data object (contained in Acid class), or sampler object. If an Acid 
            object is provided, all other arguments are taken from there. If a Data object is 
            provided, a sampler can be provided in the second argument. If a sampler object 
            is provided, it will be used as the sampler, but all other attributes will need 
            to be set manually for the Result object to be fully functional.
        sampler : emcee.EnsembleSampler, optional
            A sampler object to use if the Data object was provided. If an Acid object 
            was provided, the sampler will be taken from there. If a sampler object was
            provided in the first argument, this will be ignored (with a warning), by default None
        process_results : bool, optional
            Whether to process the results from the Acid object upon initialisation, by default True.
            If False, the profiles attribute will not be available until Result.process_results() is called.
            The process_results functions does a LSD call, which can be skipped to save time and use
            the Result object for methods that do not require the profiles attribute, such as 
            continue_sampling() or plot_walkers(). This requires a Data object with the necessary attributes, 
            and a sampler object in the initialisation, or an Acid object with the necessary attributes already set.
            By default, None.
        verbose : int | bool | str, optional
            Verbosity level, works exactly the same as Acid verbosity, if not provided
            defaults to provided Acid class verbosity otherwise defaults to 2.
        # production_run : bool, optional
        #     Whether Acid was run in production mode, by default False
        """
        obj = Acid_or_Data_or_Sampler
        self.sampler    = None
        self.data       = None
        self.profiles = None
        self.config     = Config() # default config, will be updated if Acid or Data object is provided
        self.slurm      = "SLURM_JOB_ID" in os.environ

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
        # Data.profiles to Result.profiles to keep them in sync.
        if process_results:
            self.process_results() # sets self.data.profiles, and points self.profiles to self.data.profiles
        else:
            if self.config.verbose>0:
                print("Warning: Results not processed. profiles attribute will not be available until " \
                "Result.process_results() is called.")

    @_require_data
    @_require_sampler
    def process_results(self) -> None:
        """Processes the MCMC sampler results to obtain the final LSD profiles and continuum fit, and errors on both."""
        t0 = time()

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

        # First check memory to see if all samples can be used
        if self.slurm:
            available_memory = int(os.environ.get('SLURM_MEM_PER_NODE')) # in MB
            available_memory *= 1e6  # Convert to bytes as in the else statement below
        else:
            available_memory = psutil.virtual_memory().available
        m_available = available_memory * 1e-9 * 0.8 # in GB, with 0.8 factor safety gap
        n_samples, ncoeffs = coeffs.shape
        npix = powers.shape[0]
        matrix_size_gb = (2 * n_samples * npix + n_samples * ncoeffs + npix * ncoeffs) * 8 * 1e-9

        # If memory exceeded, fallback to using 1000 random samples
        if matrix_size_gb > m_available:
            if self.config.verbose > 0:
                print(f"Warning: Calculating continuum error with all samples may exceed available memory ({matrix_size_gb:.2f} GB required, {m_available:.2f} GB available). "
                "Calculating with a max of 1000 random samples instead.")
            indices_size = min(1000, n_samples)
            random_indices = np.random.choice(n_samples, size=indices_size, replace=False)
            coeffs = coeffs[random_indices, :]

        conts = (coeffs @ powers.T)
        continuum_error = np.std(conts, axis=0)

        self.profiles = np.zeros((len(self.data.flux["input"]), 2, len(self.data.velocities)))

        # First get the combined profile, and then calculate each frame's profile if there are multiple frames.
        # If there is one frame, then the combined_profile is the same as the single frame profile.

        for counter in range(len(self.data.flux["input"])+1):
            if counter == 0:
                flux = np.copy(self.data.flux["combined"])
                error = np.copy(self.data.errors["combined"])
                wavelengths = np.copy(self.data.wavelengths["combined"])
                sn = np.copy(self.data.sn["combined"])
                error[self.data.residual_masks] = 1e12
            else:
                flux = np.copy(self.data.flux["input"][counter-1])[self.data.nanmask]
                error = np.copy(self.data.errors["input"][counter-1])[self.data.nanmask]
                wavelengths = np.copy(self.data.wavelengths["input"][counter-1])[self.data.nanmask]
                sn = np.copy(self.data.sn["input"][counter-1])

                # Masking based off residuals interpolated onto new wavelength grid
                reference_wave = self.data.wavelengths["input"][np.nanargmax(self.data.sn["input"])]
                reference_wave = reference_wave[self.data.nanmask]

                reference_mask = np.zeros_like(reference_wave, dtype=bool)
                reference_mask[self.data.residual_masks] = True
                reference_interp1d = interp1d(reference_wave, reference_mask.astype(float), kind="nearest", bounds_error=False, fill_value=0.0)
                interpolated_mask = reference_interp1d(wavelengths) > 0.5
                error[interpolated_mask] = 1e12

            # Build continuum model
            a, b = utils.get_normalisation_coeffs(wavelengths)
            norm_wavelengths = (a*wavelengths)+b
            mdl = P.polyval(norm_wavelengths, self.poly_cos)

            # correcting continuum
            error = np.sqrt((error/mdl)**2 + (continuum_error/mdl)**2)
            flux /= mdl

            # Check whether we can skip alpha by reusing the same alpha, only true if the wavelength grid is identical
            condition = np.all(wavelengths==self.data.wavelengths["combined"])
            alpha = self.data.alpha if condition is True else None

            LSD_profiles = LSD(self.data)
            LSD_profiles.run_LSD(wavelengths, flux, error, sn=sn, alpha=alpha)

            profile_f = LSD_profiles.profile_F
            profile_errors_f = LSD_profiles.profile_errors_F

            if counter == 0:
                self.combined_profile = [profile_f, profile_errors_f]
            else:
                self.profiles[counter-1] = [profile_f, profile_errors_f]

        self.data.profiles = self.profiles # point Data.profiles to Result.profiles to keep them in sync
        self.data.combined_profiles = self.combined_profile
        self.data.get_profiles_time = time() - t0
        self.data.full_run_time = self.data.initialisation_time + self.data.mcmc_time + self.data.get_profiles_time

        return

    @_require_profiles
    def __getitem__(self, item) -> np.ndarray:
        """Allows indexing into the profiles array directly from the Result object.
        """
        if isinstance(item, tuple):
            if len(item) == 3:
                # Legacy support for indexing style
                _order, frame, velocity = item
                item = (frame, velocity)
            elif len(item) == 2:
                item = item
            elif len(item) == 1:
                return self.combined_profile[item[0]]
        elif isinstance(item, int):
            return self.combined_profile[item]
        elif isinstance(item, str):
            if "combined" in item.lower():
                return self.combined_profile
            elif "profile" in item.lower():
                return self.profiles
        return self.profiles[item]

    @_require_profiles
    def __iter__(self) -> np.ndarray:
        """Allows iterating over the profiles array directly from the Result object.
        """
        return iter(self.profiles)

    def __repr__(self):
        # Only print out the sampler and data attributes, and whether profiles is available, to avoid printing large arrays
        return f"Result object with sampler={self.sampler}, data={self.data}, profiles={'available' if self.profiles is not None else 'not available'}"

    @_require_data
    @_require_sampler
    def continue_sampling(self, process_results:bool=True, sampler:EnsembleSampler|None=None, **kwargs) -> None:
        """Continue MCMC sampling for additional steps. Passes the stored sampler into a Acid instance with the saved data. See
        Acid.continue_sampling() for more details on the parameters that can be passed.

        Parameters
        ----------
        nsteps : int | None, optional
            Number of additional MCMC steps to run. Passed to Acid.continue_sampling with the stored sampler.
        max_steps : int | None, optional
            Maximum number of MCMC steps to run, by default None. Passed to Acid.continue_sampling with the stored sampler.
        max_steps_kwards : dict, optional
            Additional keyword arguments to be passed to the run_mcmc_until_converged function if max_steps is specified, by default None.
            The kwargs description can be found in Acid.ACID(), they are the 4 kwargs appearing after max_steps. Typos for kwargs are silently
            ignored. Passed to Acid.continue_sampling with the stored sampler.
        process_results : bool, optional
            Whether to process the results after continuing sampling, by default True.
            If False, the profiles attribute will not be updated until Result.process_results() is called.
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to continue sampling from, otherwise,
            takes the sampler from the Result object, by default None
        """
        if type(process_results) is int:
            raise ValueError("The process_results attribute must be a boolean, not an integer. Did you mean to set nsteps? If so, specificy nsteps=nsteps.")

        from .acid import Acid
        acid = Acid(data=self.data) # includes config data
        self.sampler = acid.continue_sampling(self.sampler, **kwargs)

        self.initiate_sampler(self.sampler) # update internal variables to match new sampler

        if process_results:
            self.process_results() # update profiles
        else:
            if self.config.verbose>0:
                print("Warning: Results not processed. profiles attribute will not be available until " \
                "Result.process_results() is called.")

    @_require_sampler
    def plot_walkers(
        self,
        sampler    : EnsembleSampler|None = None,
        burnin     : IntLike|None         = None,
        thin       : IntLike|None         = None,
        return_fig : bool                 = False
        ) -> None | tuple:
        """Plots, at maximum, the last 10 MCMC walkers for the LSD profile and continuum 
        polynomial coefficients.

        Parameters
        ----------
        sampler : emcee.EnsembleSampler, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        burnin : int | None, optional
            Optionally define the number of burnin steps, by default uses the burnin calculated when the sampler was initiated.
        thin : int | None, optional
            Optionally define the number of thinning steps, by default uses the thinning calculated when the sampler was initiated.
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False

        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing, else None
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
    def plot_corner(
        self,
        sampler    :EnsembleSampler|None = None,
        return_fig :bool                 = False,
        **kwargs,
        ) -> None | tuple:
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
        
        Returns
        ----------
        If return_fig is True, returns the figure object containing the corner plot, else None
        """

        samples = self.sampler.get_chain()

        samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, self.default_params]

        fig = corner.corner(samples, labels=self.default_param_labels, show_title=True, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
        plt.suptitle('MCMC Corner Plot')
        if return_fig:
            return fig
        plt.show()

    @_require_profiles
    def plot_profiles(
        self,
        grid            :bool      = True,
        labels          :dict|None = None,
        return_fig      :bool      = False,
        subplot_kwargs  :dict|None = None,
        errorbar_kwargs :dict|None = None,
        fig_ax                     = None,
        ) -> None | tuple:
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
        
        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing the plot.
        Otherwise, displays the plot and returns None.
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
        nframes = len(self.profiles)
        if fig_ax is None:
            fig, ax = plt.subplots(**subplot_kwargs)
        else:
            fig, ax = fig_ax

        for f, frame in enumerate(self.profiles):
            x, y, yerr = self.data.velocities, frame[0], frame[1]
            label_default = f"Frame {f+1}" if nframes > 1 else None
            # Override label in errorbar_kwargs if it is not already set, otherwise use the default label
            if "label" not in errorbar_kwargs:
                errorbar_kwargs["label"] = label_default
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

    @_require_profiles
    def plot_forward_model(
        self,
        grid            :bool      = True,
        labels          :dict|None = None,
        return_fig      :bool      = False,
        subplot_kwargs  :dict|None = None,
        ) -> None | tuple:
        """Plots the forward model calculated from the final profiles to the combined input spectrum.

        Parameters
        ----------
        grid : bool, optional
            Show or hide grid, by default True
        labels : dict | None, optional
            Keys: 'xlabel', 'ylabel', 'title', and 'residuals_ylabel'. Allows label overrides, by default None
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None
        
        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing the plot.
        Otherwise, displays the plot and returns None.
        """
        
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
        wavelengths = self.data.wavelengths["combined"]
        flux = self.data.flux["combined"]

        a, b = utils.get_normalisation_coeffs(wavelengths)
        profile = utils.flux_to_od(self.combined_profile[0])

        # Get flat_samples which are the same samples used to calculate the final profile
        flat_samples = self.sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)
        nvel = len(self.data.velocities) if self.config.deterministic_profile is False else 0
        poly_samples = flat_samples[:, nvel:] # continuum polynomial samples
        continuum_model = P.polyval(wavelengths*a+b, np.median(poly_samples, axis=0))
        model_flux = np.exp(- (self.data.alpha @ profile)) * continuum_model

        # Plotting
        fig, ax = plt.subplots(2, 1, **subplot_kwargs)
        ax[0].plot(wavelengths, flux, color='black', linewidth=1, label='Observed Spectrum')
        ax[0].plot(wavelengths, model_flux, color='C0', linewidth=1, label='Forward Model Fit')
        ax[0].plot(wavelengths, continuum_model, color='C1', linewidth=1, label='Fitted Continuum', linestyle='--')
        ax[1].plot(wavelengths, flux - model_flux, color='C0', linewidth=1, label='Residuals')
        ax[1].axhline(0, color='black', linestyle='--', linewidth=1)
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
        sampler        : EnsembleSampler|None = None,
        burnin         : IntLike|None         = None,
        thin           : IntLike|None         = None,
        n_grid         : IntLike              = 12,
        c              : float                = 5.0,
        return_fig     : bool                 = False,
        subplot_kwargs : dict|None            = None,
        min_steps      : IntLike              = 100
        ) -> None | tuple:
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
            Sokal window constant, by default 5.0.
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
        sampler        : EnsembleSampler|None = None,
        max_lag        : IntLike|None         = None,
        return_fig     : bool                 = False,
        subplot_kwargs : dict|None            = None,
        ) -> None | tuple:
        """
        Plot the autocorrelation function (ACF) for each parameter, averaged across walkers.
        
        Parameters
        ----------
        sampler : emcee.EnsembleSampler, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        max_lag : int, optional
            Maximum lag to plot, by default None (plots up to min(5000, nsteps-1))
        return_fig : bool, optional
            Whether to return the figure and axes objects, by default False
        subplot_kwargs : dict, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None

        Returns
        -------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing 
        the plot. Otherwise, displays the plot and returns None.
        """
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

    def initiate_sampler(self, sampler:EnsembleSampler|None) -> None:
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
                self.burnin = int(3 * np.max(self.tau))
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

    def initiate_data(self, data:Data) -> None:
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

        self.profiles = self.data.profiles
        self.nsteps     = self.data.nsteps

        # For convenience, let the user call the model without needing to input all required args
        MCMC_class = mcmc.MCMC(self.data)
        self.model = MCMC_class.run_model_function

    @_require_data
    @_require_sampler
    def save_result(self, filename:str="result.pkl", store_sampler:bool=True) -> None:
        """Saves the Result object to a pickle file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the Result object to, by default "result.pkl"
        store_sampler : bool, optional
            Whether to store the sampler backend in the pickle file. If False, 
            the sampler will not be stored, and the Result object will not be able to 
            continue sampling or plot walkers/corner plots
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
    def load_result(cls, result_object:str|object="result.pkl"):
        """Loads a Result object from a pickle file or from an object with the same attributes as a saved Result object.

        Parameters
        ----------
        result_object : str | object, optional
            A pickle file name or an object with the same attributes as a saved Result object, by default "result.pkl"

        Returns
        ----------
        Result
            A Result object loaded from the pickle file or from the provided object.
        """
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
        cls.initiate_data(res, res.data) # sets profiles and nsteps

        if getattr(res, "sampler", None) is not None:
            cls.initiate_sampler(res, res.sampler) # sets burnin, thin, and default params/labels
        
        if getattr(res, "config", None) is None:
            res.config = Config()
        
        if res.config.verbose > 1:
            print("Result object loaded")

        return res