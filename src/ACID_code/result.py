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
from .utils import IntLike, Scalar
from .rassine import model
try:
    from dynesty.sampler import Sampler
    from dynesty import plotting as dyplot
except ImportError:
    Sampler = None
    dyplot = None
#TODO: utils.set_dict_defaults for plots

warnings.filterwarnings("ignore")

def _require_profiles(method):
    # Make sure all results are processed before calling method
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.data.complete: # complete is flag for if profiles have been made
            name = method.__qualname__
            if self.sampler is not None:
                if self.config.verbose>0:
                    print(f"Note: The Result object was created without the profiles processed. " \
                        f"Running {name} requires all results to be processed, " \
                        "so process_results() will now be called...")
                self.process_results()
            else:
                error = f"Cannot call {name}. The profiles attribute is not available, and no " \
                "sampler object is available to process results. Please pass an Acid/Data " \
                "instance after running ACID to the results init."
                raise ValueError(error)
        return method(self, *args, **kwargs)
    return wrapper

def _require_sampler(method):
    # Make sure sampler object is available before calling method
    sig = inspect.signature(method)
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        bound = sig.bind_partial(self, *args, **kwargs)
        inputted_sampler = bound.arguments.get("sampler", None)
        self.initiate_sampler(inputted_sampler, _method_name=method.__qualname__)
        return method(self, *args, **kwargs)
    return wrapper

@beartype
class Result:
    """
    Class to handle the results from the Acid MCMC sampling, and results processing. Fundamentally, this
    class requires two objects to run, the Sampler object and the Data object, both of which can be obtained
    from the Acid object. If one or the other is not provided, some methods will not work.
    """

    def __init__(
            self,
            data                    : Data|object,
            sampler                 : EnsembleSampler|Sampler|None = None, # type:ignore
            process_results         : bool                  = True,
            verbose                 : IntLike|bool|str|None = None,
        ) -> None:
        """
        Initialize the Result class

        Parameters
        ----------
        data : :py:class:`Data` | :py:class:`Acid`
            An Acid object or Data object (contained in Acid class). If an Acid 
            object is provided, all other arguments are taken from there. If a Data object is 
            provided, a sampler can be provided in the second argument. If a sampler object 
            is provided, it will be used as the sampler, but all other attributes will need 
            to be set manually for the Result object to be fully functional.
        sampler : :py:class:`emcee.EnsembleSampler` | :py:class:`dynesty.Sampler`, optional
            Sets and overwrites the sampler in the Data object with this if provided, by default None. 
        process_results : bool, optional
            Whether to process the results from the Acid object upon initialisation, by default True.
            If False, the profiles attribute will not be available until Result.process_results() is called.
            The process_results functions does a LSD call, which can be skipped to save time and use
            the Result object for methods that do not require the profiles attribute, such as 
            continue_sampling() or plot_walkers(). This requires a Data object with the necessary attributes, 
            and a sampler object in the initialisation, or an Acid object with the necessary attributes already set.
            By default, None.
        verbose : :py:type:`IntLike | bool | str`, optional
            Verbosity level, works exactly the same as :py:class:`Acid`, if not provided
            defaults to provided :py:class:`Acid`/:py:class:`Data` class verbosity (which itself defaults to 2).
            Overwrites any value passed trough the Data object.
        """
        # Handle the different possible cases for 1st argument input
        from .acid import Acid
        if isinstance(data, Acid):
            self.data = data.data
        elif isinstance(data, Data):
            self.data = data
        else:
            raise ValueError(f"First argument must be either an Acid or Data object. Got {type(data)} instead.")

        # Handle config and verbose options
        self.config = self.data.config # point Result.config to Data.config to keep them in sync
        self.config.verbose = verbose # property overwrites or handles if verbose input was None

        # By default set sampler_initialiated = False until sampler has been initialised in function so that self.initiate_sampler can be skipped
        self.sampler_initialiated = False

        # Handle the sampler if input, initiate if one exists
        self.sampler = sampler if sampler is not None else self.sampler # update sampler if provided, otherwise keep the same
        if self.sampler is not None:
            self.dynesty = isinstance(self.sampler, Sampler)
            self.initiate_sampler(self.sampler) # set internal variables based on sampler, sets sampler_initialiated to True

        if not self.data.complete:
            if process_results:
                if self.sampler is None:
                    raise ValueError("Cannot process results without a sampler. Please provide a sampler in the initialisation or set process_results=False.")
                else:
                    self.process_results()
            elif self.config.verbose > 0:
                print("Warning: Results not processed. Profiles attribute will not be available until " \
                "Result.process_results() is called or passed through a method.")
        elif self.sampler is None and self.config.verbose>0:
            print(f"Warning: No sampler provided or found in Data object. \n" \
            f"Some methods will not work unless a sampler is provided as a parameter or if Result.initiate_sampler(sampler) is called.")

    @_require_sampler
    def process_results(self) -> None:
        """
        Processes the MCMC sampler results to obtain the final LSD profiles and continuum fit, and errors on both.
        This is effectively the final step in the ACID pipeline, and must be run before the profiles attribute is available. 
        This is automatically called if process_results is True during :py:class:`Acid` initialization.
        This function is stored here instead of the Acid class because it is not necessary to have the final profiles to use some of the
        methods contained within this class.
        """
        t0 = time()

        # Obtain flattened samples
        if self.dynesty:
            flat_samples = self.sampler.results.samples_equal()
        else:
            flat_samples = self.sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)

        # Getting the final profile and continuum values
        nvel = len(self.data.velocities) if self.config.deterministic_profile is False else 0
        quartiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        errors = np.diff(quartiles, axis=0)
        errors = np.max(errors, axis=0)
        poly_cos      = quartiles[1, nvel:]
        poly_cos_err  = errors[nvel:] # unused for now

        if self.config.verbose > 1:
            print('Getting the final profiles...')

        # Finding error for the continuum fit
        norm_wl = self.data.wavelengths["combined_normalized"]
        coeffs = flat_samples[:, nvel:]
        ncoeffs = coeffs.shape[1]
        powers = np.vander(norm_wl, N=ncoeffs, increasing=True)

        # First check memory to see if all samples can be used
        if "SLURM_JOB_ID" in os.environ:
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
            if self.config.verbose > 1:
                print(f"Warning: Calculating continuum error with all samples may exceed available memory ({matrix_size_gb:.2f} GB required, {m_available:.2f} GB available). "
                "Calculating with a max of 1000 random samples instead.")
            indices_size = min(1000, n_samples)
            random_indices = np.random.choice(n_samples, size=indices_size, replace=False)
            coeffs = coeffs[random_indices, :]

        conts = (coeffs @ powers.T)
        continuum_error = np.std(conts, axis=0)

        # First get the combined profile, and then calculate each frame's profile if there are multiple frames.
        # If there is one frame, then the combined_profile is the same as the single frame profile.
        nframes = len(self.data.flux["input"])
        profiles = [] # switch to list format to add covariance matrix to result
        for counter in range(nframes+1):
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
            if not self.config.rassine:
                mdl = P.polyval(norm_wavelengths, poly_cos)
            else:
                # print(poly_cos.shape)
                # sys.exit()
                mdl = model(norm_wavelengths, flux, poly_cos[0])[0]

            # correcting continuum
            error = np.sqrt((error/mdl)**2 + (continuum_error/mdl)**2)
            flux /= mdl

            # Check whether we can skip alpha by reusing the same alpha, only true if the wavelength grid is identical
            condition = np.array_equal(wavelengths, self.data.wavelengths["combined"])
            alpha = self.data.alpha if condition else None

            LSD_profiles = LSD(self.data)
            LSD_profiles.run_LSD(wavelengths, flux, error, sn=sn, alpha=alpha)

            profile_f = LSD_profiles.profile_F
            profile_errors_f = LSD_profiles.profile_errors_F
            cov_z_f = LSD_profiles.cov_z_F

            if counter == 0:
                self.data.combined_profile = [profile_f, profile_errors_f, cov_z_f]
                self.data.continuum_model = mdl
            else:
                profiles.append([profile_f, profile_errors_f, cov_z_f])

        self.data.profiles = profiles # point Data.profiles to Result.profiles to keep them in sync
        self.data.results_time = time() - t0
        self.data.total_time = self.data.setup_time + self.data.mcmc_time + self.data.results_time
        self.data.complete = True

        return

    @_require_profiles
    def __getitem__(self, item) -> list|np.ndarray:
        """
        Allows indexing into the profiles array directly from the Result object.
        """
        if isinstance(item, tuple):
            # Tuples allow for array-like indexing of the list
            if len(item) == 3:
                _order, frame, velocity = item
                return self.data.alphaprofiles[frame][velocity]
            elif len(item) == 2:
                return self.data.profiles[item[0]][item[1]]
            elif len(item) == 1:
                return self.data.combined_profile[item[0]]
            else:
                raise ValueError(f"Tuple indexing must be of length 1, 2, or 3. Got {len(item)} instead.")
        elif isinstance(item, int):
            # Return just the profile or error (or cov_mat) for single int input
            if item < 0 or item > 2:
                raise ValueError(f"Integer index must be 0, 1, or 2 to specify whether to return the profile, error, or covariance matrix. Got {item} instead.")
            return self.data.combined_profile[item]
        elif isinstance(item, str):
            # Various different options for string inputs, why not
            if "error" in item.lower():
                return self.data.combined_profile[1]
            elif "cov" in item.lower():
                return self.data.combined_profile[2]
            elif "profile" in item.lower():
                return self.data.combined_profile[0]
            else:
                raise ValueError(f"String index must contain either 'error', 'cov', or 'profile' to specify which to return. Got {item} instead.")
        else:
            raise ValueError(f"Invalid index type. Must be either a tuple, int, or str. Got {type(item)} instead.")

    @_require_profiles
    def __iter__(self):
        """Allows iterating over the profiles array directly from the Result object."""
        return iter(self.data.profiles)

    def __repr__(self):
        # Only print out the sampler and data attributes, and whether profiles is available, to avoid printing large arrays
        return f"Result object with sampler={self.sampler}, data={self.data}, profiles={'available' if self.data.profiles is not None else 'not available'}"

    def __str__(self):
        return self.__repr__()

    @_require_sampler
    def continue_sampling(self, process_results:bool=True, sampler:EnsembleSampler|None=None, **kwargs) -> None:
        """
        Continue MCMC sampling for additional steps. Passes the stored sampler into a Acid instance with the saved data. See
        :py:function:`Acid.continue_sampling` for more details on the parameters that can be passed.

        Parameters
        ----------
        process_results : bool, optional
            Whether to process the results after continuing sampling, by default True.
            If False, the profiles attribute will not be updated until Result.process_results() is called.
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to continue sampling from, otherwise,
            takes the sampler from the Result object, by default None
        nsteps : :py:type:`IntLike`, optional
            Number of additional MCMC steps to run. Passed to :py:function:`Acid.continue_sampling` through **kwargs.
        max_steps : :py:type:`IntLike`, optional
            Maximum number of MCMC steps to run, by default None. Passed to :py:function:`Acid.continue_sampling` through **kwargs.
        max_steps_kwargs : dict, optional
            Additional keyword arguments to be passed to the run_mcmc_until_converged function if max_steps is specified, by default None.
            The kwargs description can be found in Acid.ACID(), they are the 4 kwargs appearing after max_steps. Typos for kwargs are silently
            ignored. Passed to :py:function:`Acid.continue_sampling` through **kwargs.
        parallel : bool, optional
            Overwrites config with whether to run the MCMC in parallel. If None, uses already existing configuration. Default is None.
            Passed to :py:function:`Acid.continue_sampling` through **kwargs.
        cores : int, optional
            Overwrites config with the number of cores to use for parallel MCMC. If None, uses already existing configuration. Default is None.
            Passed to :py:function:`Acid.continue_sampling` through **kwargs.
        moves : dict, optional
            Overwrites config with the dictionary specifying the moves to use for MCMC sampling. If None, uses already existing configuration. 
            Default is None. See :py:function:`Acid.ACID` for format. Passed to :py:function:`Acid.continue_sampling` through **kwargs.
        """
        # Note that sampler input updates the sampler stored using the @_require_sampler decorator

        # Continue sampling using the Acid method
        from .acid import Acid
        acid = Acid(data=self.data) # includes config data
        acid.continue_sampling(**kwargs) # updates self.data.sampler (or self.sampler)

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
        sampler : :py:class:`emcee.EnsembleSampler`, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        burnin : :py:type:`IntLike`, optional
            Optionally define the number of burnin steps, by default uses the burnin calculated when the sampler was initiated.
        thin : :py:type:`IntLike`, optional
            Optionally define the number of thinning steps, by default uses the thinning calculated when the sampler was initiated.
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False

        Returns
        ----------
        If return_fig is True, returns a tuple (fig, ax) of the figure and axes objects containing, else None
        """
        # Set burnin and thin to defaults if not provided
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin
        samples = self.sampler.get_chain(thin=int(thin))
        steps = np.arange(samples.shape[0]) * thin

        # Setup plot and plot the walkers for the default parameters
        naxes = len(self.default_params)
        fig, ax = plt.subplots(naxes, 1, figsize=(10, 20), sharex=True)
        for i in range(naxes):
            ax[i].plot(steps, samples[:, :, self.default_params[i]], "k", alpha=0.3)
            ax[i].axvspan(0, burnin, color="red", alpha=0.1, label="burn-in")
            ax[i].set_ylabel(self.default_param_labels[i])
        ax[-1].legend()
        ax[-1].set_xlabel("Step number")
        ax[-1].set_xlim(0, self.data.nsteps)
        ax[0].set_title('MCMC Walkers')
        plt.subplots_adjust(hspace=0.05)
        if return_fig:
            return fig, ax
        plt.show()

    @_require_sampler
    def plot_traceplot(self, return_fig:bool=False) -> None | tuple:
        if not self.dynesty:
            raise ValueError("Traceplot is only available for dynesty samplers, as emcee traceplots are already plotted in plot_walkers.")
        fig, ax = dyplot.traceplot(self.sampler.results, labels=self.default_param_labels)
        plt.suptitle('Dynesty Traceplot')
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
        if self.dynesty:
            fig, axes = dyplot.cornerplot(self.sampler.results, labels=self.default_param_labels, show_titles=True, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
            plt.suptitle('Dynesty Corner Plot')
            if return_fig:
                return fig, axes
            plt.show()
            return

        # Get samples and thin and burnin from the class variables
        samples = self.sampler.get_chain()
        samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, self.default_params]

        # Use corner.corner to handle corner plot
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
        nframes = len(self.data.profiles)
        if fig_ax is None:
            fig, ax = plt.subplots(**subplot_kwargs)
        else:
            fig, ax = fig_ax

        # Iterate through and plot frames
        for f, frame in enumerate(self.data.profiles):
            x, y, yerr = self.data.velocities, frame[0], frame[1]
            label_default = f"Frame {f+1}" if nframes > 1 else None
            # Override label in errorbar_kwargs if it is not already set, otherwise use the default label
            if "label" not in errorbar_kwargs:
                errorbar_kwargs["label"] = label_default
            ax.errorbar(x, y-1, yerr=yerr, **errorbar_kwargs)

        # Add labels and titles
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
        """
        Plots the forward model calculated from the final profiles to the combined input spectrum.

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

        # Get flat_samples which are the same samples used to calculate the final profile, alpha is OD, 
        # so convert profile back to OD and reconvert to flux for forward model
        if self.config.od:
            profile = utils.flux_to_od(self.data.combined_profile[0])
            model_flux = utils.od_to_flux(self.data.alpha @ profile) * self.data.continuum_model
        else:
            profile = self.data.combined_profile[0]-1
            model_flux = (1+(self.data.alpha @ profile)) * self.data.continuum_model

        # Plotting
        fig, ax = plt.subplots(2, 1, **subplot_kwargs)
        ax[0].plot(wavelengths, flux, color='black', linewidth=1, label='Observed Spectrum')
        ax[0].plot(wavelengths, model_flux, color='C0', linewidth=1, label='Forward Model Fit')
        ax[0].plot(wavelengths, self.data.continuum_model, color='C1', linewidth=1, label='Fitted Continuum', linestyle='--')
        ax[1].plot(wavelengths[:-10], (model_flux-flux)[:-10], color='C0', linewidth=1, label='Residuals')
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
        sampler : :py:class:`emcee.EnsembleSampler` | None, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        n_grid : :py:type:`IntLike`, optional
            Number of N values (prefix lengths) to evaluate, by default 12.
        c : float, optional
            Sokal window constant, by default 5.0.
        return_fig : bool, optional
            Whether to return the figure and axes objects, by default False
        subplot_kwargs : dict | None, optional
            Keyword arguments to be passed to plt.subplots(). Allows label overrides, by default None
        min_steps : :py:type:`IntLike`, optional
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

        ax.set_xlabel("Number of post-burnin samples per walker (N)")
        ax.set_ylabel(r"Estimated integrated autocorrelation time $\tau$")
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
        sampler : :py:class:`emcee.EnsembleSampler`, optional
            Optionally provide a different sampler to plot from, otherwise,
            takes the sampler from the Result object, by default None
        max_lag : :py:type:`IntLike`, optional
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

    def initiate_sampler(self, sampler:EnsembleSampler|Sampler|None, _method_name=None) -> None: # type:ignore
        """
        Initiates the sampler attribute from an external sampler.

        Parameters
        ----------
        sampler : :py:class:`emcee.EnsembleSampler` or object, optional
            An emcee EnsembleSampler object or a compatible sampler object to set as the sampler attribute.
        _method_name : str, optional
            Internal parameter used to track which method is calling initiate_sampler, for error messages. 
            Not intended for user input, by default None.
        """
        if self.sampler_initialiated:
            if sampler is None:
                return # sampler already initiated from initialisation, so skip the rest of the method
            # else: continues to update the sampler and internal variables based on new sampler input
        self.sampler = sampler if sampler is not None else self.sampler
        if self.sampler is None:
            if _method_name is not None:
                error_msg = f"Cannot run {_method_name} without a sampler, please pass in a sampler to the method or during initialisation."
            else:
                error_msg = "Cannot initiate sampler without a sampler stored in the instance or passed as a parameter, please pass in a sampler "
            raise AttributeError(error_msg)

        if self.dynesty:
            a=ord('a')
            alph=[chr(i) for i in range(a,a+26)]
            poly_labels = [alph[i] for i in range(self.config.poly_ord + 1)]
            self.default_param_labels = poly_labels
            self.default_params = None
            return

        # Calculate autocorr time, burnin, thin
        # Suppress output from get_autocorr_time call
        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.sampler.get_autocorr_time(quiet=True)

        self.converged = True
        if self.data.nsteps < 50 * np.max(self.tau):
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
                self.burnin = self.data.nsteps - 1000 # just the last 1000 steps
        except:
            if self.config.verbose>0:
                print(f"Warning: Could not compute autocorrelation time for burnin and thinning.\n This is likely" \
                f" due to all posterior samples being rejected (possibly by prior constraints).\n The resulting profile is likely" \
                f" wrong. Setting defaults: burnin=nsteps-1000, and thin=1.")
            self.burnin = self.data.nsteps - 1000 # just the last 1000 steps
            self.thin = 1
        
        self.burnin = int(np.clip(self.burnin, 0, self.data.nsteps-1)) # ensure burnin is at least 0 and less than total steps
        self.thin = int(np.clip(self.thin, 1, self.data.nsteps-1)) # ensure thin is at least 1, and not clipping to nsteps

        # Below is used for the parameters for the walker and corner plots
        n_poly_params = self.config.poly_ord + 1
        poly_params = np.arange(-1, -n_poly_params-1, -1).tolist()
        
        # Generates labels for the polynomial coefficients, starting from 'a' for the highest order term, and going backwards in the alphabet.
        a=ord('a')
        alph=[chr(i) for i in range(a,a+26)]
        poly_labels = [alph[i] for i in range(n_poly_params)]
        
        samples = self.sampler.get_chain(thin=self.thin, discard=self.burnin)
        if not self.config.deterministic_profile:
            max_profile_idx = np.argmax(samples[:,:,:-n_poly_params].mean(axis=(0,1)))
            poly_params.extend([-5, max_profile_idx, 1])
            poly_labels.extend(["$Z_{-1}$", "$Z_{max}$", "$Z_0$"])

        self.default_params = poly_params
        self.default_param_labels = poly_labels

    @property
    def sampler(self) -> EnsembleSampler|Sampler|None: # type:ignore
        """Returns the sampler attribute, by default is None if not saved."""
        return self.data.sampler

    @sampler.setter
    def sampler(self, value: EnsembleSampler|Sampler|None) -> None: # type:ignore
        """Sets the sampler in the data class."""
        self.data.sampler = value

    def save(self, filename:str="result.pkl", store_sampler:bool=True, size_limit:Scalar|None=1) -> None:
        """Saves the Result object to a pickle file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the Result object to, by default "result.pkl"
        store_sampler : bool, optional
            Whether to store the sampler backend in the pickle file. If False, 
            the sampler will not be stored, and the Result object will not be able to 
            continue sampling or plot walkers/corner plots
        size_limit : Scalar | None, optional
            A hard size limit to the sampler in GB.
            If the sampler exceeds this size, it will not be stored regardless of the store_sampler flag.
            This is to avoid accidentally storing very large samplers. If None, no limit is set. Default is 1GB.
            A warning will be printed if this size_limit forces the store_sampler to be False if store_sampler was set to True.
        """

        # Use the Data class's save method to handle saving, 
        # which will handle the sampler backend appropriately based on the store_sampler flag
        self.data.save(filename, store_sampler=store_sampler, size_limit=size_limit)

        if getattr(self, "config", None) is not None and self.config.verbose > 1:
            print(f"Result object saved to {filename}")

    @classmethod
    def load(cls, result:str|Result|Data="result.pkl") -> Result:
        """Loads a Result object from a pickle file or from a Data/Result object.

        Parameters
        ----------
        result : str | :py:class:`Result` | :py:class:`Data`, optional
            A pickle file name or an object with the same attributes as a saved Result object, by default "result.pkl"

        Returns
        ----------
        :py:class:`Result`
            A Result object loaded from the pickle file or from the provided object.
        """
        if isinstance(result, str):
            return cls(Data.load(result))
        elif isinstance(result, Result):
            return cls(result.data)
        elif isinstance(result, Data):
            return cls(result)
