from math import tau
import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os, pickle, warnings, contextlib
from beartype import beartype
from numpy import integer as npint

warnings.filterwarnings("ignore")

__all__ = ['Result']

def _require_all_results(method):
    # Make sure all results are processed before calling method
    def wrapper(self, *args, **kwargs):
        if self.production_run:
            name = method.__qualname__
            if self.verbose>0:
                print(f"Note: The Result object was in production_run mode. Running {name} " \
                      "requires all results to be processed, so process_results() has been " \
                      "called automatically.")
            self.process_results()
        return method(self, *args, **kwargs)
    return wrapper

def _require_Acid(method):
    # Make sure Acid object is available before calling method
    def wrapper(self, *args, **kwargs):
        if self.Acid is None:
            error = f"Cannot call {method.__qualname__}. The Acid object is not available in this " \
            "Result instance. This can occur if Acid was set to None to allow for pickling in the " \
            "case that multiple orders or frames were used."
            raise ValueError(error)
        return method(self, *args, **kwargs)
    return wrapper

@beartype
class Result:
    """Class to handle the results from the Acid MCMC sampling, and results processing."""

    def __init__(self, Acid, ACID_HARPS:bool=False, production_run:bool=False):
        """Initiate Result class

        Parameters
        ----------
        Acid : object
            An Acid object after MCMC sampling has been performed.
        ACID_HARPS : bool, optional
            Whether the ACID_HARPS function was used, by default False
        production_run : bool, optional
            Whether Acid was run in production mode, by default False
        """
        self.Acid = Acid

        # This should be a temporary measure until Acid can be pickled properly
        self.sampler = Acid.sampler

        self.ACID_HARPS = ACID_HARPS
        self.production_run = production_run
        self.nsteps = Acid.nsteps
        self.velocities = Acid.velocities
        self.model_inputs = Acid.model_inputs
        self.verbose = Acid.verbose
        self.order_range = Acid.order_range
        self.BJDs = getattr(Acid, 'BJDs', None)
        self.profiles = getattr(Acid, 'profiles', None)
        self.errors = getattr(Acid, 'errors', None)

        if not production_run:
            self.all_frames = Acid.all_frames
        else:
            self.all_frames = None

        self.ndim = len(self.model_inputs)

        # Calculate autocorr time, burnin, thin
        # Suppress output from get_autocorr_time call
        with open(os.devnull, "w") as devnull, \
            contextlib.redirect_stdout(devnull), \
            contextlib.redirect_stderr(devnull):
            self.tau = self.Acid.sampler.get_autocorr_time(quiet=True)
        
        if self.nsteps < 50 * np.max(self.tau):
            if self.verbose>1:
                print("The number of MCMC steps is less than 50 times the maximum autocorrelation " \
                "time. The sampler may not have converged. Consider running more steps or checking " \
                "the walker plots.")

        self.burnin = int(2 * np.max(self.tau))
        self.thin = int(np.min(self.tau)/5)

    @_require_all_results
    def __getitem__(self, item):
        """Allows indexing into the all_frames array directly from the Result object.
        """

        if self.ACID_HARPS:
            return self.BJDs[item], self.profiles[item], self.errors[item]
        else:
            return self.all_frames[item]

    @_require_all_results
    def __iter__(self):
        """Allows iteration over the BJDs, profiles, and errors if ACID_HARPS was used.
        """
        if self.ACID_HARPS:
            return iter((self.BJDs, self.profiles, self.errors))
        # Acid is not subscriptable normally, only when ACID_HARPS was called 
        raise TypeError("Result is not iterable unless ACID_HARPS=True")

    @_require_Acid
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
    
    @_require_Acid
    def process_results(self):
        """Processes the MCMC results to extract the LSD profiles and errors.
        """
        self.Acid.process_results(return_result=False)
        self.production_run = False
        self.all_frames = self.Acid.all_frames

    def plot_walkers(self, burnin:int|npint|None=None, thin:int|npint|None=None):
        """Plots, at maximum, the last 10 MCMC walkers for the LSD profile and continuum 
        polynomial coefficients.

        Parameters
        ----------
        burnin : int | None, optional
            Optionally define the number of burnin steps, by default None
        thin : int | None, optional
            Optionally define the number of thinning steps, by default None
        """
        burnin = burnin if burnin is not None else self.burnin
        thin = thin if thin is not None else self.thin

        naxes = min(10, self.ndim)
        fig, axes = plt.subplots(naxes, 1, figsize=(10, 20), sharex=True)
        samples = self.sampler.get_chain(discard=burnin, thin=int(thin))
        steps = np.arange(samples.shape[0]) * thin + burnin
        for i in range(naxes):
            ax = axes[i]
            ax.plot(steps, samples[:, :, i], "k", alpha=0.3)
            ax.axvspan(0, burnin, color="red", alpha=0.1, label="burn-in")

        axes[-1].legend()
        axes[-1].set_xlabel("Step number")
        axes[-1].set_xlim(0, self.nsteps)
        axes[0].set_title('MCMC Walkers')
        plt.subplots_adjust(hspace=0.05)
        plt.show()

    def plot_corner(self, **kwargs):
        """Creates a corner plot for at maximum the last 8 LSD profile and continuum polynomial coefficients.
        
        Parameters
        ----------
        **kwargs:
            Additional keyword arguments to pass to corner.corner().
        """

        naxes = min(8, self.ndim)
        samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, -naxes:]
        fig = corner.corner(samples, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
        plt.suptitle('MCMC Corner Plot')
        plt.show()

    @_require_all_results
    def plot_profiles(
        self,
        grid            :bool      = True,
        labels          :dict|None = None,
        return_fig      :bool      = False,
        subplot_kwargs  :dict|None = None,
        errorbar_kwargs :dict|None = None
        ):
        """Plots the LSD profile result from Acid.

        Parameters
        ----------
        grid : bool, optional
            Show or hide grid, by default True
        labels : dict, optional
            Keys: 'xlabel', 'ylabel', and 'title'. Allows label overrides., by default None
        return_fig : bool, optional
            Whether to return the figure and axis objects instead of showing the plot, by default False
        subplot_kwargs : dict, optional
            Keyword arguments to be passed to plt.subplots(), by default None
        errorbar_kwargs : dict, optional
            Keyword arguments to be passed to ax.errorbar(), by default None
        """
        # Set default errorbar kwargs
        errorbar_kwargs = dict(errorbar_kwargs or {})
        errorbar_defaults = {
            "fmt"      : ".-",
            "ecolor"   : "red",
            "linewidth": 1,
        }
        for key, value in errorbar_defaults.items():
            errorbar_kwargs.setdefault(key, value)

        # Set default subplot kwargs
        subplot_kwargs = dict(subplot_kwargs or {})
        subplot_kwargs.setdefault("figsize", (10, 6))

        # Set default labels
        default_labels = {
            "title" : "Acid Profile",
            "xlabel": "Velocity (km/s)",
            "ylabel": "Normalised Flux"
        }
        labels = dict(labels or {})
        for key, value in default_labels.items():
            labels.setdefault(key, value)

        # Set useful variables
        nframes = len(self.all_frames)
        norders = len(self.all_frames[0])
        frames = np.copy(self.all_frames)
        fig, ax = plt.subplots(**subplot_kwargs)
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
                ax.errorbar(x, y, yerr=yerr, label=f"Frame {f+1}, Order {o+1}", **errorbar_kwargs)

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

    @_require_all_results
    def plot_forward_model(self):
        """Plots the forward model fit to the observed spectrum.
        """
        raise NotImplementedError("plot_forward_model is not yet implemented")
        # x, y, yerr = self.velocities, self.all_frames[0,0,0], self.all_frames[0,0,1]
        # theta_median = np.median(self.Acid.sampler.get_chain(discard=self.burnin, flat=True,
        # thin=self.thin), axis=0)
        # from src.Acid_code_v2.mcmc_utils import model_func
        # model = model_func(theta_median, x)

        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.errorbar(x, y, yerr=yerr, ecolor="red", linewidth=1, label='Observed Spectrum')
        # ax.plot(x, model, color='blue', linewidth=1, label='Forward Model Fit')
        # ax.set_title('Forward Model Fit to Observed Spectrum')
        # ax.set_xlabel('Velocity (km/s)')
        # ax.set_ylabel('Normalised Flux')
        # ax.axhline(0, color='black', linestyle='--', linewidth=1)
        # ax.legend()
        # ax.grid()
        # plt.show()

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