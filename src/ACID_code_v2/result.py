from math import tau
import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os, pickle, warnings

warnings.filterwarnings("ignore")

def _require_all_results(method):
    def wrapper(self, *args, **kwargs):
        if self.production_run:
            name = method.__qualname__
            raise RuntimeError(f"{name} cannot be called because ACID was run in production mode. Use "
                               "ACID.process_results() to extract all results.")
        return method(self, *args, **kwargs)
    return wrapper

class Result:
    """Class to handle the results from the ACID MCMC sampling, and results processing."""

    def __init__(self, ACID=None, ACID_HARPS=False, production_run=False):

        self.ACID = ACID

        self.ACID_HARPS = ACID_HARPS
        self.production_run = production_run
        self.nsteps = ACID.nsteps
        self.velocities = ACID.velocities
        self.model_inputs = ACID.model_inputs
        self.verbose = ACID.verbose
        self.BJDs = getattr(ACID, 'BJDs', None)
        self.profiles = getattr(ACID, 'profiles', None)
        self.errors = getattr(ACID, 'errors', None)
        self.linelist_wl = ACID.linelist_wl
        self.linelist_depths = ACID.linelist_depths
        if not production_run:
            self.all_frames = ACID.all_frames
        else:
            self.all_frames = None

        self.ndim = len(self.model_inputs)

        self.tau = self.ACID.sampler.get_autocorr_time(quiet=True)
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
        # ACID is not subscriptable normally, only when ACID_HARPS was called 
        raise TypeError("Result is not iterable unless ACID_HARPS=True")

    def continue_sampling(self, nsteps, production_run=None):
        """Continue MCMC sampling for additional steps.

        Parameters
        ----------
        nsteps : int
            Number of additional MCMC steps to run.
        production_run : bool, optional
            Whether to set the run as a production run, by default None, keeping the current setting.
            The default is False from ACID.
        """
        if production_run is not None:
            self.production_run = production_run

        self.ACID.continue_sampling(nsteps)

        # Update attributes from ACID object
        self.nsteps = self.ACID.nsteps
        if not self.production_run:
            self.all_frames = self.ACID.all_frames

        # Recalculate autocorr time, burnin, thin
        self.tau = self.ACID.sampler.get_autocorr_time(quiet=True)
        self.burnin = int(2 * np.max(self.tau))
        self.thin = int(np.min(self.tau)/5)
    
    def process_results(self):
        """Processes the MCMC results to extract the LSD profiles and errors.
        """
        self.ACID.process_results()
        self.production_run = False
        self.all_frames = self.ACID.all_frames

    def plot_walkers(self):
        """Plots the MCMC walkers for the LSD profile and continuum polynomial coefficients.
        """

        naxes = min(10, self.ndim)
        fig, axes = plt.subplots(naxes, 1, figsize=(10, 20), sharex=True)
        samples = self.ACID.sampler.get_chain(discard=self.burnin, thin=int(self.thin))
        steps = np.arange(samples.shape[0]) * self.thin + self.burnin
        for i in range(naxes):
            ax = axes[i]
            ax.plot(steps, samples[:, :, i], "k", alpha=0.3)
            ax.axvspan(0, self.burnin, color="red", alpha=0.1, label="burn-in")

        axes[-1].legend()
        axes[-1].set_xlabel("Step number")
        axes[-1].set_xlim(0, self.nsteps)
        axes[0].set_title('MCMC Walkers')
        plt.subplots_adjust(hspace=0.05)
        plt.show()

    def plot_corner(self, **kwargs):
        """Plots the corner plot for the LSD profile and continuum polynomial coefficients.
        """

        naxes = min(8, self.ndim)
        samples = self.ACID.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, -naxes:]
        fig = corner.corner(samples, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
        plt.suptitle('MCMC Corner Plot')
        plt.show()

    @_require_all_results
    def plot_profile(self):
        """Plots the LSD profile result from ACID.
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        x, y, yerr = self.velocities, self.all_frames[0,0,0], self.all_frames[0,0,1]
        ax.errorbar(x, y, yerr=yerr, ecolor="red", linewidth=1, label='LSD Profile with Errors')
        ax.set_title('LSD Profile')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Normalised Flux')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.legend()
        ax.grid()
        plt.show()

    @_require_all_results
    def plot_forward_model(self):
        """Plots the forward model fit to the observed spectrum.
        """
        raise NotImplementedError("plot_forward_model is not yet implemented")
        # x, y, yerr = self.velocities, self.all_frames[0,0,0], self.all_frames[0,0,1]
        # theta_median = np.median(self.ACID.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin), axis=0)
        # from src.ACID_code_v2.mcmc_utils import model_func
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

    def save_result(self, filename="result.pkl"):
        """Saves the Result object to a pickle file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the Result object to, by default "result.pkl"
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f)
        if self.verbose is True:
            print(f"Result object saved to {filename}")

    @classmethod
    def load_result(cls, filename):
        """Loads a Result object from a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file to load the Result object from.
        """

        with open(filename, "rb") as f:
            obj = pickle.load(f)
        obj.__class__ = cls
        return obj