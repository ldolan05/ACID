from math import tau
import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os, pickle

class Result:

    def __init__(self, ACID=None, ACID_HARPS=False):

        self.ACID_HARPS = ACID_HARPS

        self.nsteps = ACID.nsteps
        self.all_frames = ACID.all_frames
        self.velocities = ACID.velocities
        self.sampler = ACID.sampler
        self.model_inputs = ACID.model_inputs
        self.verbose = ACID.verbose
        self.BJDs = getattr(ACID, 'BJDs', None)
        self.profiles = getattr(ACID, 'profiles', None)
        self.errors = getattr(ACID, 'errors', None)
        self.linelist_wl = ACID.linelist_wl
        self.linelist_depths = ACID.linelist_depths

        self.ndim = len(self.model_inputs)

        self.tau = self.sampler.get_autocorr_time(quiet=True)
        self.burnin = int(2 * np.max(self.tau))
        self.thin = int(np.min(self.tau)/5)

    def __getitem__(self, item):
        """Allows indexing into the all_frames array directly from the Result object.
        """

        if self.ACID_HARPS:
            return self.BJDs[item], self.profiles[item], self.errors[item]
        else:
            return self.all_frames[item]

    def __iter__(self):
        """Allows iteration over the BJDs, profiles, and errors if ACID_HARPS was used.
        """
        if self.ACID_HARPS:
            return iter((self.BJDs, self.profiles, self.errors))
        # ACID is not subscriptable normally, only when ACID_HARPS was called 
        raise TypeError("Result is not iterable unless ACID_HARPS=True")

    def plot_walkers(self):
        """Plots the MCMC walkers for the LSD profile and continuum polynomial coefficients.
        """

        naxes = min(10, self.ndim)
        fig, axes = plt.subplots(naxes, 1, figsize=(10, 20), sharex=True)
        samples = self.sampler.get_chain(discard=self.burnin, thin=int(self.thin))
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
        samples = self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin)[:, -naxes:]
        fig = corner.corner(samples, title_fmt=".3f", title_kwargs={"fontsize": 16}, **kwargs)
        plt.suptitle('MCMC Corner Plot')
        plt.show()

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

    def plot_forward_model(self):
        """Plots the forward model fit to the observed spectrum.
        """
        
        # x, y, yerr = self.velocities, self.all_frames[0,0,0], self.all_frames[0,0,1]
        # theta_median = np.median(self.sampler.get_chain(discard=self.burnin, flat=True, thin=self.thin), axis=0)
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