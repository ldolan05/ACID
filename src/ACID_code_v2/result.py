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

    def __getitem__(self, item):
        """Allows indexing into the all_frames array directly from the Result object.
        """

        if self.ACID_HARPS:
            return self.BJDs[item], self.profiles[item], self.errors[item]
        else:
            return self.all_frames[item]

    def __iter__(self):

        if self.ACID_HARPS:
            return iter((self.BJDs, self.profiles, self.errors))
        # ACID is not subscriptable normally, only when ACID_HARPS was called 
        raise TypeError("Result is not iterable unless ACID_HARPS=True")

    def plot_walkers(self):
        """Plots the MCMC walkers for the LSD profile and continuum polynomial coefficients.
        """

        fig, axes = plt.subplots(len(self.model_inputs)-40, figsize=(10, 20), sharex=True)
        samples = self.sampler.get_chain()
        for i in range(len(self.model_inputs)-40):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)

        axes[-1].set_xlabel("Step number")
        axes[0].set_title('MCMC Walkers')
        plt.show()

    def plot_corner(self, **kwargs):
        """Plots the corner plot for the LSD profile and continuum polynomial coefficients.
        """
        tau = self.sampler.get_autocorr_time(quiet=True)
        burnin = int(2 * np.max(tau))
        thin = int(np.min(tau)/2)
        samples = self.sampler.get_chain(discard=burnin, flat=True, thin=thin)[:, :5]
        # print(samples.shape)
        # sys.exit()
        print(np.max(tau))
        print(samples.shape)
        # sys.exit()
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

    def save_result(self, filename="tests/test_data/result.pkl"):
        """Saves the Result object to a pickle file.
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f)
        if self.verbose is True:
            print(f"Result object saved to {filename}")

    @classmethod
    def load_result(cls, filename="tests/test_data/result.pkl"):
        """Loads a Result object from a pickle file.
        """

        with open(filename, "rb") as f:
            obj = pickle.load(f)
        obj.__class__ = cls
        return obj