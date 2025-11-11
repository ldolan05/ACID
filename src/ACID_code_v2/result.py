import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os, pickle

class Result:

    def __init__(self, ACID=None, savefile=None):

        self.ACID = ACID # Ideally this gets deleted later to save memory, for now can use to debug
        self.nsteps = ACID.nsteps
        self.all_frames = ACID.all_frames
        self.velocities = ACID.velocities
        self.sampler = ACID.sampler
        self.model_inputs = ACID.model_inputs
        self.verbose = ACID.verbose

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

        samples = self.sampler.get_chain(discard=int(np.floor(self.nsteps-1000)), flat=True, thin=15)
        tau = self.sampler.get_autocorr_time()
        print(tau)
        print(samples.shape)
        sys.exit()
        fig = corner.corner(samples, title_fmt=".3f", title_kwargs={"fontsize": 12}, **kwargs)
        plt.suptitle('MCMC Corner Plot', y=1.02)
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