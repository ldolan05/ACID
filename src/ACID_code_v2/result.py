import numpy as np
import matplotlib.pyplot as plt
import corner, sys, os

class Result:

    def __init__(self, ACID):
        self.all_frames = ACID.all_frames
        self.velocities = ACID.velocities
        self.sampler = ACID.sampler
        self.model_inputs = ACID.model_inputs

    def plot_walkers(self):
        """Plots the MCMC walkers for the LSD profile and continuum polynomial coefficients.

        Raises
        ------
        ValueError
            If sampler is not found. This occurs if ACID has not been run before
            calling this function.
        """

        if not hasattr(self, 'sampler'):
            raise ValueError("Sampler not found. Please run ACID before plotting the walkers.")

        fig, axes = plt.subplots(len(self.model_inputs), figsize=(10, 20), sharex=True)
        samples = self.sampler.get_chain()
        for i in range(len(self.model_inputs)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_ylabel(f"{i+1}")
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("Step number")
        plt.suptitle('MCMC Walkers')
        plt.show()

    def plot_corner(self, **kwargs):
        """Plots the corner plot for the LSD profile and continuum polynomial coefficients.

        Raises
        ------
        ValueError
            If flat_samples is not found. This occurs if ACID has not been run before
            calling this function.
        """

        if not hasattr(self, 'flat_samples'):
            raise ValueError("Flat samples not found. Please run ACID before plotting the corner plot.")

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

        Raises
        ------
        ValueError
            If profile and profile errors are not found. This occurs if ACID has not been run before
            calling this function.
        """

        if not hasattr(self, 'profile') or not hasattr(self, 'profile_err'):
            raise ValueError("Profile and profile errors not found. Please run ACID before plotting the profile.")

        fig, ax = plt.subplots(figsize=(10, 6))
        x, y, yerr = self.velocities, self.all_frames[0,0,0], self.all_frames[0,0,1]
        ax.errorbar(x, y, yerr=yerr, fmt='o', ecolor='red', capsize=2, label='LSD Profile with Errors')
        ax.set_title('LSD Profile')
        ax.set_xlabel('Velocity (km/s)')
        ax.set_ylabel('Normalised Flux')
        ax.axhline(0, color='black', linestyle='--', linewidth=0.7)
        ax.legend()
        ax.grid()
        plt.show()
