import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from beartype import beartype
from scipy.special import wofz

@beartype
class Profiles:
    """A class for fitting spectral line profiles such as Voigt and Gaussian profiles.
    """
    def __init__(self, velocities, flux, flux_err=None):
        """Initializes the Profiles class with velocity, flux, and optional flux error data.

        Parameters
        ----------
        velocities : array_like
            The velocity values corresponding to the spectral line profile.
        flux : array_like
            The flux values of the spectral line profile.
        flux_err : array_like, optional
            The errors associated with the flux values, by default None.
        """

        self.velocities = velocities
        self.flux = flux
        self.flux_err = flux_err

        self.fitted_x = np.linspace(np.min(velocities), np.max(velocities), 1000)
        pass

    def plot_fit(self, model:str|None='all'):
        """Plots the original data and the fitted profile if available.
        
        Parameters
        ----------
        model : str | None, optional
            The type of model to plot. String options are 'voigt', 'gaussian', 
            'lorentzian', or 'all'. Choosing None will plot whichever models have 
            already been fitted for, by default 'all'.
        Returns
        -------
        None
        """
        models = ["voigt", "gaussian", "lorentzian"]
        if model is not None:
            model = model.lower()
            if model not in models + ['all']:
                raise ValueError("Model must be 'voigt', 'gaussian', 'lorentzian' or 'all'.")
        
        fitted_y = []
        fit_on_x = []
        labels = []
        colors = []
        if model in ['voigt', 'all']:
            if not hasattr(self, 'fitted_voigt'):
                self.fit_voigt()
            fitted_y.append(self.fitted_voigt)
            fit_on_x.append(self.voigt_on_x)
            labels.append('Voigt Fit')
            colors.append('C1')
        if model in ['gaussian', 'all']:
            if not hasattr(self, 'fitted_gaussian'):
                self.fit_gaussian()
            fitted_y.append(self.fitted_gaussian)
            fit_on_x.append(self.gaussian_on_x)
            labels.append('Gaussian Fit')
            colors.append('C2')
        if model in ['lorentzian', 'all']:
            if not hasattr(self, 'fitted_lorentzian'):
                self.fit_lorentzian()
            fitted_y.append(self.fitted_lorentzian)
            fit_on_x.append(self.lorentzian_on_x)
            labels.append('Lorentzian Fit')
            colors.append('C3')

        fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax[0].plot(self.velocities, self.flux, 'b.', label='ACID Profile', color='C0')
        ax[1].axhline(0, color='C0', linestyle='--')
        for y_fit, y_fit_on_x, label, color in zip(fitted_y, fit_on_x, labels, colors):
            ax[0].plot(self.fitted_x, y_fit, label=label, color=color)
            ax[1].plot(self.velocities, y_fit_on_x-self.flux, label=label, color=color)
        ax[0].set_xlabel('Velocity')
        ax[0].set_title('Profile Fit(s)')
        ax[0].set_ylabel('Flux')
        ax[0].legend()
        plt.show()

    def fit_voigt(self, x=None, y=None, yerr=None, p0=None, **kwargs):
        """Fits a Voigt profile to the given data.

        Parameters
        ----------
        x : array_like, optional
            The x values of the data. If None, uses self.velocities.
        y : array_like, optional
            The y values of the data. If None, uses self.flux.
        yerr : array_like, optional
            The y errors of the data. If None, uses self.flux_err.
        p0 : list, optional
            Initial guess for the parameters [amplitude, centre, sigma, gamma], by default None.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        x    = np.copy(self.velocities) if x    is None else x
        y    = np.copy(self.flux)       if y    is None else y
        yerr = np.copy(self.flux_err)   if yerr is None else yerr

        if p0 is None:
            amplitude_guess = np.min(y)
            centre_guess = x[np.argmin(y)]
            sigma0 = (x.max() - x.min()) / 10.0
            gamma0 = sigma0
            p0 = [amplitude_guess, centre_guess, sigma0, gamma0]

        popt, pcov = curve_fit(self.voigt_func, x, y, sigma=yerr, p0=p0, **kwargs)

        self.fitted_voigt = self.voigt_func(self.fitted_x, *popt)
        self.voigt_on_x = self.voigt_func(x, *popt)
        return popt, pcov

    def fit_gaussian(self, x=None, y=None, yerr=None, p0=None, **kwargs):
        """Fits a Gaussian profile to the given data.

        Parameters
        ----------
        x : array_like, optional
            The x values of the data. If None, uses self.velocities.
        y : array_like, optional
            The y values of the data. If None, uses self.flux.
        yerr : array_like, optional
            The y errors of the data. If None, uses self.flux_err.
        p0 : list, optional
            Initial guess for the parameters [amplitude, mean, stddev], by default None.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        x    = np.copy(self.velocities) if x    is None else x
        y    = np.copy(self.flux)       if y    is None else y
        yerr = np.copy(self.flux_err)   if yerr is None else yerr

        if p0 is None:
            amplitude_guess = np.min(y)
            mean_guess = x[np.argmin(y)]
            stddev_guess = (x.max() - x.min()) / 10.0
            p0 = [amplitude_guess, mean_guess, stddev_guess]

        popt, pcov = curve_fit(self.gaussian_func, x, y, sigma=yerr, p0=p0, **kwargs)

        self.fitted_gaussian = self.gaussian_func(self.fitted_x, *popt)
        self.gaussian_on_x = self.gaussian_func(x, *popt)
        return popt, pcov

    def fit_lorentzian(self, x=None, y=None, yerr=None, p0=None, **kwargs):
        """Fits a Lorentzian profile to the given data.

        Parameters
        ----------
        x : array_like, optional
            The x values of the data. If None, uses self.velocities.
        y : array_like, optional
            The y values of the data. If None, uses self.flux.
        yerr : array_like, optional
            The y errors of the data. If None, uses self.flux_err.
        p0 : list, optional
            Initial guess for the parameters [amplitude, centre, gamma], by default None.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        x    = np.copy(self.velocities) if x    is None else x
        y    = np.copy(self.flux)       if y    is None else y
        yerr = np.copy(self.flux_err)   if yerr is None else yerr

        if p0 is None:
            amplitude_guess = np.min(y)
            centre_guess = x[np.argmin(y)]
            gamma0 = (x.max() - x.min()) / 10.0
            p0 = [amplitude_guess, centre_guess, gamma0]

        popt, pcov = curve_fit(self.lorentzian_func, x, y, sigma=yerr, p0=p0, **kwargs)

        self.fitted_lorentzian = self.lorentzian_func(self.fitted_x, *popt)
        self.lorentzian_on_x = self.lorentzian_func(x, *popt)
        return popt, pcov

    @staticmethod
    def voigt_func(x, amplitude, centre, sigma, gamma):
        """Calculates the Voigt profile at given x values.

        Parameters
        ----------
        x : array_like
            The x values where the Voigt profile is evaluated.
        amplitude : float
            The amplitude of the Voigt profile.
        centre : float
            The center position of the Voigt profile.
        sigma : float
            The Gaussian standard deviation.
        gamma : float
            The Lorentzian half-width at half-maximum.

        Returns
        -------
        array_like
            The Voigt profile evaluated at the input x values.
        """
        from scipy.special import wofz

        z = ((x - centre) + 1j*gamma) / (sigma * np.sqrt(2))
        voigt_profile = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
        return voigt_profile

    @staticmethod
    def gaussian_func(x, amplitude, mean, stddev):
        """Calculates the Gaussian profile at given x values.

        Parameters
        ----------
        x : array_like
            The x values where the Gaussian profile is evaluated.
        amplitude : float
            The amplitude of the Gaussian profile.
        mean : float
            The mean (center) of the Gaussian profile.
        stddev : float
            The standard deviation of the Gaussian profile.

        Returns
        -------
        array_like
            The Gaussian profile evaluated at the input x values.
        """
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
    
    @staticmethod
    def lorentzian_func(x, amplitude, centre, gamma):
        """Calculates the Lorentzian profile at given x values.

        Parameters
        ----------
        x : array_like
            The x values where the Lorentzian profile is evaluated.
        amplitude : float
            The amplitude of the Lorentzian profile.
        centre : float
            The center position of the Lorentzian profile.
        gamma : float
            The half-width at half-maximum.

        Returns
        -------
        array_like
            The Lorentzian profile evaluated at the input x values.
        """
        return (amplitude * (gamma**2)) / ((x - centre)**2 + gamma**2)