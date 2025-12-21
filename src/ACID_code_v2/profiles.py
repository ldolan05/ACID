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

        self.fitted_y    = {}
        self.fitted_yerr = {}
        self.fit_on_x    = {} # Store fitted values on original x for residuals

        self.fitted_x = np.linspace(np.min(velocities), np.max(velocities), 1000)
        pass

    def plot_fit(self, model:str|None='all', **kwargs):
        """Plots the original data and the fitted profile if available.
        
        Parameters
        ----------
        model : str | None, optional
            The type of model to plot. String options are 'voigt', 'gaussian', 
            'lorentzian', or 'all'. Choosing None will plot whichever models have 
            already been fitted for, by default 'all'.
        **kwargs : dict
            Additional keyword arguments to pass to the fitting functions if the models
            have not been fitted yet.
        Returns
        -------
        None
        """
        models = ["voigt", "gaussian", "lorentzian"]
        if model is not None:
            model = model.lower()
            if model not in models + ['all']:
                raise ValueError("Model must be 'voigt', 'gaussian', 'lorentzian' or 'all'.")
        
        if model == 'all':
            model_list = models
        elif model is None:
            model_list = list(self.fitted_y.keys())
        else:
            model_list = [model]
        
        for m in model_list:
            if m not in self.fitted_y:
                fit_func = getattr(self, f"fit_{m}")
                fit_func(**kwargs)

        # Plotting
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax[0].errorbar(self.velocities, self.flux, yerr=self.flux_err, fmt='b.', label='ACID Profile', color='C0')
        ax[1].axhline(0, color='black', linestyle='--')

        for i, model in enumerate(self.fitted_y.keys()):
            y_fit = self.fitted_y[model]
            y_fit_on_x = self.fit_on_x[model]
            y_err_lo, y_err_hi = self.fitted_yerr[model]
            ax[0].plot(self.fitted_x, y_fit, label=f'{model.capitalize()} Fit', color=f'C{i+1}')
            ax[0].fill_between(self.fitted_x, y_fit - y_err_lo, y_fit + y_err_hi, color=f'C{i+1}', alpha=0.3)
            ax[1].plot(self.velocities, y_fit_on_x - self.flux, label=f'{model.capitalize()} Residuals', color=f'C{i+1}')
        ax[1].set_xlabel('Velocity')
        ax[0].set_title('Profile Fit(s)')
        ax[1].set_ylabel('Flux')
        ax[0].set_ylabel('Flux')
        ax[0].legend()
        ax[1].legend()
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
        x, y, yerr = self._copy_inputs(x, y, yerr)

        if p0 is None:
            amplitude_guess = np.min(y)
            centre_guess = x[np.argmin(y)]
            sigma0 = (x.max() - x.min()) / 10.0
            gamma0 = sigma0
            p0 = [amplitude_guess, centre_guess, sigma0, gamma0]

        popt, pcov = self._fit_model("voigt", x, y, yerr, p0, **kwargs)
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
        x, y, yerr = self._copy_inputs(x, y, yerr)

        if p0 is None:
            amplitude_guess = np.min(y)
            mean_guess = x[np.argmin(y)]
            stddev_guess = (x.max() - x.min()) / 10.0
            p0 = [amplitude_guess, mean_guess, stddev_guess]

        popt, pcov = self._fit_model("gaussian", x, y, yerr, p0, **kwargs)
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
        x, y, yerr = self._copy_inputs(x, y, yerr)

        if p0 is None:
            amplitude_guess = np.min(y)
            centre_guess = x[np.argmin(y)]
            gamma0 = (x.max() - x.min()) / 10.0
            p0 = [amplitude_guess, centre_guess, gamma0]

        popt, pcov = self._fit_model("lorentzian", x, y, yerr, p0, **kwargs)
        return popt, pcov

    def _copy_inputs(self, x, y, yerr):
        """Internal method to copy input data or use class attributes.

        Parameters
        ----------
        x : array_like | None
            The x values of the data. If None, uses self.velocities.
        y : array_like | None
            The y values of the data. If None, uses self.flux.
        yerr : array_like | None
            The y errors of the data. If None, uses self.flux_err.

        Returns
        -------
        tuple
            A tuple containing the copied x, y, and yerr arrays.
        """
        x    = np.copy(self.velocities) if x    is None else x
        y    = np.copy(self.flux)       if y    is None else y
        yerr = self.flux_err   if yerr is None else yerr
        return x, y, yerr

    def _fit_model(self, model_name, x, y, yerr, p0, **kwargs):
        """Internal method to fit a specified model to the data.

        Parameters
        ----------
        model_name : str
            The name of the model to fit ('voigt', 'gaussian', 'lorentzian').
        x : array_like
            The x values of the data.
        y : array_like
            The y values of the data.
        yerr : array_like
            The y errors of the data.
        p0 : list
            Initial guess for the parameters.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.
        
        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        # Get the model function
        model_func = getattr(self, f"{model_name}_func")

        # Perform the curve fitting
        popt, pcov = curve_fit(model_func, x, y, sigma=yerr, p0=p0, **kwargs)
        self.fitted_y[model_name] = model_func(self.fitted_x, *popt)
        self.fit_on_x[model_name] = model_func(x, *popt)

        # Get errors
        samples = np.random.multivariate_normal(mean=popt, cov=pcov, size=1000)
        y_samples = np.array([model_func(self.fitted_x, *sample) for sample in samples])
        y_lo, y_med, y_hi = np.quantile(y_samples, [0.16, 0.50, 0.84], axis=0)
        self.fitted_yerr[model_name] = (y_med - y_lo, y_hi - y_med)

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