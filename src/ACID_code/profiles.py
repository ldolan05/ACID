from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from beartype import beartype
from scipy.special import wofz
from .utils import Array1D, Array2D
from .data import Data

@beartype
class Profiles:
    """
    A class for fitting spectral line profiles to Voigt, Gaussian, and Lorentzian models.
    """
    def __init__(
            self,
            velocities : Array1D = None,
            flux       : Array1D = None,
            flux_err   : Array1D = None,
            cov_matrix : Array2D = None,
            data       : Data    = None,
        ) -> None:
        """
        Initializes the Profiles class with velocity, flux, and optional flux error data.

        Parameters
        ----------
        velocities : :py:type:`Array1D`, optional
            The velocity values corresponding to the spectral line profile.
            Must be provided if no data instance is passed, by default None.
        flux : :py:type:`Array1D`, optional
            The flux values of the spectral line profile
            Must be provided if no data instance is passed, by default None.
        flux_err : :py:type:`Array1D`, optional
            The errors associated with the flux values, by default None. If not
            input, they won't be used in the fitting process.
        cov_matrix : :py:type:`Array2D`, optional
            The covariance matrix associated with the flux values, by default None. If not
            input, it won't be used in the fitting process. Inputting this overrides the errors when fitting.
        data : :py:class:`Data`, optional
            A data instance to draw velocities, flux, flux errors, and covariance matrix. Will raise an
            exception if they do not exist within the class.
            Must be provided if all four of the above inputs were not passed, by default None. 
        """

        if data is not None:
            if not getattr(data, 'velocities', None) or not getattr(data, 'profiles', None):
                raise ValueError("Data instance must have attributes 'velocities' and 'profiles'. Try running ACID first.")
            velocities = data.velocities
            flux = data.profiles[0,0]
            flux_err = data.profiles[0,1]
            cov_matrix = data.profiles[0,2]
        else:
            if velocities is None or flux is None:
                raise ValueError("If no data instance is provided, then at least velocities and flux must be provided.")

        self.velocities = velocities
        self.flux = flux-1 # Subtract 1 to convert from normalized flux to absorption depth
        self.flux_err = flux_err
        self.cov_matrix = cov_matrix

        self.fitted_y    = {}
        self.fitted_yerr = {}
        self.fit_on_x    = {} # Store fitted values on original x for residuals

        self.fitted_x = np.linspace(np.min(velocities), np.max(velocities), 1000)
        pass

    def plot_fit(self, model:str|None='voigt', return_fig=False, **kwargs) -> tuple|None:
        """Plots the original data and the fitted profile if available.
        
        Parameters
        ----------
        model : str | None, optional
            The type of model to plot. String options are 'voigt', 'gaussian', 
            'lorentzian', 'none', or 'all'. Choosing 'none' or None will plot whichever models have 
            already been fitted for, by default 'all'.
        return_fig : bool, optional
            Whether to return the (fig, ax) tuple and not call plt.show(). If False, calls
            plt.show() and returns None. By default False.
        **kwargs : dict
            Additional keyword arguments to pass to the fitting functions if the models
            have not been fitted yet.
        Returns
        -------
        tuple | None
            If return_fig is True, returns the (fig, ax) tuple. Otherwise, returns None.
        """
        models = ["voigt", "gaussian", "lorentzian", "none"]
        if model is not None:
            model = model.lower()
            if model not in models + ['all']:
                raise ValueError("Model must be 'voigt', 'gaussian', 'lorentzian' or 'all'.")

        if model == 'all':
            model_list = models[:-1] # Exclude 'none'
        elif model is None or model == "none":
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
            ax[1].errorbar(self.velocities, y_fit_on_x - self.flux, yerr=self.flux_err, fmt='b.', label=f'{model.capitalize()} Residuals', color=f'C{i+1}')
        ax[1].set_xlabel('Velocity')
        ax[0].set_title('Profile Fit(s)')
        ax[1].set_ylabel('Flux')
        ax[0].set_ylabel('Flux')
        ax[0].legend()
        ax[1].legend()
        if return_fig:
            return fig, ax
        plt.show()

    def fit_voigt(self, x=None, y=None, yerr=None, cov_matrix=None, p0=None, **kwargs) -> tuple:
        """Fits a Voigt profile to the given data.

        Parameters
        ----------
        x : array_like, optional
            The x values of the data. If None, uses self.velocities.
        y : array_like, optional
            The y values of the data. If None, uses self.flux.
        yerr : array_like, optional
            The y errors of the data. If None, uses self.flux_err.
        cov_matrix : np.ndarray, optional
            The covariance matrix associated with the flux values. If None, uses self.cov_matrix.
        p0 : list, optional
            Initial guess for the parameters [amplitude, centre, sigma, gamma, offset], by default None.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        x, y, yerr, cov_matrix = self._copy_inputs(x, y, yerr, cov_matrix)

        if p0 is None:
            amplitude_guess = np.min(y)
            centre_guess = x[np.argmin(y)]
            sigma0 = (x.max() - x.min()) / 10.0
            gamma0 = sigma0
            offset = 0
            p0 = [amplitude_guess, centre_guess, sigma0, gamma0, offset]

        popt, pcov = self._fit_model("voigt", x, y, yerr, cov_matrix, p0, **kwargs)
        return popt, pcov

    def fit_gaussian(self, x=None, y=None, yerr=None, cov_matrix=None, p0=None, **kwargs) -> tuple:
        """Fits a Gaussian profile to the given data.

        Parameters
        ----------
        x : array_like, optional
            The x values of the data. If None, uses self.velocities.
        y : array_like, optional
            The y values of the data. If None, uses self.flux.
        yerr : array_like, optional
            The y errors of the data. If None, uses self.flux_err.
        cov_matrix : np.ndarray, optional
            The covariance matrix associated with the flux values. If None, uses self.cov_matrix.
        p0 : list, optional
            Initial guess for the parameters [amplitude, mean, stddev], by default None.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        x, y, yerr, cov_matrix = self._copy_inputs(x, y, yerr, cov_matrix)

        if p0 is None:
            amplitude_guess = np.min(y)
            mean_guess = x[np.argmin(y)]
            stddev_guess = (x.max() - x.min()) / 10.0
            offset = 0
            p0 = [amplitude_guess, mean_guess, stddev_guess, offset]

        popt, pcov = self._fit_model("gaussian", x, y, yerr, cov_matrix, p0, **kwargs)
        return popt, pcov

    def fit_lorentzian(self, x=None, y=None, yerr=None, cov_matrix=None, p0=None, **kwargs) -> tuple:
        """Fits a Lorentzian profile to the given data.

        Parameters
        ----------
        x : array_like, optional
            The x values of the data. If None, uses self.velocities.
        y : array_like, optional
            The y values of the data. If None, uses self.flux.
        yerr : array_like, optional
            The y errors of the data. If None, uses self.flux_err.
        cov_matrix : np.ndarray, optional
            The covariance matrix associated with the flux values. If None, uses self.cov_matrix.
        p0 : list, optional
            Initial guess for the parameters [amplitude, centre, gamma, offset], by default None.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        x, y, yerr, cov_matrix = self._copy_inputs(x, y, yerr, cov_matrix)

        if p0 is None:
            amplitude_guess = np.min(y)
            centre_guess = x[np.argmin(y)]
            gamma0 = (x.max() - x.min()) / 10.0
            offset = 0
            p0 = [amplitude_guess, centre_guess, gamma0, offset]

        popt, pcov = self._fit_model("lorentzian", x, y, yerr, cov_matrix, p0, **kwargs)
        return popt, pcov

    def _copy_inputs(self, x, y, yerr, cov_matrix) -> tuple:
        """Internal method to copy input data or use class attributes.

        Parameters
        ----------
        x : array_like | None
            The x values of the data. If None, uses self.velocities.
        y : array_like | None
            The y values of the data. If None, uses self.flux.
        yerr : array_like | None
            The y errors of the data. If None, uses self.flux_err.
        cov_matrix : np.ndarray | None
            The covariance matrix associated with the flux values. If None, uses self.cov_matrix.
            
        Returns
        -------
        tuple
            A tuple containing the copied x, y, yerr arrays, and the covariance matrix.
        """
        x    = np.copy(self.velocities) if x    is None else x
        y    = np.copy(self.flux)       if y    is None else y
        yerr = self.flux_err            if yerr is None else yerr
        yerr_copy = np.copy(yerr) if yerr is not None else yerr
        cov_matrix = self.cov_matrix if cov_matrix is None else cov_matrix
        cov_matrix_copy = np.copy(cov_matrix) if cov_matrix is not None else cov_matrix
        return x, y, yerr_copy, cov_matrix_copy

    def _fit_model(self, model_name, x, y, yerr, cov_matrix, p0, **kwargs) -> tuple:
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
        cov_matrix : np.ndarray
            The covariance matrix associated with the flux values.
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
        sigma = yerr if cov_matrix is None else cov_matrix
        popt, pcov = curve_fit(model_func, x, y, sigma=sigma, p0=p0, absolute_sigma=True, **kwargs)
        self.fitted_y[model_name] = model_func(self.fitted_x, *popt)
        self.fit_on_x[model_name] = model_func(x, *popt)

        # Get errors
        samples = np.random.multivariate_normal(mean=popt, cov=pcov, size=1000)
        y_samples = np.array([model_func(self.fitted_x, *sample) for sample in samples])
        y_lo, y_med, y_hi = np.quantile(y_samples, [0.16, 0.50, 0.84], axis=0)
        self.fitted_yerr[model_name] = (y_med - y_lo, y_hi - y_med)

        return popt, pcov

    @staticmethod
    def voigt_func(x, amplitude, centre, sigma, gamma, offset=0) -> Array1D:
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
        offset : float, optional
            The continuum offset, by default 0.

        Returns
        -------
        array_like
            The Voigt profile evaluated at the input x values.
        """
        z = ((x - centre) + 1j*gamma) / (sigma * np.sqrt(2))
        voigt_profile = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
        return voigt_profile + offset

    @staticmethod
    def gaussian_func(x, amplitude, mean, stddev, offset=0) -> Array1D:
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
        offset : float, optional
            The continuum offset, by default 0.

        Returns
        -------
        array_like
            The Gaussian profile evaluated at the input x values.
        """
        return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset
    
    @staticmethod
    def lorentzian_func(x, amplitude, centre, gamma, offset=0) -> Array1D:
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
        offset : float, optional
            The continuum offset, by default 0.

        Returns
        -------
        array_like
            The Lorentzian profile evaluated at the input x values.
        """
        return (amplitude * (gamma**2)) / ((x - centre)**2 + gamma**2) + offset