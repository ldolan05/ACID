from __future__ import annotations
from .diagnostics.errors import *
import numpy as np
import copy, warnings, functools
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from beartype import beartype
from scipy.special import wofz
from .utils import Array1D, Array2D, Array3D, profile_name_map
from .data import Data
from .diagnostics.warnings import ACIDRuntimeWarning

def _fit_profiles(method):
    """Apply a single-profile fitter independently to each row of a 2D flux array."""
    @functools.wraps(method)
    def wrapper(self, x=None, y=None, yerr=None, cov_matrix=None, p0=None, on_exception="raise", **kwargs):
        # Keep single-profile calls on the original fitting path.
        if np.ndim(self.flux if y is None else y) != 2:
            return method(self, x, y, yerr, cov_matrix, p0, on_exception, **kwargs)

        # Resolve defaults and validate the shared grid and profile uncertainties.
        inputs = Profiles(*self._copy_inputs(x, y, yerr, cov_matrix))
        model = method.__name__.removeprefix("fit_")
        results, fitted, errors, on_x = [], [], [], []
        for i, flux in enumerate(inputs.flux):
            # Isolate this row's data and fit caches before calling the fitter again.
            profile = copy.copy(inputs)
            profile.flux = flux
            profile.flux_err = None if inputs.flux_err is None else inputs.flux_err[i]
            profile.cov_matrix = inputs._profile_covariance(i)
            profile.fitted_y, profile.fitted_yerr, profile.fit_on_x = {}, {}, {}
            profile._rng = copy.deepcopy(self._rng)
            profile.fitted_x = self.fitted_x

            # Fit with a shared or per-profile initial guess and failure policy.
            guess = p0[i] if p0 is not None and np.ndim(p0) == 2 else p0
            popt, pcov = wrapper(profile, p0=guess, on_exception=on_exception, **kwargs)
            results.append((popt, pcov))

            # Preserve row positions, including failed fits, on the shared plotting grids.
            failed = np.isnan(popt).any()
            fitted.append(np.full_like(self.fitted_x, np.nan) if failed else profile.fitted_y[model])
            errors.append(np.full((2, len(self.fitted_x)), np.nan) if failed else profile.fitted_yerr[model])
            on_x.append(getattr(self, f"{model}_func")(inputs.velocities, *popt))

        # Collect per-profile results and plot caches on the original instance.
        self.fitted_y[model] = np.array(fitted)
        self.fitted_yerr[model] = np.array(errors)
        self.fit_on_x[model] = np.array(on_x)
        return np.stack([r[0] for r in results]), np.stack([r[1] for r in results])
    return wrapper


@beartype
class Profiles:
    """
    A class for fitting spectral line profiles to Voigt, Gaussian, and Lorentzian models.

    With 2D flux, each row is fitted independently. Fitting methods return parameters
    of shape (n_profiles, n_parameters) and covariances of shape
    (n_profiles, n_parameters, n_parameters), with a shared or per-profile p0.
    """
    def __init__(
            self,
            velocities    : Array1D|Array2D|None = None,
            flux          : Array1D|Array2D|None = None,
            flux_err      : Array1D|Array2D|None = None,
            cov_matrix    : Array2D|Array3D|None = None,
            data          : Data|None            = None,
            key           : str                  = "final",
            profile_names : list|dict|None       = None,
        ) -> None:
        """
        Initializes the Profiles class with velocity, flux, and optional flux error data.

        Parameters
        ----------
        velocities : :py:type:`Array1D | Array2D`, optional
            The velocity values corresponding to the spectral line profile.
            A 2D input must repeat the same velocity grid for every profile.
            Must be provided if no data instance is passed, by default None.
        flux : :py:type:`Array1D | Array2D`, optional
            The flux values of the spectral line profile, normalized to a continuum at 1.
            For multiple profiles, use shape (n_profiles, n_velocities).
            Must be provided if no data instance is passed, by default None.
        flux_err : :py:type:`Array1D | Array2D`, optional
            The errors associated with the flux values, by default None. If not
            input, they won't be used in the fitting process. A 1D array is shared across profiles.
        cov_matrix : :py:type:`Array2D | Array3D`, optional
            The covariance matrix associated with the flux values, by default None. If not
            input, it won't be used in the fitting process. Inputting this overrides the errors when fitting.
            Accepts a shared matrix, one matrix per profile, or a flattened multi-profile
            matrix. Profiles are fitted independently using their diagonal covariance blocks.
        data : :py:class:`Data`, optional
            A data instance to draw velocities, flux, flux errors, and covariance matrix. Will raise an
            exception if they do not exist within the class.
            Must be provided if all four of the above inputs were not passed, by default None. 
        key : str, optional
            The key to select a specific profile from the data instance, by default "final".
        profile_names : list | dict, optional
            Names for profile rows, or actual group IDs when using Data. Defaults to
            Data.profile_names when available. Direct array inputs use row IDs 0, 1, ...
        """

        if data is not None:
            if data.velocities is None or key not in data.profile:
                raise ACIDStateError(f"Data instance must have attributes 'velocities' and 'profile[key]' (key={key}).\n"\
                                 f"Try running ACID first.")
            velocities = data.velocities
            flux = data.profile[key][0]
            flux_err = data.profile[key][1]
            cov_matrix = data.profile[key][2]
        else:
            if velocities is None or flux is None:
                raise ACIDInputError("If no data instance is provided, then at least velocities and flux must be provided.")

        self._rng = data.get_result_rng() if data is not None else np.random.default_rng()
        self.velocities = None
        self.flux = None
        self.flux_err = None
        self.cov_matrix = None

        # Copy inputs and reduce repeated velocity rows to a shared 1D grid.
        x, y, yerr, cov = self._copy_inputs(velocities, flux, flux_err, cov_matrix)
        if x.ndim == 2:
            if x.shape != y.shape or not np.array_equal(x, np.broadcast_to(x[0], x.shape), equal_nan=True):
                raise ACIDInputError("2D velocities must contain the same velocity grid for every profile.")
            x = x[0]
        if x.ndim != 1 or y.ndim not in (1, 2) or y.shape[-1] != len(x) or y.size == 0:
            raise ACIDInputError("Flux must be 1D or 2D with one value per velocity.")

        # Broadcast shared errors and validate the supported covariance layouts.
        if yerr is not None:
            if yerr.shape != y.shape and not (y.ndim == 2 and yerr.shape == x.shape):
                raise ACIDInputError("Flux errors must match the flux or the shared velocity grid.")
            yerr = np.broadcast_to(yerr, y.shape).copy()
        if cov is not None:
            n = len(x)
            shapes = [(n, n)]
            if y.ndim == 2:
                shapes += [(y.shape[0], n, n), (y.size, y.size)]
            if cov.shape not in shapes:
                raise ACIDInputError("Covariance must be shared, per-profile, or a flattened multi-profile matrix.")

        # Each profile can have different invalid pixels; clean rows separately when fitting.
        if y.ndim == 1:
            x, y, yerr, cov = self._remove_nans(x, y, yerr, cov)
        self.velocities, self.flux, self.flux_err, self.cov_matrix = x, y, yerr, cov

        # Resolve labels in the same sorted group order as the profile rows.
        n_profiles = y.shape[0] if y.ndim == 2 else 1
        self.profile_ids = np.arange(n_profiles)
        if data is not None:
            if y.ndim == 2 and data.profile_groups is not None:
                self.profile_ids = np.unique(data.profile_groups)
            if profile_names is None:
                profile_names = data.profile_names
                # A single-profile diagnostic does not correspond to individual groups.
                if y.ndim == 1 and data.profile_groups is not None:
                    profile_names = None
        self.profile_names = profile_name_map(profile_names, self.profile_ids) or {}

        # Cache fitted curves and uncertainties on a dense grid for plotting.
        self.fitted_y    = {}
        self.fitted_yerr = {}
        self.fit_on_x    = {} # Store fitted values on original x for residuals

        self.fitted_x = np.linspace(np.nanmin(self.velocities), np.nanmax(self.velocities), 1000)

    def _profile_covariance(self, i):
        """Select this profile's covariance, excluding cross-profile blocks."""
        if self.cov_matrix is None:
            return None
        if self.cov_matrix.ndim == 3:
            return self.cov_matrix[i]
        n = len(self.velocities)
        if self.cov_matrix.shape == (n, n):
            return self.cov_matrix
        # ACID's flattened covariance stores each profile in a contiguous block.
        block = slice(i * n, (i + 1) * n)
        return self.cov_matrix[block, block]

    def plot_fit(self, model:str|None='voigt', return_fig=False, **kwargs) -> tuple|None:
        """Plots the original data and the fitted profile if available.
        
        Parameters
        ----------
        model : str | None, optional
            The type of model to plot. String options are 'voigt', 'gaussian', 
            'lorentzian', 'none', or 'all'. Choosing 'none' or None will plot whichever models have 
            already been fitted for, by default 'voigt'. Multiple profiles can only
            be plotted with one model at a time.
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
                raise ACIDInputError("Model must be 'voigt', 'gaussian', 'lorentzian' or 'all'.")

        # Select requested models and restrict multi-profile plots to one model.
        if model == 'all':
            model_list = models[:-1] # Exclude 'none'
        elif model is None or model == "none":
            model_list = list(self.fitted_y.keys())
        else:
            model_list = [model]

        if self.flux.ndim == 2 and len(model_list) > 1:
            raise ACIDInputError("Plotting multiple profiles requires a single model: 'gaussian', 'voigt', or 'lorentzian'.")

        # Reuse cached fits, fitting only models that have not been requested before.
        for m in model_list:
            if m not in self.fitted_y:
                fit_func = getattr(self, f"fit_{m}")
                fit_func(**kwargs)

        # Plotting
        fig, ax = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
        ax[1].axhline(0, color='black', linestyle='--')

        # Give each profile a consistent colour across its data, fit and residuals.
        multiple = self.flux.ndim == 2
        for i, flux in enumerate(np.atleast_2d(self.flux)):
            error = None if self.flux_err is None else (self.flux_err[i] if multiple else self.flux_err)
            label = self.profile_names.get(self.profile_ids[i], f'Profile {i}' if multiple else 'ACID Profile')
            ax[0].errorbar(self.velocities, flux, yerr=error, fmt='.', label=label, color=f'C{i}')
            for j, name in enumerate(model_list):
                y_fit = self.fitted_y[name][i] if multiple else self.fitted_y[name]
                on_x = self.fit_on_x[name][i] if multiple else self.fit_on_x[name]
                color = f'C{i}' if multiple else f'C{j+1}'
                prefix = f'{label} {name.capitalize()}' if multiple else name.capitalize()
                ax[0].plot(self.fitted_x, y_fit, label=f'{prefix} Fit', color=color)
                ax[1].errorbar(self.velocities, on_x - flux, yerr=error, fmt='.', label=f'{prefix} Residuals', color=color)
        ax[1].set_xlabel('Velocity')
        ax[0].set_title('Profile Fit(s)')
        ax[1].set_ylabel('Flux')
        ax[0].set_ylabel('Flux')
        ax[0].legend()
        ax[1].legend()
        if return_fig:
            return fig, ax
        plt.show()

    @_fit_profiles
    def fit_voigt(self, x=None, y=None, yerr=None, cov_matrix=None, p0=None, on_exception:str="raise", **kwargs) -> tuple:
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
        on_exception : str, optional
            Determines the behavior when an exception occurs during fitting.
            Options are "raise" (default) to raise the exception, "warn" to issue a warning, or "pass" to silently continue.
            If "pass" or "warn", returns NaN-filled parameter (5,) and covariance (5, 5) arrays on failure.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
            If the fit fails and on_exception is "pass" or "warn", returns NaN-filled arrays with the same shapes.
        """
        if on_exception not in ["raise", "warn", "pass"]:
            raise ValueError(f"Invalid value for on_exception: {on_exception}. Must be 'raise', 'warn', or 'pass'.")

        try:
            x, y, yerr, cov_matrix = self._remove_nans(*self._copy_inputs(x, y, yerr, cov_matrix))

            bounds = (
                [-1.5, np.min(x), 0, 0, -np.inf], # Lower bounds
                [0, np.max(x), np.inf, np.inf, np.inf] # Upper
            )

            if p0 is None:
                amplitude_guess = np.nanmin(y) - 1
                centre_guess = x[np.nanargmin(y)]

                half = 0.5 * (np.nanmax(y) + np.nanmin(y))
                above = np.flatnonzero(y < half)
                if above.size >= 2:
                    fwhm = abs(x[above[-1]] - x[above[0]])
                else:
                    raise ACIDResultError("Could not estimate FWHM for Voigt profile. Not enough points above half maximum.")
                sigma0 = fwhm / 2.355

                gamma0 = sigma0 * 0.1
                offset = 0
                p0 = [amplitude_guess, centre_guess, sigma0, gamma0, offset]

                # Raise an error if the initial guess is outside the bounds
                for i in range(len(p0)):
                    if not (bounds[0][i] <= p0[i] <= bounds[1][i]):
                        raise ACIDResultError(f"Initial guess for parameter {i} is outside the bounds.\n"
                                        f"Initial guess: {p0[i]}, Bounds: {bounds[0][i]} to {bounds[1][i]}")

            popt, pcov = self._fit_model("voigt", x, y, yerr, cov_matrix, p0, bounds=bounds, **kwargs)
            return popt, pcov
        
        except Exception as e:
            if on_exception == "raise":
                raise
            elif on_exception == "warn":
                warnings.warn(f"Voigt fit failed: {e}", ACIDRuntimeWarning, stacklevel=2)
            return np.full(5, np.nan), np.full((5, 5), np.nan)

    @_fit_profiles
    def fit_gaussian(self, x=None, y=None, yerr=None, cov_matrix=None, p0=None, on_exception:str="raise", **kwargs) -> tuple:
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
            Initial guess for the parameters [amplitude, mean, stddev, offset], by default None.
        on_exception : str, optional
            Determines behavior when an exception occurs. Options are "raise" (default), "warn", or "pass".
            If "pass" or "warn", returns NaN-filled parameter (4,) and covariance (4, 4) arrays on failure.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.
        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        if on_exception not in ["raise", "warn", "pass"]:
            raise ValueError(f"Invalid value for on_exception: {on_exception}. Must be 'raise', 'warn', or 'pass'.")

        try:
            x, y, yerr, cov_matrix = self._remove_nans(*self._copy_inputs(x, y, yerr, cov_matrix))

            bounds = (
                [-1.5, np.min(x), 0, -np.inf], # Lower bounds
                [0, np.max(x), np.inf, np.inf] # Upper bounds
            )

            if p0 is None:
                amplitude_guess = np.nanmin(y) - 1
                mean_guess = x[np.nanargmin(y)]
                half = 0.5 * (np.nanmax(y) + np.nanmin(y))
                above = np.flatnonzero(y < half)
                if above.size >= 2:
                    fwhm = abs(x[above[-1]] - x[above[0]])
                else:
                    raise ACIDResultError("Could not estimate FWHM for Gaussian profile. Not enough points above half maximum.")
                sigma0 = fwhm / 2.355
                offset = 0
                p0 = [amplitude_guess, mean_guess, sigma0, offset]
                
                for i in range(len(p0)):
                    if not (bounds[0][i] <= p0[i] <= bounds[1][i]):
                        raise ACIDResultError(f"Initial guess for parameter {i} is outside the bounds.\n"
                                         f"Initial guess: {p0[i]}, Bounds: {bounds[0][i]} to {bounds[1][i]}")

            popt, pcov = self._fit_model("gaussian", x, y, yerr, cov_matrix, p0, bounds=bounds, **kwargs)
            return popt, pcov

        except Exception as e:
            if on_exception == "raise":
                raise
            elif on_exception == "warn":
                warnings.warn(f"Gaussian fit failed: {e}", ACIDRuntimeWarning, stacklevel=2)
            return np.full(4, np.nan), np.full((4, 4), np.nan)

    @_fit_profiles
    def fit_lorentzian(self, x=None, y=None, yerr=None, cov_matrix=None, p0=None, on_exception:str="raise", **kwargs) -> tuple:
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
        on_exception : str, optional
            Determines behavior when an exception occurs. Options are "raise" (default), "warn", or "pass".
            If "pass" or "warn", returns NaN-filled parameter (4,) and covariance (4, 4) arrays on failure.
        **kwargs : dict
            Additional keyword arguments to pass to curve_fit.

        Returns
        -------
        tuple
            A tuple containing the optimal parameters and the covariance matrix.
        """
        if on_exception not in ["raise", "warn", "pass"]:
            raise ValueError(f"Invalid value for on_exception: {on_exception}. Must be 'raise', 'warn', or 'pass'.")

        try:
            x, y, yerr, cov_matrix = self._remove_nans(*self._copy_inputs(x, y, yerr, cov_matrix))

            bounds = (
                [-1, np.min(x), 0, -np.inf], # Lower bounds
                [0, np.max(x), np.inf, np.inf] # Upper bounds
            )

            if p0 is None:
                amplitude_guess = np.nanmin(y) - 1
                centre_guess = x[np.nanargmin(y)]
                half = 0.5 * (np.nanmax(y) + np.nanmin(y))
                above = np.flatnonzero(y < half)
                if above.size >= 2:
                    fwhm = abs(x[above[-1]] - x[above[0]])
                else:
                    raise ACIDResultError("Could not estimate FWHM for Voigt profile. Not enough points above half maximum.")
                sigma0 = fwhm / 2.355
                gamma0 = sigma0 * 0.1
                offset = 0
                p0 = [amplitude_guess, centre_guess, gamma0, offset]

                for i in range(len(p0)):
                    if not (bounds[0][i] <= p0[i] <= bounds[1][i]):
                        raise ACIDResultError(f"Initial guess for parameter {i} is outside the bounds.\n"
                                            f"Initial guess: {p0[i]}, Bounds: {bounds[0][i]} to {bounds[1][i]}")

            popt, pcov = self._fit_model("lorentzian", x, y, yerr, cov_matrix, p0, bounds=bounds, **kwargs)
            return popt, pcov

        except Exception as e:
            if on_exception == "raise":
                raise
            elif on_exception == "warn":
                warnings.warn(f"Lorentzian fit failed: {e}", ACIDRuntimeWarning, stacklevel=2)
            return np.full(4, np.nan), np.full((4, 4), np.nan)

    @staticmethod
    def _remove_nans(x, y, yerr, cov_matrix):
        mask = np.isfinite(x) & np.isfinite(y)

        if yerr is not None:
            mask &= np.isfinite(yerr) & (yerr > 0)

        x = x[mask]
        y = y[mask]

        if yerr is not None:
            yerr = yerr[mask]

        if cov_matrix is not None:
            idx = np.where(mask)[0]
            cov_matrix = cov_matrix[np.ix_(idx, idx)]

        return x, y, yerr, cov_matrix

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
        x    = np.copy(self.velocities) if x    is None else np.array(x)
        y    = np.copy(self.flux)       if y    is None else np.array(y)
        yerr = self.flux_err            if yerr is None else np.array(yerr)
        yerr_copy = np.copy(yerr) if yerr is not None else yerr
        cov_matrix = self.cov_matrix if cov_matrix is None else np.array(cov_matrix)
        cov_matrix_copy = np.copy(cov_matrix) if cov_matrix is not None else cov_matrix
        return x, y, yerr_copy, cov_matrix_copy

    def _fit_model(self, model_name, x, y, yerr, cov_matrix, p0, bounds=None, **kwargs) -> tuple:
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
        bounds : tuple, optional
            Bounds for the parameters in the form (lower_bounds, upper_bounds), by default None.
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
        popt, pcov = curve_fit(model_func, x, y, sigma=sigma, p0=p0, bounds=bounds, **kwargs)
        self.fitted_y[model_name] = model_func(self.fitted_x, *popt)
        self.fit_on_x[model_name] = model_func(x, *popt)

        # Get errors
        samples = copy.deepcopy(self._rng).multivariate_normal(mean=popt, cov=pcov, size=1000)
        samples = samples[np.all((samples >= bounds[0]) & (samples <= bounds[1]), axis=1)] if bounds is not None else samples
        if len(samples) == 0:
            raise ACIDResultError("No uncertainty samples fall within the fit bounds; profile uncertainties cannot be estimated.")
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
        z = ((x - centre) + 1j * gamma) / (sigma * np.sqrt(2))
        profile = np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

        z0 = 1j * gamma / (sigma * np.sqrt(2))
        peak = np.real(wofz(z0)) / (sigma * np.sqrt(2*np.pi))

        profile = profile / peak
        return 1 + amplitude * profile + offset

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
        return 1 + amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2)) + offset
    
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
        return 1 + (amplitude * (gamma**2)) / ((x - centre)**2 + gamma**2) + offset
