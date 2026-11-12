from __future__ import annotations
import warnings
from ..diagnostics.warnings import ACIDInputWarning
import numpy as np
import scipy.linalg
from ..utils import Array1D, Array2D
from beartype import beartype
from ..data import Config, Data
from ..lsd import LSD

class MCMC:

    """
    Shared probability model and priors for NumPy/SciPy and JAX sampling.
    Invariant terms are precomputed, and only initialization uses beartype checks.
    """

    @beartype
    def __init__(
        self,
        x_or_data             : Array1D|Array2D|Data,
        y                     : Array1D|None         = None,
        yerr                  : Array1D|None         = None,
        alpha                 : Array2D|None         = None,
        velocities            : Array1D|None         = None,
        c_factor                                     = None,
        deterministic_profile : bool|None            = None,
        sampler_type          : str|None             = None,
        od                    : bool|None            = None,
        continuum_method      : str|None             = None,
        use_jax               : bool|None            = None,
        numpy_module                                 = np,
        linalg                                       = scipy.linalg,
        ) -> None:
        """
        Initialise MCMC functions with necessary data.
        Called once per worker if using multiprocessing.
        Model settings default to None because they draw from the :py:class:`~ACID_code.Config` defaults.
        Passing a parameter in the signature overides any defaults from the :py:class:`~ACID_code.Config` instance.
        The parameter listings below show the defaults of the config for convenience.
        Otherwise obtain them with the :py:meth:`~ACID_code.Config.print_defaults` method.

        Parameters
        ----------
        x_or_data : :py:type:`~ACID_code.Array1D` | :py:type:`~ACID_code.Array2D` | :py:class:`~ACID_code.Data`
            Normalized wavelength array or :py:class:`~ACID_code.Data` instance. If a :py:class:`~ACID_code.Data` instance is provided, takes all
            the arguments below from there. If a :py:class:`~ACID_code.Data` instance is provided, all other
            arguments are ignored.
        y : :py:type:`~ACID_code.Array1D`, optional
            Observed flux array, required if x_or_data is a :py:type:`~ACID_code.Array1D` or :py:type:`~ACID_code.Array2D`.
        yerr : :py:type:`~ACID_code.Array1D`, optional
            Observed flux error array, required if x_or_data is a :py:type:`~ACID_code.Array1D` or :py:type:`~ACID_code.Array2D`.
        alpha : :py:type:`~ACID_code.Array2D`, optional
            Precomputed alpha matrix, required if x_or_data is a :py:type:`~ACID_code.Array1D` or :py:type:`~ACID_code.Array2D`.
        velocities : :py:type:`~ACID_code.Array1D` | None, optional
            Velocity grid for LSD profile, only needed when calling log-probability
            function, by default None.
        c_factor : tuple, optional
            Precomputed c_factor for LSD profile calculation, by default None.
        deterministic_profile : bool, optional
            Infer profile points from the continuum (True), or fit them explicitly (False), by default True.
        sampler_type : str, optional
            Type of sampler to use, by default "emcee".
        od : bool, optional
            Whether to use optical depth (True) or flux (False) for the profile points, by default True.
        continuum_method : str, optional
            Method to use for continuum fitting, by default "polyval".
            Options are "polyval" for standard polynomial evaluation or "chebval" for Chebyshev polynomial evaluation.
        use_jax : bool, optional
            Use an optional JAX-jitted log-probability kernel. JAX is imported
            lazily, and NumPy/SciPy is used if it is unavailable, by default False.
        numpy_module, linalg : module, optional
            Array operations and Cholesky solver, defaulting to NumPy and SciPy.
            When use_jax is enabled, these are replaced with the JAX equivalents.
        """

        if isinstance(x_or_data, Data):
            data = x_or_data
        else:
            # Validate raw inputs through the same Data/Config interface.
            data = Data(norm_wavelengths={"mcmc": np.asarray(x_or_data)}, flux={"mcmc": y},
                        errors={"mcmc": yerr}, alpha={"mcmc": alpha},
                        c_factor={"mcmc": c_factor} if c_factor is not None else {})
            if velocities is not None:
                data.velocities = velocities
            data.config = Config(deterministic_profile=deterministic_profile,
                                 sampler_type=sampler_type, od=od,
                                 continuum_method=continuum_method, use_jax=use_jax)

        self.continuum_slices = data.continuum_slices.get("mcmc")
        self.x = data.norm_wavelengths["mcmc"]
        self.y = data.flux["mcmc"]
        self.yerr = data.errors["mcmc"]
        self.alpha = data.alpha["mcmc"]
        self.velocities = data.velocities
        self.c_factor = data.c_factor.get("mcmc")
        self.deterministic_profile = data.config.deterministic_profile
        self.sampler_type = data.config.sampler_type
        self.od = data.config.od
        self.continuum_method = data.config.continuum_method
        self.use_jax = data.config.use_jax

        self.k_max = self.alpha.shape[1] # the number of velocity points in the profile
        self.regularization = data.config.regularization

        if self.od:
            # For deterministic model, the below variables are used, and are precomputed for speed
            err_od = self.yerr / self.y # independent of continuum, since it's a ratio
            V = 1.0 / (err_od ** 2) # variance vector in log space, error already in log space
        else:
            # In this case for deterministic sampling, the errors are not accurate. They are warned through the Config getters.
            # For non-OD case, we need to precompute the variance vector in flux space for the likelihood calculation
            V = 1.0 / (self.yerr ** 2) # variance vector in flux space

        self.AtV = self.alpha.T * V # precompute alpha matrix multiplication once
        if self.regularization:
            scales = np.ones(self.k_max // len(self.velocities))
            if data.config.scale_regularization:
                # Match LSD's scaling, using the same fitting-space errors.
                information = np.sum(self.alpha * self.AtV.T, axis=0)
                scales = LSD._regularization_scales(information, len(self.velocities))
            self._regularization_weights = self.regularization * scales

        # TODO: add an option for the diff and var be a comparison also in OD
        # Precompute the terms which are invariant for every likelihood call
        self._likelihood_var = self.yerr * self.yerr
        self._likelihood_log_norm = np.log(2 * np.pi * self._likelihood_var)

        # Configure whether to use full or deterministic model
        if self.deterministic_profile is False:
            #: The model function can be called to directly run the model based on the deterministic_profile flag from initialization
            self.model_function = self.full_model # include profile fitting
        else:
            self.model_function = self.deterministic_model # infer profile points from continuum

        # For dynesty
        if isinstance(x_or_data, Data):
            profile0 = np.asarray(data.profile["masked"][0]).reshape(-1)
            self.model_inputs = np.concatenate((profile0, data.poly_coeffs["masked"]))
        else:
            self.model_inputs = None

        self.np, self.linalg = numpy_module, linalg
        self.jax_enabled = False
        if self.use_jax:
            from .jax import _import_jax
            modules = _import_jax()
            if modules is not None:
                jax, self.np, self.linalg = modules
                self.jax_enabled = True

        if self.np is not np:
            # Precompute with NumPy first to preserve its input precision and promotion.
            for name in ("x", "y", "yerr", "alpha", "AtV", "_likelihood_var", "_likelihood_log_norm"):
                setattr(self, name, self.np.asarray(getattr(self, name), dtype=float))
            if self.c_factor is not None:
                self.c_factor = (self.np.asarray(self.c_factor[0], dtype=float), bool(self.c_factor[1]))
            if self.regularization:
                self._regularization_weights = self.np.asarray(self._regularization_weights)

        if self.jax_enabled:
            self.log_probability = jax.jit(self.log_probability)
            self.dynesty_logprob = jax.jit(self.dynesty_logprob)

        # Lastly, just warn once for determinstic non-OD option
        if self.deterministic_profile and not self.od:
            warnings.warn("Using precomputed AtV matrix for non-OD deterministic model; errors may not be accurate.\n" \
                "Explanation:\nFor deterministic (default) sampling (ie. inferring the profile from the continuum), the errors do not change in OD.\n" \
                "However, in flux space, the errors do depend on the continuum model, and AtV would need to be recomputed at each MCMC step.\n" \
                "On testing, this took a significant amount of time, and hence we use the precomputed AtV matrix on the assumption that the" \
                "continuum does not move much from the initial guess.\n" \
                "If you wish to get accurate errors, use deterministic=False, which will fit each profile point (we then recommend increasing " \
                "the number of steps).",
                ACIDInputWarning, stacklevel=2)

    def __call__(self, *args, **kwargs):
        # Sets the default call is the log_probability function
        return float(self.log_probability(*args, **kwargs))

    # Models
    # ------
    def eval_continuum(self, coefs, x=None):
        """Evaluate ordinary or log-Chebyshev continua, independently per order."""
        np = self.np
        coefs = np.asarray(coefs, dtype=float)
        if x is None:
            if self.continuum_slices is not None:
                return np.concatenate([self.eval_continuum(coefs[a:b], self.x[i:j])
                                       for i, j, a, b in self.continuum_slices])
            x = self.x
        if self.continuum_method == "polyval":
            value = coefs[-1] + x * 0
            for coef in coefs[-2::-1]:
                value = coef + value * x
            return value

        # Clenshaw recurrence is shared because JAX has no chebval.
        if len(coefs) == 1:
            c0, c1 = coefs[0], 0
        else:
            c0, c1 = coefs[-2], coefs[-1]
            for coef in coefs[-3::-1]:
                c0, c1 = coef - c1, c0 + c1 * (2 * x)
        return np.exp(c0 + c1 * x)

    def full_model(self, theta):
        """
        Full model for mcmc - takes all inputs (profile points + continuum coefficents)
        to create a model spectrum.

        Parameters
        ----------
        theta : np.ndarray
            Model parameters: first k_max values are profile points (z),
            followed by continuum polynomial coefficients and scale factor.
        
        Returns
        -------
        tuple
            Model spectrum and profile points (z).
        """
        # Extract profile points and continuum coefficients from theta
        np = self.np
        z = np.asarray(theta[:self.k_max], dtype=float).copy()
        if not self.od:
            z = z - 1 # if not using OD, profile points are in flux space and need to be shifted by 1
        mdl = self.alpha @ z

        # Converting model from optical depth to flux
        mdl = np.exp(-mdl) if self.od else mdl+1 # if not using OD, just use flux directly

        # Calculate continuum polynomial
        coefs = np.asarray(theta[self.k_max:], dtype=float)

        # Apply continuum model
        mdl = mdl * self.eval_continuum(coefs)

        return mdl, z

    def deterministic_model(self, theta):
        """
        Deterministic model for mcmc - takes only continuum coefficents and scale factor
        to create a model spectrum, solving for the profile points (z) directly.

        Parameters
        ----------
        theta : array-like
            Model parameters: continuum polynomial coefficients followed by scale factor.
        
        Returns
        -------
        tuple
            Model spectrum and profile points (z).
        """
        np = self.np
        coefs = np.asarray(theta, dtype=float)

        # Build continuum model
        mdl = self.eval_continuum(coefs)

        # Calculate fitted flux and convert to OD
        fitted_flux = self.y/mdl

        # Do OD/non-OD conversions
        if self.od:
            valid = np.all((fitted_flux > 0) & np.isfinite(fitted_flux))
            flux = -np.log(np.where(valid, fitted_flux, 1.0))
        else:
            valid = True
            flux = fitted_flux - 1

        # Solve for the profile points
        z = self.linalg.cho_solve(self.c_factor, self.AtV @ flux, check_finite=False)

        # Convert back from optical depth to flux
        dot_prod = self.alpha @ z
        dot_prod = np.exp(-dot_prod) if self.od else dot_prod + 1
        forward = dot_prod * mdl

        return np.where(valid, forward, fitted_flux), np.where(valid, z, -2)

    # Log probability
    # ---------------
    def log_probability(self, theta):
        """Log posterior for either profile model, using NumPy or JAX operations."""
        np = self.np
        theta = np.asarray(theta, dtype=float)
        forward, z = self.model_function(theta)
        lp = self.log_prior(theta, z)
        diff = self.y - forward
        ll = -0.5 * np.sum(diff * diff / self._likelihood_var + self._likelihood_log_norm)
        return np.where(np.isfinite(lp), lp + ll, -np.inf)

    def dynesty_logprob(self, theta):
        """Constrained log likelihood; dynesty applies the coefficient prior transform."""
        np = self.np
        theta = np.asarray(theta, dtype=float)
        forward, z = self.model_function(theta)
        lp = self.log_prior(theta, z)
        diff = self.y - forward
        ll = -0.5 * np.sum(diff * diff / self._likelihood_var + self._likelihood_log_norm)
        valid = np.all(np.isfinite(forward)) & np.isfinite(lp)
        return np.where(valid, ll + self.regularization_prior(z), -np.inf)

    # Priors
    # ------

    def continuum_prior(self, coefs):
        """Gaussian prior on log-Chebyshev coefficients; ordinary polynomials are unconstrained."""
        np = self.np
        if self.continuum_method != "chebval":
            return 0.0 # No prior for non-Chebyshev methods

        k = np.arange(len(coefs), dtype=float)
        sigma = np.where(k == 0, 1.0, 0.25 / (1.0 + k)**2)

        return -0.5 * np.sum((coefs / sigma)**2)

    def log_prior(self, theta, z):
        """
        Combine the soft profile constraints, regularization, and continuum priors.

        Parameters
        ----------
        theta : array-like
            Model parameters.
        z : array-like
            Profile points.

        Returns
        -------
        float
            Log prior probability.
        """

        np = self.np
        lp = self.soft_z_prior(z)
        valid = np.isfinite(lp)

        coefs = theta[self.k_max:] if not self.deterministic_profile else theta
        if self.continuum_slices is None:
            lp += self.continuum_prior(coefs)
        else:
            lp += sum(self.continuum_prior(coefs[a:b]) for _, _, a, b in self.continuum_slices)

        return np.where(valid, lp, -np.inf)

    def soft_z_prior(self, z):
        """
        Soft prior on the profile points (z) to prevent them from going too far outside the expected range.
        """
        np = self.np

        # Set the expected range of acceptable values, we dont expect "good" profiles to fall outside this
        if self.od:
            lo, hi = -0.5, 1.8
        else:
            lo, hi = -1.0, 0.5

        scale = 0.05
        below = np.maximum(lo - z, 0.0)
        above = np.maximum(z - hi, 0.0)

        # Set a gaussian penalty for values outside the expected range, with a scale factor to control the strength of the penalty
        prior = -0.5 * np.sum((below / scale)**2 + (above / scale)**2) + self.regularization_prior(z)
        return np.where(np.all(z == -2), -np.inf, prior)

    def regularization_prior(self, z):
        """First-difference prior, optionally information-scaled, independently per profile."""
        if not self.regularization:
            return 0.0
        np = self.np
        differences = np.diff(z.reshape(-1, len(self.velocities)), axis=1)
        return -0.5 * np.sum(self._regularization_weights[:, None] * differences**2)

    def ptform(self, u):
        """
        Map unit-cube samples to uniform continuum priors around the fitted coefficients.
        """
        u = np.asarray(u, dtype=float)
        theta0 = np.asarray(self.model_inputs, dtype=float)[self.k_max:]
        width = np.maximum(5 * np.abs(theta0), 0.05)
        if len(width) > 0:
            width[0] = max(width[0], 0.25)
        lower, upper = theta0 - width, theta0 + width
        return lower + u * (upper - lower)
