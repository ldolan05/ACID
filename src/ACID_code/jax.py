"""Optional JAX backend for the MCMC log-probability calculation."""
from __future__ import annotations
import warnings, os
import numpy as np


def _import_jax():
    """Import the optional JAX stack only when the backend is requested."""
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg as jsp_linalg
    except ImportError:
        return None
    return jax, jnp, jsp_linalg


class JAXBackend:
    """
    JAX implementation of the numerical work in :class:`MCMC`.
    This class has all the copies of the standard functions used in MCMC but with JAX compilations.
    """

    def __init__(self, mcmc) -> None:
        self.enabled = False

        # Importing JAX can initialise threads that are unsafe to inherit through
        # SLURM's fork-based worker pool, so detect SLURM before importing it.
        # if "SLURM_JOB_ID" in os.environ:
        #     warnings.warn(
        #         "JAX is disabled in SLURM environment; falling back to NumPy/SciPy.",
        #         RuntimeWarning,
        #         stacklevel=3,
        #     )
        #     return

        # Import JAX only when it has been requested
        # ------------------------------------------
        jax_modules = _import_jax()
        if jax_modules is None:
            # TODO: All warnings put out with print should be logged like this
            warnings.warn("use_jax=True was requested, but JAX is not importable; falling back to NumPy/SciPy.", RuntimeWarning, stacklevel=3)
            return

        self.jax, self.jnp, self.jsp_linalg = jax_modules

        # ACID and its Cholesky solve use double precision, so enable the equivalent JAX mode
        try:
            self.jax.config.update("jax_enable_x64", True)
        except Exception as exc:
            warnings.warn(f"JAX could not enable 64-bit calculations ({exc}); falling back to NumPy/SciPy.", RuntimeWarning, stacklevel=3)
            return

        # Prepare the static JAX inputs
        # -----------------------------
        # FITS flux/error arrays are often float32 while alpha is float64. Explicitly using float64 preserves NumPy's promotion behaviour.
        self.x                     = self.jnp.asarray(mcmc.x, dtype=self.jnp.float64)
        self.y                     = self.jnp.asarray(mcmc.y, dtype=self.jnp.float64)
        self.alpha                 = self.jnp.asarray(mcmc.alpha, dtype=self.jnp.float64)
        self.AtV                   = self.jnp.asarray(mcmc.AtV, dtype=self.jnp.float64)
        self.variance              = self.jnp.asarray(mcmc._likelihood_var, dtype=self.jnp.float64)
        self.log_norm              = self.jnp.asarray(mcmc._likelihood_log_norm, dtype=self.jnp.float64)
        self.k_max                 = mcmc.k_max
        self.od                    = mcmc.od
        self.deterministic_profile = mcmc.deterministic_profile
        self.continuum_method      = mcmc.continuum_method
        self.neg_inf               = self.jnp.asarray(-np.inf, dtype=self.y.dtype)

        if self.deterministic_profile:
            factor        = self.jnp.asarray(mcmc.c_factor[0], dtype=self.jnp.float64)
            lower         = bool(np.asarray(mcmc.c_factor[1]).item())
            self.c_factor = (factor, lower)

        # Compile the scalar entry points used by emcee and dynesty. Batching was slower for the Cholesky-heavy model on the benchmark CPU.
        self._log_probability = self.jax.jit(self._calculate_log_probability)
        self._log_likelihood = self.jax.jit(self._calculate_log_likelihood)
        self.enabled = True

    def log_probability(self, theta):
        """Return the JAX log posterior as the scalar expected by emcee."""
        return float(self._log_probability(np.asarray(theta, dtype=float)))

    def log_likelihood(self, theta):
        """Return the JAX log likelihood as the scalar expected by dynesty."""
        return float(self._log_likelihood(np.asarray(theta, dtype=float)))

    def _calculate_log_probability(self, theta):
        return self._calculate_probabilities(theta)[0]

    def _calculate_log_likelihood(self, theta):
        return self._calculate_probabilities(theta)[1]

    def _eval_continuum(self, coefs):
        # The basis and coefficient count are static, so JAX compiles away these branches and loops
        if self.continuum_method == "polyval":
            continuum = self.jnp.zeros_like(self.x)
            for coef in coefs[::-1]:
                continuum = continuum * self.x + coef
            return continuum

        if self.continuum_method != "chebval":
            raise ValueError(f"Unknown method: '{self.continuum_method}', must be 'polyval' or 'chebval'.")

        # Use the Clenshaw recurrence, equivalent to numpy.polynomial.chebyshev.chebval
        b_kplus1 = self.jnp.zeros_like(self.x)
        b_kplus2 = self.jnp.zeros_like(self.x)
        for coef in coefs[:0:-1]:
            b_k = coef + 2 * self.x * b_kplus1 - b_kplus2
            b_kplus2, b_kplus1 = b_kplus1, b_k
        log_continuum = coefs[0] + self.x * b_kplus1 - b_kplus2
        return self.jnp.exp(log_continuum)

    def _soft_z_prior(self, z):
        if self.od:
            lo, hi = -0.5, 1.8
        else:
            lo, hi = -1.0, 0.5

        below = self.jnp.maximum(lo - z, 0.0)
        above = self.jnp.maximum(z - hi, 0.0)
        prior = -0.5 * self.jnp.sum((below / 0.05) ** 2 + (above / 0.05) ** 2)
        return self.jnp.where(self.jnp.all(z == -2), self.neg_inf, prior)

    def _continuum_prior(self, coefs):
        if self.continuum_method != "chebval":
            return self.jnp.asarray(0.0, dtype=self.y.dtype)

        k = self.jnp.arange(coefs.shape[0], dtype=self.y.dtype)
        sigma = 0.25 / (1.0 + k) ** 2
        if coefs.shape[0] > 0:
            sigma = sigma.at[0].set(1.0)
        return -0.5 * self.jnp.sum((coefs / sigma) ** 2)

    def _probabilities_from_model(self, forward, z, coefs):
        lp = self._soft_z_prior(z) + self._continuum_prior(coefs)
        diff = self.y - forward
        ll = -0.5 * self.jnp.sum(diff * diff / self.variance + self.log_norm)

        posterior           = self.jnp.where(self.jnp.isfinite(lp), lp + ll, self.neg_inf)
        likelihood_is_valid = self.jnp.all(self.jnp.isfinite(forward)) & self.jnp.isfinite(lp)
        likelihood          = self.jnp.where(likelihood_is_valid, ll, self.neg_inf)
        return posterior, likelihood

    def _calculate_probabilities(self, theta):
        # The full model fits both profile and continuum parameters
        if not self.deterministic_profile:
            z = theta[:self.k_max]
            if not self.od:
                z = z - 1
            dot_prod = self.alpha @ z
            line_model = self.jnp.exp(-dot_prod) if self.od else dot_prod + 1
            coefs = theta[self.k_max:]
            forward = line_model * self._eval_continuum(coefs)
            return self._probabilities_from_model(forward, z, coefs)

        # The deterministic model solves the profile from the continuum parameters
        coefs = theta
        continuum = self._eval_continuum(coefs)
        fitted_flux = self.y / continuum

        if self.od:
            valid = self.jnp.all((fitted_flux > 0) & self.jnp.isfinite(fitted_flux))
            return self.jax.lax.cond(valid, self._solve_od_model, self._invalid_model, (fitted_flux, continuum, coefs))

        return self._solve_and_evaluate(fitted_flux - 1, continuum, coefs)

    def _solve_od_model(self, inputs):
        fitted_flux, continuum, coefs = inputs
        return self._solve_and_evaluate(-self.jnp.log(fitted_flux), continuum, coefs)

    def _invalid_model(self, inputs):
        return self.neg_inf, self.neg_inf

    def _solve_and_evaluate(self, flux, continuum, coefs):
        z = self.jsp_linalg.cho_solve(self.c_factor, self.AtV @ flux, check_finite=False)
        dot_prod = self.alpha @ z
        line_model = self.jnp.exp(-dot_prod) if self.od else dot_prod + 1
        forward = line_model * continuum
        return self._probabilities_from_model(forward, z, coefs)
