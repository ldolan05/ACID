import numpy as np
from .lsd import LSD
from . import utils
from .data import Data
from beartype import beartype
from numpy.polynomial import polynomial as P

# The following two wrapper functions are required for multiprocessing
# support, without it, the fork method would need to reserialize everything
# which is very inefficient. See parallelization in the emcee documentation
# for more details.
def _mp_init_worker(data):
    """Initializes each worker process with global data."""
    global _MCMC
    _MCMC = MCMC(data)

def _mp_log_probability(theta):
    """Wrapper for log probability function for multiprocessing."""
    return _MCMC(theta)

class MCMC:

    @beartype
    def __init__(
            self,
            x_or_data   : np.ndarray|Data,
            y           : np.ndarray      = None,
            yerr        : np.ndarray      = None,
            alpha       : np.ndarray      = None,
            velocities  : np.ndarray|None = None,
            c_factor                      = None,
            deterministic_profile : bool  = False,
        ):
        """Initialise MCMC functions with necessary data.
        Called once per worker if using multiprocessing.

        Parameters
        ----------
        x_or_data : np.ndarray|Data
            Wavelength array or Data object. If a Data object is provided, takes all 
            the arguments below from there. If a Data object is provided, all other 
            arguments are ignored.
        y : np.ndarray, optional
            Observed flux array, required if x_or_data is a np.ndarray.
        yerr : np.ndarray, optional
            Observed flux error array, required if x_or_data is a np.ndarray.
        alpha : np.ndarray, optional
            Precomputed alpha matrix, required if x_or_data is a np.ndarray.
        velocities : np.ndarray|None, optional
            Velocity grid for LSD profile, only needed when calling log-probability
            function, by default None.
        c_factor : optional
            Precomputed c_factor for LSD profile calculation, by default None.
        deterministic_profile : bool, optional
            Whether to fit the full profile (True) or use the fast model (False), by default True.
        """

        # No checks are performed here - assume data is valid from ACID class checks,
        # else user is on their own!
        if isinstance(x_or_data, Data):
            data = x_or_data
            self.x = data.wavelengths["masked"]
            self.y = data.flux["masked"]
            self.yerr = data.errors["masked"]
            self.alpha = data.alpha
            self.velocities = data.velocities
            self.c_factor = data.c_factor
            self.deterministic_profile = data.config.deterministic_profile

        else:
            self.x = x_or_data
            self.y = y
            self.yerr = yerr
            self.alpha = alpha
            self.velocities = velocities
            self.c_factor = c_factor
            self.deterministic_profile = deterministic_profile

        self.k_max = self.alpha.shape[1] # the number of velocity points in the profile

        # Precompute normalization coefficients these are used to adjust the wavelengths to
        # between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
        a, b = utils.get_normalisation_coeffs(self.x)
        self.u = (a * self.x) + b

        # Configure whether to use full or deterministic model
        if self.deterministic_profile is False:
            self.model_function = self.full_model # include profile fitting
        else:
            self.model_function = self.deterministic_model # infer profile points from continuum

    def __call__(self, *args, **kwargs):
        # Sets the default call is the log_probability function
        return self.log_probability(*args, **kwargs)

    def full_model(self, theta):
        """Full model for mcmc - takes all inputs (profile points + continuum coefficents)
        to create a model spectrum.

        Parameters
        ----------
        theta : array-like
            Model parameters: first k_max values are profile points (z),
            followed by continuum polynomial coefficients and scale factor.
        
        Returns
        -------
        tuple
            Model spectrum and profile points (z).
        """
        z = theta[:self.k_max]
        mdl = self.alpha @ z

        # Converting model from optical depth to flux
        mdl = np.exp(-mdl)

        # Calculate continuum polynomial
        coefs = np.asarray(theta[self.k_max:], dtype=float)

        # Apply continuum model
        mdl *= P.polyval(self.u, coefs)

        return mdl, z

    def deterministic_model(self, theta):
        """Deterministic model for mcmc - takes only continuum coefficents and scale factor
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
        coefs = np.asarray(theta, dtype=float)

        # Build continuum model
        mdl = P.polyval(self.u, coefs)

        if np.any(mdl <= 0): # force positive continuum at all points
            return mdl, np.full(self.k_max, -2) # return very low z to trigger prior rejection

        fitted_flux = self.y/mdl
        fitted_err = self.yerr/mdl
        err_od = fitted_err / fitted_flux
        flux_od = - np.log(fitted_flux)

        z = LSD.solve_z(self.alpha, flux_od, err_od, self.c_factor, return_error=False)
        # TODO: See if I can try removing the exp+log and get profiles without those steps
        # TODO: And see if the flux_to_od function is just as fast as below and 2 above
        forward = np.exp(- (self.alpha @ z)) * mdl

        return forward, z

    def run_model_function(self, *args, **kwargs):
        """Runs the selected model function (full or fast) with given arguments.

        Returns
        -------
        tuple
            Model spectrum and profile points (z).
        """
        return self.model_function(*args, **kwargs)

    def log_prior(self, z):
        """Calculates the log prior probability of the profile points (z) and imposes the prior
        restrictions on the inputs - rejects if profile point is less than -0.5 or greater than 1.5.

        Parameters
        ----------
        z : array-like
            Profile points.

        Returns
        -------
        float
            Log prior probability.
        """

        # Hard box prior on each z[i]
        if np.any((z < -0.3) | (z > 1.5)):
            return -np.inf

        # # excluding the continuum points in the profile (in flux)
        # z_cont = []
        # v_cont = []
        # for i in range(0, 5):
        #         z_cont.append(np.exp(z[len(z)-i-1])-1)
        #         v_cont.append(self.velocities[len(self.velocities)-i-1])
        #         z_cont.append(np.exp(z[i])-1)
        #         v_cont.append(self.velocities[i])

        # z_cont = np.array(z_cont)
        # v_cont = np.array(v_cont)

        # p_pent = np.sum((np.log((1/np.sqrt(2*np.pi*0.01**2)))-0.5*(z_cont/0.01)**2))

        return 0 

    def log_probability(self, theta):
        """Calculates log probability depending on which model (full or fast).
        
        Parameters
        ----------
        theta : array-like
            Model parameters.
        
        Returns
        -------
        float
            Log probability.
        """
        forward, z = self.model_function(theta)
    
        lp = self.log_prior(z)
        if not np.isfinite(lp):
            return -np.inf

        diff = self.y - forward
        ll = -0.5 * np.sum(diff*diff / (self.yerr*self.yerr) + np.log(2*np.pi*(self.yerr*self.yerr)))
        return lp + ll

    @staticmethod
    def get_mcmc_stopping_criterion(tau_list, step_number, min_checks, min_tau_factor, tau_rel_tol):
        # --- always try to compute a tolerance metric as early as possible ---
        tol_metric = np.nan  # show NaN until computable
        tol_converged = False
        n_eff = 0

        if len(tau_list) >= 2:
            # compute a "most recent" change metric from the last two tau estimates
            a, b = tau_list[-2], tau_list[-1]
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                den = np.maximum(b, 1e-12)
                tol_metric = float(np.percentile(np.abs(b - a) / den, 90))
            n_eff = int(step_number / np.max(tau_list))
            if np.isnan(tol_metric):
                False, np.inf, n_eff

        # --- now decide if we should STOP (stricter) ---
        # Need enough tau estimates to do min_checks consecutive deltas
        if len(tau_list) < (min_checks + 1):
            return False, tol_metric, n_eff

        tau = tau_list[-1]
        if not np.all(np.isfinite(tau)):
            return False, tol_metric, n_eff
        if np.max(tau) <= 1:
            return False, tol_metric, n_eff

        tau_ref = np.max(tau)
        if step_number < 10 * tau_ref:
            return False, tol_metric, n_eff
        if not (step_number > min_tau_factor * tau_ref):
            return False, tol_metric, n_eff

        # Stability over last min_checks deltas
        recent_tau = tau_list[-(min_checks + 1):]
        rel_changes = []
        for a, b in zip(recent_tau[:-1], recent_tau[1:]):
            den = np.maximum(b, 1e-12)
            rel_changes.append(np.percentile(np.abs(b - a) / den, 90))

        tol_metric = float(np.max(rel_changes))  # overwrite with the stricter metric
        return (tol_metric < tau_rel_tol), tol_metric, n_eff

    @staticmethod
    def get_tqdm_desc(last_tolerance, last_neff, config):
        tol_str = "<" if last_tolerance < config.tau_tol else ">"
        tol_str = f"{last_tolerance:.4f}{tol_str}{config.tau_tol}"
        neff_str = ">" if last_neff > config.min_tau_factor else "<"
        neff_str = f"{last_neff:.2f}{neff_str}{config.min_tau_factor}"
        return tol_str, neff_str