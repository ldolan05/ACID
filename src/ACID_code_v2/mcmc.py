import numpy as np
from .lsd import LSD
from . import utils
from beartype import beartype
from numpy import integer as npint

# The following two wrapper functions are required for multiprocessing
# support, without it, the fork method would need to reserialize everything
# which is very inefficient. See parallelization in the emcee documentation
# for more details.
def _mp_init_worker(global_data):
    """Initializes each worker process with global data."""
    global _MCMC
    _MCMC = MCMC(**global_data)

def _mp_log_probability(theta):
    """Wrapper for log probability function for multiprocessing."""
    return _MCMC(theta)

@beartype
class MCMC:

    def __init__(
            self,
            x           : np.ndarray,
            y           : np.ndarray,
            yerr        : np.ndarray,
            alpha       : np.ndarray,
            velocities  : np.ndarray|None = None, # 
            c_factor                      = None,
            fit_profile : bool            = True,
            seed        : int|npint|None  = None,
        ):
        """Initialise MCMC functions with necessary data.
        Called once per worker if using multiprocessing."""
        
        # No checks are performed here - assume data is valid from ACID class checks,
        # else user is on their own!
        self.x = x
        self.y = y
        self.yerr = yerr
        self.alpha = alpha
        self.velocities = velocities
        self.k_max = self.alpha.shape[1]
        self.c_factor = c_factor
        self.fit_profile = fit_profile
        np.random.seed(seed)

        # Configure whether to use full or fast model
        if self.fit_profile:
            self.model_function = self.full_func
        else:
            self.model_function = self.fast_func
        
        # Precompute normalization coefficients these are used to adjust the wavelengths to
        # between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
        
        self.a, self.b = utils.get_normalisation_coeffs(self.x)
    
    def __call__(self, *args, **kwargs):
        return self._log_probability(*args, **kwargs)

    def full_func(self, theta):

        z = theta[:self.k_max]
        mdl = np.dot(self.alpha, z)

        #converting model from optical depth to flux
        mdl = np.exp(mdl)

        # Calculate continuum polynomial
        coefs = np.asarray(theta[self.k_max:-1], dtype=float)
        scale = theta[-1]
        u = (self.a * self.x) + self.b

        # Build continuum model
        mdl1 = 0.0
        for c in reversed(coefs):
            mdl1 = mdl1 * u + c
        mdl *= mdl1 * scale

        return mdl, z

    def fast_func(self, inputs):
        ## faster model for mcmc - takes the continuum coefficents(inputs) to create a model spectrum.

        coefs = np.asarray(inputs[:-1], dtype=float)
        scale = inputs[-1]
        u = (self.a * self.x) + self.b

        # Build continuum model
        mdl = 0.0
        for c in reversed(coefs):
            mdl = mdl * u + c
        mdl *= scale

        if np.any(mdl <= 0):
            return mdl, np.full(self.k_max, 1) # return very low z to trigger prior rejection

        fitted_flux = self.y/mdl
        fitted_err = self.yerr/mdl
        err_od = fitted_err / fitted_flux
        flux_od = np.log(fitted_flux)

        z = LSD.solve_z(self.alpha, flux_od, err_od, self.c_factor, return_error=False)

        forward = np.exp(self.alpha @ z) * mdl

        return forward, z

    def _log_prior(self, z):
        ## imposes the prior restrictions on the inputs - rejects if profile point is less than -10 or greater than 0.5.

        # Hard box prior on each z[i]
        if np.any((z < -10.0) | (z > 0.5)):
            return -np.inf

        # excluding the continuum points in the profile (in flux)
        z_cont = []
        v_cont = []
        for i in range(0, 5):
                z_cont.append(np.exp(z[len(z)-i-1])-1)
                v_cont.append(self.velocities[len(self.velocities)-i-1])
                z_cont.append(np.exp(z[i])-1)
                v_cont.append(self.velocities[i])

        z_cont = np.array(z_cont)
        v_cont = np.array(v_cont)

        p_pent = np.sum((np.log((1/np.sqrt(2*np.pi*0.01**2)))-0.5*(z_cont/0.01)**2))

        return p_pent

    def _log_probability(self, theta):
        ## calculates log probability depending on which model (full or fast)
        forward, z = self.model_function(theta)
    
        lp = self._log_prior(z)
        if not np.isfinite(lp):
            return -np.inf

        diff = self.y - forward
        ll = -0.5 * np.sum(diff*diff / (self.yerr*self.yerr) + np.log(2*np.pi*(self.yerr*self.yerr)))
        return lp + ll
