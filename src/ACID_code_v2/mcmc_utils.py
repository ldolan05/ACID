import numpy as np
import multiprocessing as mp
from .lsd import LSD
from . import utils
from time import sleep
import matplotlib.pyplot as plt
import sys

# TODO: Move a,b init to _init_worker to avoid recomputation
class MCMC:

    def __init__(self, global_data):
        """Called once per worker."""
        self.x = global_data.get("x")
        self.y = global_data.get("y")
        self.yerr = global_data.get("yerr")
        self.alpha = global_data.get("alpha")
        self.velocities = global_data.get("velocities")
        self.k_max = self.alpha.shape[1]
        self.c_factor = global_data.get("c_factor")
        self.fit_profile = global_data.get("fit_profile")
        np.random.seed(global_data.get("seed"))

        # Configure whether to use full or fast model
        global model_function
        if self.fit_profile:
            model_function = self.full_func
        else:
            model_function = self.fast_func
        
        self.a, self.b = utils.get_normalisation_coeffs(self.x)

    def full_func(self, inputs, x, **kwargs):
        ## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
        alpha = kwargs.get("alpha", self.alpha)
        k_max = kwargs.get("k_max", alpha.shape[1])

        z = inputs[:k_max]

        mdl = np.dot(alpha, z)

        #converting model from optical depth to flux
        mdl = np.exp(mdl)

        ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
        a = 2/(np.max(x)-np.min(x))
        b = 1 - a*np.max(x)

        # Calculate continuum polynomial
        coefs = np.asarray(inputs[k_max:-1], dtype=float)
        scale = inputs[-1]
        u = (a * x) + b

        # Build continuum model
        mdl1 = 0.0
        for c in reversed(coefs):
            mdl1 = mdl1 * u + c
        mdl *= mdl1 * scale

        return mdl, z

    def fast_func(self, inputs, x, **kwargs):
        ## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
        alpha = kwargs.get("alpha", self.alpha)

        ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
        a, b = utils.get_normalisation_coeffs(x)

        coefs = np.asarray(inputs[:-1], dtype=float)
        scale = inputs[-1]
        u = (a * x) + b

        # Build continuum model
        mdl = 0.0
        for c in reversed(coefs):
            mdl = mdl * u + c
        mdl *= scale

        if np.any(mdl <= 0):
            return mdl, np.full(alpha.shape[1], 1) # return very low z to trigger prior rejection

        fitted_flux = self.y/mdl
        fitted_err = self.yerr/mdl
        err_od = fitted_err / fitted_flux
        flux_od = np.log(fitted_flux)

        z = LSD.solve_z(alpha, flux_od, err_od, self.c_factor, return_error=False)

        forward = np.exp(alpha @ z) * mdl

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
        forward, z = model_function(theta, self.x, alpha=self.alpha, k_max=self.k_max)

        lp = self._log_prior(z)
        if not np.isfinite(lp):
            return -np.inf

        diff = self.y - forward
        ll = -0.5 * np.sum(diff*diff / (self.yerr*self.yerr) + np.log(2*np.pi*(self.yerr*self.yerr)))
        return lp + ll
