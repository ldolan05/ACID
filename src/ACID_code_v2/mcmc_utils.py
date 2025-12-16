import numpy as np
import multiprocessing as mp
from .lsd import LSD
from . import utils
from time import sleep
import matplotlib.pyplot as plt
import sys

def _init_worker(global_data):
    """Called once per worker."""
    global x, y, yerr, alpha, k_max, velocities, c_factor, fit_profile
    x = global_data["x"]
    y = global_data["y"]
    yerr = global_data["yerr"]
    alpha = global_data["alpha"]
    velocities = global_data["velocities"]
    k_max = alpha.shape[1]
    c_factor = global_data["c_factor"]
    fit_profile = global_data["fit_profile"]
    np.random.seed(global_data["seed"])

    global model_function
    if global_data["fit_profile"]:
        model_function = full_func
    else:
        model_function = fast_func

def full_func(inputs, x, **kwargs):
        ## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
        alpha = kwargs.get("alpha", globals().get("alpha"))
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

def fast_func(inputs, x, **kwargs):
    ## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
    alpha = kwargs.get("alpha", globals().get("alpha"))

    ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
    a, b = utils.get_normalisation_coeffs(x)

    coefs = np.asarray(inputs[:-1], dtype=float)
    scale = inputs[-1]
    u = (a * x) + b

    mdl = 0.0
    for c in reversed(coefs):
        mdl = mdl * u + c
    mdl *= scale

    if np.any(mdl <= 0):
        return mdl, np.full(alpha.shape[1], 1) # return very low z to trigger prior rejection


    fitted_flux = y/mdl
    fitted_err = yerr/mdl
    err_od = fitted_err / fitted_flux
    flux_od = np.log(fitted_flux)

    z = LSD.solve_z(alpha, flux_od, err_od, c_factor, return_error=False)

    forward = np.exp(alpha @ z) * mdl

    return forward, z

def _log_prior(z):
    ## imposes the prior restrictions on the inputs - rejects if profile point is less than -10 or greater than 0.5.

    # Hard box prior on each z[i]
    if np.any((z < -10.0) | (z > 0.5)):
        return -np.inf

    # excluding the continuum points in the profile (in flux)
    z_cont = []
    v_cont = []
    for i in range(0, 5):
            z_cont.append(np.exp(z[len(z)-i-1])-1)
            v_cont.append(velocities[len(velocities)-i-1])
            z_cont.append(np.exp(z[i])-1)
            v_cont.append(velocities[i])

    z_cont = np.array(z_cont)
    v_cont = np.array(v_cont)

    p_pent = np.sum((np.log((1/np.sqrt(2*np.pi*0.01**2)))-0.5*(z_cont/0.01)**2))

    return p_pent

def _log_probability(theta):
    ## calculates log probability depending on which model (full or fast)
    forward, z = model_function(theta, x, alpha=alpha, k_max=k_max)

    lp = _log_prior(z)
    if not np.isfinite(lp):
        return -np.inf

    diff = y - forward
    ll = -0.5 * np.sum(diff*diff / (yerr*yerr) + np.log(2*np.pi*(yerr*yerr)))
    return lp + ll
