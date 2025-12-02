import emcee
import numpy as np
import multiprocessing as mp
from scipy.linalg import cho_factor, cho_solve

def _init_worker(global_data):
    """Called once per worker."""
    global x, y, yerr, alpha, k_max, velocities
    x = global_data["x"]
    y = global_data["y"]
    yerr = global_data["yerr"]
    alpha = global_data["alpha"]
    velocities = global_data["velocities"]
    k_max = alpha.shape[1]
    np.random.seed(global_data["seed"])

    # Precompute for fast solving
    global L_factor, AT_over_yerr
    # Weighted design matrix
    Aw = alpha / yerr[:, None]
    # Precompute AᵀA and its Cholesky factor
    ATA = Aw.T @ Aw
    L_factor = cho_factor(ATA, overwrite_a=True)
    # Precompute Aᵀ/σ for later RHS computation
    AT_over_yerr = Aw.T

    global current_z

def solve_z(theta_cont):
    """
    Analytic solve for z | theta_cont.
    This is the fast block replacing sampling z in emcee.
    """
    # Unpack continuum parameters
    coefs = theta_cont[:-1]
    scale = theta_cont[-1]

    # Build continuum
    a = 2/(np.max(x)-np.min(x))
    b = 1 - a*np.max(x)
    u = a*x + b

    cont = np.zeros_like(x)
    for c in reversed(coefs):
        cont = cont*u + c
    cont *= scale

    if np.any(cont <= 0):
        return None

    # RHS = AT_over_yerr @ log(y/cont)
    rhs = AT_over_yerr @ np.log(y/cont)

    # z_hat = solve (A^T A)z = RHS
    z_hat = cho_solve(L_factor, rhs)

    return z_hat

def model_func(inputs, x, **kwargs):
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

        # mdl1 = 0
        # for i in range(k_max, len(inputs) - 1):
        #     mdl1 = mdl1 + (inputs[i] * ((x * a) + b) ** (i - k_max))
        # mdl1 = mdl1 * inputs[-1]
        # mdl = mdl * mdl1

        # The above commented section is replaced with the following for enormous speedup
        coefs = np.asarray(inputs[k_max:-1], dtype=float)
        scale = inputs[-1]
        u = (a * x) + b

        mdl1 = 0.0
        for c in reversed(coefs):
            mdl1 = mdl1 * u + c
        mdl *= mdl1 * scale

        return mdl

def _log_likelihood(theta, x, y, yerr):
    model = model_func(theta, x)
    diff = y - model
    return -0.5 * np.sum(diff*diff / (yerr*yerr) + np.log(2*np.pi*(yerr*yerr)))

def _log_prior(theta):
    ## imposes the prior restrictions on the inputs - rejects if profile point is less than -10 or greater than 0.5.
    check = 0
    z = theta[:k_max]

    for i in range(len(theta)):
        if i < k_max: ## must lie in z
            if -10 <= theta[i] <= 0.5:
                pass
            else:
                check = 1

    if check == 0:

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

    return -np.inf

def _log_probability(theta):
    ## calculates log probability - used for mcmc
    # theta are now just cont_params
    # print(current_theta_array.shape, current_z.shape, theta.shape)
    # idx = np.where((current_theta_array == theta).all(axis=1))[0][0]
    full_theta = np.concatenate([current_z, theta])

    lp = _log_prior(full_theta)
    if not np.isfinite(lp):
        return -np.inf
    final = lp + _log_likelihood(full_theta, x, y, yerr)
    return final

