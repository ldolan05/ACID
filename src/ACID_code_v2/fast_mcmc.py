import numpy as np
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

    global L_factor, AT_over_yerr
    # Weighted design matrix
    Aw = alpha / yerr[:, None]
    # Precompute AᵀA and its Cholesky factor
    ATA = Aw.T @ Aw
    L_factor = cho_factor(ATA, overwrite_a=True)
    # Precompute Aᵀ/σ for later RHS computation
    AT_over_yerr = Aw.T

def _log_prior_from_z(z):
    """
    Apply your original prior to a given LSD profile z (length k_max).
    This is basically the old _log_prior, but with z passed in instead
    of taken from theta[:k_max].
    """
    z = np.asarray(z, dtype=float)
    k_max_local = len(z)

    # Hard box prior on each z[i]
    if np.any((z < -10.0) | (z > 0.5)):
        return -np.inf

    # Gaussian penalty on continuum points (as before)
    z_cont = []
    v_cont = []

    for i in range(0, 5):
        z_cont.append(np.exp(z[k_max_local - i - 1]) - 1.0)
        v_cont.append(velocities[k_max_local - i - 1])

        z_cont.append(np.exp(z[i]) - 1.0)
        v_cont.append(velocities[i])

    z_cont = np.array(z_cont)

    # same p_pent as before
    p_pent = np.sum(
        (np.log(1.0 / np.sqrt(2.0 * np.pi * 0.01**2)))
        - 0.5 * (z_cont / 0.01) ** 2
    )

    return p_pent

def _solve_profile_and_model(theta_cont):
    """
    Given continuum parameters only (theta_cont), solve for best-fit
    LSD profile z_hat using precomputed Cholesky factors.
    """
    theta_cont = np.asarray(theta_cont, dtype=float)
    n_poly = len(theta_cont) - 1
    coefs = theta_cont[:n_poly]
    scale = theta_cont[-1]

    # --- Build continuum C(x) as before ---
    a = 2.0 / (np.max(x) - np.min(x))
    b = 1.0 - a * np.max(x)
    u = a * x + b

    cont = np.zeros_like(x, dtype=float)
    for c in reversed(coefs):
        cont = cont * u + c
    cont *= scale
    if np.any(cont <= 0):
        return None, None

    # --- Weighted RHS for the linear system ---
    rhs = AT_over_yerr @ np.log(y / cont)

    # --- Solve (AᵀA) z = Aᵀ log(y/cont)/σ² using cached Cholesky factor ---
    z_hat = cho_solve(L_factor, rhs)

    # --- Build forward model ---
    tau_model = alpha @ z_hat
    mdl = np.exp(tau_model) * cont

    return mdl, z_hat


def _log_probability(theta_cont):
    """
    Log-posterior as a function of continuum parameters only.

    theta_cont: [c0, ..., c_{n_poly-1}, scale]
    """
    # Solve for profile and model
    mdl, z_hat = _solve_profile_and_model(theta_cont)

    # If continuum was invalid (e.g. cont <= 0), reject
    if mdl is None:
        return -np.inf

    # Prior on z_hat (reusing your old structure)
    lp = _log_prior_from_z(z_hat)
    if not np.isfinite(lp):
        return -np.inf

    # Gaussian log-likelihood in flux space (same as your existing one)
    diff = y - mdl
    ll = -0.5 * np.sum(diff * diff / (yerr * yerr) +
                       np.log(2.0 * np.pi * (yerr * yerr)))

    return lp + ll

