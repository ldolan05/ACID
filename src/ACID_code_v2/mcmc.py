import emcee
import numpy as np
import multiprocessing as mp

# Global storage for large arrays
GLOBAL_DATA = {}

def _init_worker(global_data):
    """Called once per worker."""
    global GLOBAL_DATA
    GLOBAL_DATA = global_data

def _log_probability(theta):
    """Worker-safe version of log-prob using global data."""
    global GLOBAL_DATA
    data = GLOBAL_DATA["data"]
    # Example likelihood
    return -0.5 * np.sum((data - theta[0]) ** 2)

def model_func(self, inputs, x):
        ## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
        z = inputs[:self.k_max]

        mdl = np.dot(self.alpha, z) ##alpha has been declared a global variable after LSD is run.

        #converting model from optical depth to flux
        mdl = np.exp(mdl)

        ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
        a = 2/(np.max(x)-np.min(x))
        b = 1 - a*np.max(x)

        mdl1 = 0
        for i in range(self.k_max, len(inputs) - 1):
            mdl1 = mdl1 + (inputs[i] * ((x * a) + b) ** (i - self.k_max))

        # coefs = np.asarray(inputs[k_max:-1], dtype=float) # Potential improvement - Ben
        # X = (a * x) + b
        # if coefs.size:
        #     powers = np.arange(coefs.size)
        #     # X[:, None] ** powers -> shape (len(x), coefs.size); dot with coefs -> (len(x),)
        #     mdl1 = np.dot(X[:, None] ** powers[None, :], coefs)
        # else:
        #     mdl1 = 0.0

        mdl1 = mdl1 * inputs[-1]
        
        mdl = mdl * mdl1
    
        return mdl

def log_likelihood(self, theta, x, y, yerr):
    ## maximum likelihood estimation for the mcmc model.
    model = self.model_func(theta, x)

    lnlike = -0.5 * np.sum(((y) - (model)) ** 2 / yerr**2 + np.log(yerr**2)+ np.log(2*np.pi))

    return lnlike

def log_prior(self, theta):
    ## imposes the prior restrictions on the inputs - rejects if profile point is less than -10 or greater than 0.5.
    check = 0
    z = theta[:self.k_max]

    for i in range(len(theta)):
        if i < self.k_max: ## must lie in z
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
                v_cont.append(self.velocities[len(self.velocities)-i-1])
                z_cont.append(np.exp(z[i])-1)
                v_cont.append(self.velocities[i])

        z_cont = np.array(z_cont)
        v_cont = np.array(v_cont)

        p_pent = np.sum((np.log((1/np.sqrt(2*np.pi*0.01**2)))-0.5*(z_cont/0.01)**2))

        return p_pent

    return -np.inf

def log_probability(self, theta):
    # TODO!: Turn to classes and potentially if possible move to separate file
    ## calculates log probability - used for mcmc
    lp = self.log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    final = lp + self.log_likelihood(theta, self.x, self.y, self.yerr)
    return final

def run_parallel_example(data, nwalkers=16, ndim=2, nsteps=500, cores=4):
    ctx = mp.get_context("fork")

    global_data = {"data": data}

    with ctx.Pool(processes=cores, initializer=_init_worker, initargs=(global_data,)) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, _log_probability, pool=pool)
        initial_state = np.random.randn(nwalkers, ndim)
        sampler.run_mcmc(initial_state, nsteps, progress=True)
        return sampler