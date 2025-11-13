import emcee
import numpy as np
import multiprocessing as mp


class MCMCUtils:
     
    def __init__(self, global_data=None):

        self.x = getattr(global_data, "x", None)
        self.y = getattr(global_data, "y", None)
        self.yerr = getattr(global_data, "yerr", None)
        self.alpha = getattr(global_data, "alpha", None)
        self.k_max = getattr(global_data, "k_max", None)
        self.velocities = getattr(global_data, "velocities", None)
    
    def __call__(self, theta):
        return self._log_probability(theta)

    def model_func(self, inputs, x, **kwargs):
        ## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
        alpha = kwargs.get("alpha", self.alpha)
        k_max = kwargs.get("k_max", self.k_max)
    
        z = inputs[:k_max]

        mdl = np.dot(alpha, z) ##alpha has been declared a global variable after LSD is run.

        #converting model from optical depth to flux
        mdl = np.exp(mdl)

        ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
        a = 2/(np.max(x)-np.min(x))
        b = 1 - a*np.max(x)

        mdl1 = 0
        for i in range(k_max, len(inputs) - 1):
            mdl1 = mdl1 + (inputs[i] * ((x * a) + b) ** (i - k_max))

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

    def _log_likelihood(self, theta, x, y, yerr):
        ## maximum likelihood estimation for the mcmc model.
        model = self.model_func(theta, x)

        lnlike = -0.5 * np.sum(((y) - (model)) ** 2 / yerr**2 + np.log(yerr**2)+ np.log(2*np.pi))

        return lnlike

    def _log_prior(self, theta):
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

    def _log_probability(self, theta):
        # TODO!: Turn to classes and potentially if possible move to separate file
        ## calculates log probability - used for mcmc
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        final = lp + self._log_likelihood(theta, self.x, self.y, self.yerr)
        return final
