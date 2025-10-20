import numpy as np
from ACID_code_v2.ACID import model_func


class Model:
    def __init__(self, model_func, x, y, yerr, velocities, k_max):
        self.model_func = model_func
        self.x = x
        self.y = y
        self.yerr = yerr
        self.velocities = velocities
        self.k_max = k_max

    def log_likelihood(self, theta, x, y, yerr):
        ## maximum likelihood estimation for the mcmc model.
        model = model_func(theta, x)

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
        ## calculates log probability - used for mcmc
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        final = lp + self.log_likelihood(theta, self.x, self.y, self.yerr)
        return final