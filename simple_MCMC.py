import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize
import emcee
#import corner
import LSD_func_faster as LSD
'''
fits_file = '/Users/lucydolan/Documents/Ernst_Rm_Codes/HD189733b_profiles/August2007_master_out_ccfs.fits'
linelist = '/Users/lucydolan/Documents/Least_Squares_Deconvolution/LSD/Archive_stuff/archive/fulllinelist018.txt'
directory = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/'
'''

fits_file = '/home/lsd/Documents/HD189733/August2007_master_out_ccfs.fits'
linelist = '/home/lsd/Documents/fulllinelist018.txt'
directory = '/home/lsd/Documents/HD189733/August2007/'

## get LSD profile and spectrum.
def get_data(file, frame, order):

    fluxes, wavelengths, flux_error_order = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'masked')
    velocities, profile, profile_errors, alpha = LSD.LSD(wavelengths, fluxes-1, flux_error_order, linelist)

    #alpha = np.reshape(alpha, (len(wavelengths)*len(velocities)))
    #alpha = np.array(alpha)
    profile = np.array(profile)
    inputs = profile
    #print(np.shape(alpha))
    #print(np.shape(profile))
    #inputs = np.concatenate((alpha, profile))
    '''
    plt.figure()
    plt.plot(velocities, profile)

    profile = profile+1
    plt.figure()
    plt.plot(velocities, profile)

    plt.figure()
    plt.plot(wavelengths, fluxes-1)
    #plt.show()
    '''
    return wavelengths, fluxes-1, flux_error_order, inputs, alpha
'''
def z_func(inputs, x):
    #print('o, w, h: %s'%theta)
    #print(phase, inputs)
    mdl = inputs[3]+(1+inputs[1]*np.exp(-(x-inputs[2])**2/2.*inputs[0]**2))
    for i in np.arange(4,len(inputs)):
        mdl = mdl+inputs[i]*x**(i-3)
    return mdl
'''
## thousands of parameters - must be varying alpha also
def model_func(inputs, x):
    ## make LSD profile flux the first input  - are you varying this as well
    ## setting up the alpha matrix depending on the size of the spectrum and velocity grid
    #alpha = inputs[:j_max*k_max]
    #print(inputs)
    z = inputs[:k_max]

    #alpha=np.reshape(alpha, (j_max, k_max))
    mdl = np.dot(alpha, z)
    mdl = mdl+1
    #plt.plot(x, mdl)
    #plt.show()
    mdl1=0
    for i in range(k_max,len(inputs)):
        mdl1 = mdl1 + (inputs[i]*x**(i-(k_max)))
        #print('y = %s*x**%s'%(inputs[i], (i-(k_max))))
        #print(inputs[i])
        #print(i-(k_max))
        #print(mdl1)
    '''
    plt.figure('poly and non-adjust mdl')
    plt.plot(x, mdl1, 'k')
    plt.plot(x, mdl)
    plt.show()


    plt.figure('adjusted mdl')
    plt.plot(x, mdl)
    plt.show()
    '''
    mdl = mdl * mdl1
    mdl = mdl-1

    return mdl

## maximum likelihood estimation for a gaussian with offset, o, width, w and height h.
def log_likelihood(theta, x, y, yerr):
    #print('o, w, h: %s'%theta)
    model = model_func(theta, x)
    #sigma2 = yerr ** 2 + model ** 2
    #lnlike = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    lnlike = -0.5 * np.sum((y - model) ** 2 / yerr**2)

    return lnlike

## caluclating maximum liklihood estimates for x, y and y_err
def likelihood_estimate(x, y, yerr, initial):
    #sigma_init, h_init, mu_init, a_init, b_init = initial
    nll = lambda *args: -log_likelihood(*args)
    #print(nll)
    #print(initial)
    soln = minimize(nll, initial, args=(x, y, yerr))
    #print(soln)
    model_inputs = soln.x
    return model_inputs , soln

## imposes the prior restrictions on the inputs
def log_prior(theta):

    check = 0
    #alpha = inputs[:j_max*k_max]
    z = theta[:k_max]
    for i in range(len(theta)):
        if i<k_max: ## must lie in z
            if -1.5<=theta[i]<0.5: pass
            else:check = 1
        if i>j_max*k_max+k_max:pass ## no restrictions on polynomial coefficents

    if check==0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    #print('hello')
    return lp + log_likelihood(theta, x, y, yerr)

def run_emcee(x, y, yerr, initial_inputs):

    model_inputs, soln = likelihood_estimate(x, y, yerr, initial_inputs)
    print(model_inputs)

    pos = soln.x + 0.01 * np.random.randn(10000, len(initial_inputs))
    #print(pos)
    #print(soln.x)
    #pos = initial_inputs + 1e-4 * np.random.randn(32, 2)
    nwalkers, ndim = pos.shape
    print(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, 5000, progress=True, skip_initial_state_check=True);
    '''
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["sigma", "h", "mu", "offset_in_y"]
    for l in range(4, len(model_inputs)):
        labels.append('poly_ord %s'%(l-3))
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    '''
    #tau = sampler.get_autocorr_time()
    #print("Tau: %s"%tau)

    #dis_no = int(input("How many values to discard: "))
    dis_no = 200
    #av_tau = np.sum(tau)/len(tau)
    flat_samples = sampler.get_chain(discard=dis_no, thin=15, flat=True)
    print(flat_samples.shape)

    #fig = corner.corner(flat_samples, labels=labels);

    plt.figure()
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        #plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
        plt.plot(x, model_func(sample, x), "C1", alpha=0.1)
    plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
    #plt.plot(x, np.exp(initial_inputs[1])*np.exp(-x**2/2.*initial_inputs[0]**2))
    #plt.plot(velocities, ccf_spec, "k", label="truth")
    #plt.legend(fontsize=14)
    #plt.xlim(0, 10)
    plt.xlabel("x")
    plt.ylabel("y");
    #plt.show()
    print(ndim)
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        #txt = txt.format(mcmc[1], q[0], q[1], labels[i])

    return x, y, yerr

def continuumfit(fluxes, wavelengths, errors, poly_ord):
        fluxes = fluxes+1
        idx = wavelengths.argsort()
        wavelength = wavelengths[idx]
        fluxe = fluxes[idx]
        clipped_flux = []
        clipped_waves = []
        binsize =100
        for i in range(0, len(wavelength), binsize):
            waves = wavelength[i:i+binsize]
            flux = fluxe[i:i+binsize]
            indicies = flux.argsort()
            flux = flux[indicies]
            waves = waves[indicies]
            clipped_flux.append(flux[len(flux)-1])
            clipped_waves.append(waves[len(waves)-1])
        coeffs=np.polyfit(clipped_waves, clipped_flux, poly_ord)
        '''
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)
        print('fit: %s'%fit)
        flux_obs = fluxes/fit-1
        new_errors = errors/fit

        print(coeffs)
        print(poly)
        fig = plt.figure('fit')
        plt.plot(wavelengths, fluxes)
        plt.plot(wavelengths, fit)
        plt.plot(wavelengths, flux_obs)
        #plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        plt.show()
        '''
        return coeffs

#file = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_e2ds_A.fits'
file = '/home/lsd/Documents/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_e2ds_A.fits'
frame = 0
order = 28

#fits_file = fits.open(file)
wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1 = get_data(file, frame, order)

## making alpha a global variable
alpha = alpha1
## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
j_max = int(len(flux_init))
k_max = len(initial_inputs)
#k_max = int(len(initial_inputs)/(j_max+1))
print(k_max)

#poly_ord = input('Enter order of polynomial for continuum fit:')
poly_ord = 1
poly_inputs=continuumfit(flux_init,  wavelength_init, flux_error_init, poly_ord)
poly_inputs=poly_inputs[::-1]
print(poly_inputs)
#poly_inputs = [0.01]*poly_ord
poly_inputs = np.array(poly_inputs)
initial_inputs = np.concatenate((initial_inputs, poly_inputs))

print('Number of inputs: %s'%len(initial_inputs))
print(initial_inputs)
#for order in range(0, 72):
#wavelength, flux, flux_error = run_emcee(wavelength_init, flux_init, flux_error_init, initial_inputs)
#plt.close('all')
x = wavelength_init
y = flux_error_init
yerr = flux_error_init

model_inputs, soln = likelihood_estimate(x, y, yerr, initial_inputs)
print(model_inputs)

pos = soln.x + 0.1 * np.random.randn(10000, len(initial_inputs))
#print(pos)
#print(soln.x)
#pos = initial_inputs + 1e-4 * np.random.randn(32, 2)
nwalkers, ndim = pos.shape
print(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True, skip_initial_state_check=True);
'''
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["sigma", "h", "mu", "offset_in_y"]
for l in range(4, len(model_inputs)):
    labels.append('poly_ord %s'%(l-3))
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
'''
#tau = sampler.get_autocorr_time()
#print("Tau: %s"%tau)

#dis_no = int(input("How many values to discard: "))
dis_no = 200
#av_tau = np.sum(tau)/len(tau)
flat_samples = sampler.get_chain(discard=dis_no, thin=15, flat=True)
print(flat_samples.shape)

#fig = corner.corner(flat_samples, labels=labels);

plt.figure()
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    #plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
    plt.plot(x, model_func(sample, x), "C1", alpha=0.1)
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, model_func(model_inputs, x))
#plt.plot(velocities, ccf_spec, "k", label="truth")
#plt.legend(fontsize=14)
#plt.xlim(0, 10)
plt.xlabel("x")
plt.ylabel("y");
#plt.show()
print(ndim)
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    q = np.diff(mcmc)
    #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    #txt = txt.format(mcmc[1], q[0], q[1], labels[i])

wavelength = x
flux = y
flux_error = yerr

plt.show()
