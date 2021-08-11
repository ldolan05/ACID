import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize
import emcee
#import corner
import LSD_func_faster as LSD
import time
import synthetic_data as syn

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
    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux= LSD.LSD(wavelengths, fluxes, flux_error_order, linelist, 'True')
    #alpha = np.reshape(alpha, (len(wavelengths)*len(velocities)))
    #alpha = np.array(alpha)
    #profile = np.array((profile/mp.max(profile))-1)
    profile = np.array(profile)
    print(len(profile))
    #print(np.shape(alpha))
    #print(np.shape(profile))
    #inputs = np.concatenate((alpha, profile))

    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.title('Profile from LSD (no continuum correction)')
    plt.savefig('/home/lsd/Documents/original_profile_LSD.png')
    '''
    profile = profile+1
    plt.figure()
    plt.plot(velocities, profile)

    plt.figure()
    plt.plot(wavelengths, fluxes-1)
    #plt.show()
    '''

    inputs = profile
    print(profile)
    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.title('Profile from LSD (continuum correction)')

    return wavelengths, fluxes-1, flux_error_order, inputs, alpha, velocities, continuum_waves, continuum_flux

def continuumfit_profile(fluxes, wavelengths, errors, poly_ord):
        fluxes = fluxes+1


        idx = wavelengths.argsort()
        wavelength = wavelengths[idx]
        fluxe = fluxes[idx]
        clipped_flux = []
        clipped_waves = []
        binsize =20
        for i in range(0, len(wavelength), binsize):
            waves = wavelength[i:i+binsize]
            flux = fluxe[i:i+binsize]
            indicies = flux.argsort()
            flux = flux[indicies]
            waves = waves[indicies]
            clipped_flux.append(flux[len(flux)-2])
            clipped_waves.append(waves[len(waves)-2])
        coeffs=np.polyfit(clipped_waves, clipped_flux, poly_ord)
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)
        flux_obs = fluxes/fit -1
        new_errors = errors/fit

        flux_obs[len(flux_obs)-3:] = 0
        flux_obs[:2] = 0

        fig = plt.figure('fit')
        plt.plot(wavelengths, fluxes)
        plt.plot(wavelengths, fit)
        plt.plot(wavelengths, flux_obs)
        plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        plt.show()

        return wavelengths, flux_obs, new_errors

def get_synthetic_data(vgrid, linelist, p0, wavelengths):

    fluxes, flux_error_order, original_profile = syn.make_spectrum(vgrid, p0, wavelengths, linelist)
    #fluxes = fluxes+0.1*np.random.randn()
    velocities, profile, profile_errors, alpha, RHS_1, LHS_final, continuum_waves, continuum_flux = LSD.LSD(wavelengths, fluxes, flux_error_order, linelist, 'False', poly_ord)
    #velocities, profile, profile_errors = continuumfit_profile(profile, velocities, profile_errors, 1)  ## NOT IN INITIAL VERSION
    #alpha = np.reshape(alpha, (len(wavelengths)*len(velocities)))
    #alpha = np.array(alpha)
    profile = np.array(profile)
    print(len(profile))
    #print(np.shape(alpha))
    #print(np.shape(profile))
    #inputs = np.concatenate((alpha, profile))

    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.title('Profile from LSD (no continuum correction)')
    plt.savefig('/home/lsd/Documents/original_profile_LSD.png')
    plt.show()
    '''
    profile = profile+1
    plt.figure()
    plt.plot(velocities, profile)

    plt.figure()
    plt.plot(wavelengths, fluxes-1)
    #plt.show()
    '''


    inputs = profile
    '''
    print(profile)
    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.title('Profile from LSD (continuum correction)')
    '''
    return wavelengths, fluxes, flux_error_order, inputs, alpha, RHS_1, LHS_final, velocities, continuum_waves, continuum_flux, original_profile

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
## needs data_spec, RHS_1, alpha, LHS_final all to be global variables.
def model_func(inputs, x):

    mdl1=0
    for i in range(0,len(inputs)):
        mdl1 = mdl1 + (inputs[i]*(x)**(i))
        #mdl1 = mdl1 + (inputs[i]*(x/np.max(x))**(i-k_max))   ###### USING THE MODULATED POLYNOMIAL - COULD BE ISSUE

    R_matrix = (data_spec+1)/mdl1 -1

    RHS_final = np.dot(RHS_1, R_matrix)
    profile = np.dot(LHS_final, RHS_final)

    m_spec = np.dot(alpha, profile)

    mdl = m_spec * mdl1
    mdl = mdl-1

    '''
    plt.figure('adjusted mdl')
    plt.plot(x, y, color='k', marker='.')
    plt.plot(x, mdl)
    plt.show()
    '''
    #error = yerr*mdl1
    #print(error)
    return mdl


def convolve(profile, alpha):
    spectrum = np.dot(alpha, profile)
    return spectrum

## maximum likelihood estimation for a gaussian with offset, o, width, w and height h.
def log_likelihood(theta, x, y, yerr):
    #print('o, w, h: %s'%theta)
    model = model_func(theta, x)
    #sigma2 = yerr ** 2 + model ** 2
    #lnlike = -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    lnlike = -0.5 * np.sum((y - model) ** 2 / yerr**2)
    '''
    print(y)
    print(model)
    print(yerr)
    print(yerr1)
    '''
    #print(lnlike)

    #plt.show()
    return lnlike

## imposes the prior restrictions on the inputs
def log_prior(theta):

    check = 0
    #alpha = inputs[:j_max*k_max]
    #z = theta[:k_max]

    # no constraints

    if check==0:
        return 0.0
    return -np.inf

def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    #print(lp)
    if not np.isfinite(lp):
        #print(theta)
        return -np.inf
    #print('hello')
    final = lp + log_likelihood(theta, x, y, yerr)
    #plt.show()
    #print('final')
    return final


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

        return coeffs

#file = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_e2ds_A.fits'
file = '/home/lsd/Documents/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_e2ds_A.fits'
frame = 0

order = 26

t0 = time.time()

### for real data ###
#fits_file = fits.open(file)
#wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1, velocities = get_data(file, frame, order)

vgrid = np.arange(-25, 25, 0.8)
#p0=[0.36, -0.6, 0, 0, 11.38 , -0.000422450, -0.000000408] ## for synthetic data
#p0=[0.36, -0.6, 0, 0, 1, 0.05, -5, -0.5, -15, 19, -70, 100] ## for synthetic data  #### INITIAL VALUES AFFECTING IT?? THIS DIFFERS FROM OLD VERSION

### for synthetic data ###
## setting up the 8th order polynomial for the synthetic data ####
flux = [0.78, 1.048, 0.85, 1.13, 0.9, 1, 0.72, 1.037]
waves = np. arange(4560, 4640, 10)

#poly_inputs=np.polyfit(waves/np.max(waves), flux, 3)
poly_inputs=np.polyfit(waves, flux, 3)
poly_inputs = poly_inputs[::-1]
p0= [0.36, -0.6, 0, 0]
p0 = np.array(p0)
#poly_inputs = [1, 0, 0, 0]
p0 = np.concatenate((p0, poly_inputs))

wavelengths = np.arange(4575, 4626, 0.01)

### This is the order of polynomial the mcmc will try to fit ###
poly_ord = 3

wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1, RHS_11, LHS_final1, velocities, continuum_waves, continuum_flux, original_profile = get_synthetic_data(vgrid, linelist, p0, wavelengths)


## making alpha, LHS_final, RHS_1 and data_spec global variables
alpha = alpha1
RHS_1 = RHS_11
LHS_final = LHS_final1
data_spec = flux_init
## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
j_max = int(len(flux_init))
k_max = len(initial_inputs)
#k_max = int(len(initial_inputs)/(j_max+1))
print(k_max)

#poly_ord = input('Enter order of polynomial for continuum fit:')


if len(continuum_waves)<3:
    #poly_inputs=continuumfit(flux_init,  wavelength_init/np.max(wavelength_init), flux_error_init, poly_ord)
    poly_inputs=continuumfit(flux_init,  wavelength_init, flux_error_init, poly_ord)
    print('used continuumfit')
else:
    #poly_inputs=np.polyfit(continuum_waves/np.max(wavelength_init), continuum_flux, poly_ord)
    poly_inputs=np.polyfit(continuum_waves, continuum_flux, poly_ord)
    print('used polyfit')

#poly_inputs = [1.1, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001]
#poly_inputs = [1.1, 0.00001, 0.00001, 0.00001]  ## YOU WERE USING CONTINUUM FIT TO GET THE INITIAL VALUES

'''
poly = np.poly1d(poly_inputs)
fit = poly(wavelength_init/np.max(wavelength_init))
print('fit: %s'%fit)
flux_obs = flux_init/fit-1
#new_errors = errors/fit

#print(coeffs)
#print(poly)
fig = plt.figure('fit')
plt.plot(wavelengths, flux_init)
plt.plot(wavelengths, fit)
plt.plot(wavelengths, flux_obs)
plt.scatter(continuum_waves, continuum_flux, color = 'k', s=8)
plt.show()
'''

poly_inputs=poly_inputs[::-1]
print(poly_inputs)
#poly_inputs = [0.01]*poly_ord
poly_inputs = np.array(poly_inputs)
#original_profile = initial_inputs
#initial_inputs = initial_inputs*1.25

## TESTING HOW THE INPOUT PROFILE AFFECTS IT ##
#initial_inputs = profile ## upper adjustment

fixed_profile = initial_inputs
#initial_inputs = np.concatenate((initial_inputs, poly_inputs))
initial_inputs = poly_inputs
#initial_inputs = initial_inputs - 0.1

print('Number of inputs: %s'%len(initial_inputs))
print(initial_inputs)
#for order in range(0, 72):
#wavelength, flux, flux_error = run_emcee(wavelength_init, flux_init, flux_error_init, initial_inputs)
#plt.close('all')
x = wavelength_init
y = flux_init
yerr = flux_error_init

#plt.figure('data')
#plt.plot(x, y, 'k')
#plt.close('all')
#plt.show()

#model_inputs, soln = likelihood_estimate(x, y, yerr, initial_inputs)
#print('model inputs created')
#print(model_inputs)
ndim = len(initial_inputs)
nwalkers= ndim*3
#pos = soln.x + 0.1 * np.random.randn(10000, len(initial_inputs))
rng = np.random.default_rng()
#pos=initial_inputs[:]+rng.normal(0.,0.01,(nwalkers, k_max))
pos = initial_inputs[:]+rng.normal(0.,1,(nwalkers, ndim))
#pos=np.concatenate((initial_inputs[:k_max]+rng.normal(0.,0.01,(nwalkers, k_max)), initial_inputs[k_max:]+rng.normal(0.,0.1,(nwalkers, ndim-k_max))), axis = 1)
print(np.shape(pos)) ### INITIALLY USED 0.001 FOR PROFILE
#pos = initial_inputs + 0.0001 * np.random.randn(10000, len(initial_inputs))

#print(pos)
#print(soln.x)
#pos = initial_inputs + 1e-4 * np.random.randn(32, 2)

print(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, 5000, progress=True, skip_initial_state_check=True);
'''
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["sigma", "h", "mu", "offset_in_y"]
for l in range(4, len(initial_inputs)):
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
dis_no = 300
#av_tau = np.sum(tau)/len(tau)
flat_samples = sampler.get_chain(discard=dis_no, flat = True)
print(flat_samples.shape)

#fig = corner.corner(flat_samples, labels=labels);

plt.figure()
inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    #plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
    mdl = model_func(sample, x)
    plt.plot(x, mdl, "C1", alpha=0.1)
plt.scatter(x, y, color = 'k', marker = '.', label = 'data')
#model = model_func(model_inputs, x)
#plt.plot(x, model)
#plt.plot(velocities, ccf_spec, "k", label="truth")
#plt.legend(fontsize=14)
#plt.xlim(0, 10)
plt.xlabel("wavelengths")
plt.ylabel("flux")
plt.title('mcmc models and data')
plt.savefig('/home/lsd/Documents/mcmc_and_data.png')
#plt.show()

'''
plt.figure('profiles')
plt.scatter(velocities, original_profile, color = 'k' ,marker = '.', label = 'original')
plt.xlabel('velocities km/s')
plt.ylabel('flux')
plt.title('Profile directly from mcmc')

for i in range(len(flat_samples)):
    profile = flat_samples[i, :k_max]
    plt.plot(velocities, profile, color = 'r', alpha = 0.1)
'''

#plt.show()
profile = []
poly_cos = []
poly_ord = -1
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    #if i<ndim-poly_ord-1:
        #profile.append(mcmc[1])
    #else:
    poly_cos.append(mcmc[1])
    #q = np.diff(mcmc)
    #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    #txt = txt.format(mcmc[1], q[0], q[1], labels[i])

profile = fixed_profile

plt.figure('profile directly from mcmc')
plt.plot(velocities, profile, color = 'r', label = 'mcmc')
plt.scatter(velocities, original_profile, color = 'k' ,marker = '.', label = 'original')
plt.xlabel('velocities km/s')
plt.ylabel('flux')
plt.title('Profile directly from mcmc')
plt.legend()
plt.savefig('/home/lsd/Documents/mcmc_profile.png')

plt.figure('continuum fit from mcmc')
plt.plot(x, y+1, color = 'k', label = 'data')
mdl1 =0
print('mcmc')
for i in np.arange(0, len(poly_cos)):
    #mdl1 = mdl1+poly_cos[i]*(x/np.max(x))**(i)
    mdl1 = mdl1+poly_cos[i]*(x)**(i)
    print(poly_cos[i])
plt.plot(x, mdl1, label = 'mcmc continuum fit')
plt.scatter(continuum_waves, continuum_flux, label = 'continuum_points')
mdl =0
print('true')
for i in np.arange(4,len(p0)):
    #mdl = mdl+p0[i]*(x/np.max(x))**(i-4)
    mdl = mdl+p0[i]*(x)**(i-4)
    print(p0[i])
plt.plot(x, mdl, label = 'true continuum fit')
#plt.plot(x, poly_cos[1]*x + poly_cos[0], label = 'mcmc continuum fit')
plt.legend()
plt.title('continuum from mcmc')
plt.xlabel("wavelengths")
plt.ylabel("flux")
plt.savefig('/home/lsd/Documents/mcmc_continuum_fit.png')

fit = mdl1
#true_fit = p0[7]*(x-4600)**3 + p0[6]*(x-4600)**2 + p0[5]*(x-4600) + p0[4]
true_fit = mdl
#fit = poly_cos[1]*wavelength_init + poly_cos[0]
flux_adjusted  = (flux_init+1)/fit-1
flux_error_adjusted = flux_error_init/fit

true_flux = (flux_init+1)/true_fit-1
'''
velocities, profile, profile_errors, alpha, cont_waves, cont_flux = LSD.LSD(wavelength_init, flux_adjusted, flux_error_adjusted, linelist, 'False', 'nan')

contents = fits.PrimaryHDU([velocities, profile, profile_errors])
hdu = fits.HDUList(contents)
hdu.writeto('/home/lsd/Documents/LSD_profile_mcmc_order%s'%order, overwrite = 'True')
hdu.close()

t1 = time.time()

plt.figure('LSD with new continuum')
plt.plot(velocities, profile)
plt.xlabel('velocities km/s')
plt.ylabel('flux')
plt.title('Profile from LSD with continuum correction from mcmc')
plt.savefig('/home/lsd/Documents/new_continuum_LSD.png')
'''
m_flux = convolve(profile, alpha)
residuals = flux_adjusted - m_flux

fig, ax = plt.subplots(2, figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'Forwards Model - Order: %s, seperate continuum fit'%(order), sharex = True)
ax[0].plot(x, true_flux, '--', color = 'orange', label = 'data')
ax[0].plot(x, m_flux, color = 'r', linestyle = '-', label = 'model ')
ax[0].legend()
hline = [0]*len(x)
ax[1].plot(x, residuals, '.', color = 'red')
ax[1].plot(x, hline, linestyle = '--')
#ax[1].set_ylim([-0.3, 0.5])
#plt.savefig('/home/lsd/Documents/forward_model_%s.png'%order)
plt.show()

#print('Time Taken: %s minutes'%((t1-t0)/60))
