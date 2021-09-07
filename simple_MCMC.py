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
        '''
        fig = plt.figure('fit')
        plt.plot(wavelengths, fluxes)
        plt.plot(wavelengths, fit)
        plt.plot(wavelengths, flux_obs)
        plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        plt.show()
        '''
        return wavelengths, flux_obs, new_errors

def get_data(file, frame, order):

    fluxes, wavelengths, flux_error_order = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'masked')
    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux= LSD.LSD(wavelengths, fluxes, flux_error_order, linelist, 'True')
    velocities, profile, profile_errors = continuumfit_profile(profile, velocities, profile_errors, 1)
    #alpha = np.reshape(alpha, (len(wavelengths)*len(velocities)))
    #alpha = np.array(alpha)
    #profile = np.array((profile/mp.max(profile))-1)
    profile = np.array(profile)
    #print(len(profile))
    #print(np.shape(alpha))
    #print(np.shape(profile))
    #inputs = np.concatenate((alpha, profile))

    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.title('Profile from LSD (intitial for mcmc)')
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

    return wavelengths, fluxes-1, flux_error_order, inputs, alpha, velocities, continuum_waves, continuum_flux

def get_synthetic_data(vgrid, linelist, p0, wavelengths):
    fluxes, flux_error_order, original_profile = syn.make_spectrum(vgrid, p0, wavelengths, linelist)
    #fluxes = fluxes+0.01*np.random.randn()
    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux = LSD.LSD(wavelengths, fluxes, flux_error_order, linelist, 'False', poly_ord)
    velocities, profile, profile_errors = continuumfit_profile(profile, velocities, profile_errors, 1)
    #alpha = np.reshape(alpha, (len(wavelengths)*len(velocities)))
    #alpha = np.array(alpha)
    profile = np.array(profile)
    #print(len(profile))
    #print(np.shape(alpha))
    #print(np.shape(profile))
    #inputs = np.concatenate((alpha, profile))

    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.title('Profile from LSD (intitial for mcmc)')
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
    return wavelengths, fluxes, flux_error_order, inputs, alpha, velocities, continuum_waves, continuum_flux, original_profile, profile_errors

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
        mdl1 = mdl1 + (inputs[i]*(x/np.max(x))**(i-k_max))
        #print('y = %s*x**%s'%(inputs[i], (i-(k_max))))
        #print(inputs[i])
        #print(i-(k_max))
        #print(mdl1)
    '''
    plt.figure('poly and non-adjust mdl')
    plt.plot(x, mdl1, 'k')
    plt.plot(x, mdl)
    plt.show()
    '''
    mdl = mdl * mdl1
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

    lnlike = -0.5 * np.sum((y - model) ** 2 / yerr**2 + np.log(yerr**2)+ np.log(2*np.pi))
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
    z = theta[:k_max]
    '''
    print(z)
    plt.figure()
    plt.plot(z)
    plt.show()
    '''
    for i in range(len(theta)):
        if i<k_max: ## must lie in z
            if -3<=theta[i]<=0.5: pass
            else:
                check = 1
                #print(theta[i])
                #print(i)
                #stop = input('hi')
                #plt.figure()
                #plt.plot(z)
                #plt.show()

    '''
    if len(continuum_waves)>3:
        for n in range(len(continuum_waves)):
            mdl1=0
            for i in range(k_max,len(theta)):
                mdl1 = mdl1 + (theta[i]*continuum_waves[n]**(i-(k_max)))
            if mdl1 <= (continuum_flux[n]+1)+0.1 and mdl1 >= (continuum_flux[n]+1)-0.1: pass
            else:check = 1
                #plt.figure()
                #plt.plot(continuum_waves, mdl1)
                #plt.show()
    else:pass
    '''

    if check==0:

        # excluding the continuum points in the profile
        z_cont = []
        for i in range(0, k_max):
            if velocities[i]<-v_min or velocities[i]>v_max:
                z_cont.append(theta[i])
        #print(z_cont)
        #print(velocities)
        z_cont = np.array(z_cont)
        # calcualte gaussian probability for each point in continuum
        p_pent = np.sum((1/np.sqrt(2*np.pi*p_var**2))*np.exp(-0.5*(z_cont/p_var)**2))

        return 0
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

order = 26
poly_ord = 9
t0 = time.time()

### for real data ###
#fits_file = fits.open(file)
#wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1, velocities = get_data(file, frame, order)

vgrid = np.arange(-25, 25, 0.8)
#p0=[0.36, -0.6, 0, 0, 11.38 , -0.000422450, -0.000000408] ## for synthetic data
#p0=[0.36, -0.6, 0, 0, 1, 0.05, -5, -0.5, -15, 19, -70, 100] ## for synthetic data

### for synthetic data ###
## setting up the 8th order polynomial for the synthetic data ####
flux = [0.78, 1.048, 0.85, 1.13, 0.9, 1, 0.72, 1.037]
waves = np. arange(4560, 4640, 10)
poly_inputs=np.polyfit(waves/np.max(waves), flux, poly_ord)
poly_inputs = poly_inputs[::-1]
p0= [0.36, -0.6, 0, 0]
p0 = np.array(p0)
p0 = np.concatenate((p0, poly_inputs))

wavelengths = np.arange(4575, 4626, 0.01)

### This is the order of polynomial the mcmc will try to fit ###
poly_ord = poly_ord

#bye = get_data(file, frame, 32)
wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1, velocities, continuum_waves, continuum_flux, original_profile, profile_errors = get_synthetic_data(vgrid, linelist, p0, wavelengths)

## parameters for working out continuum points in the LSD profile
p_var = np.max(profile_errors)
#p_var = 0.001
v_min = -10
v_max = 10

true_inputs = np.concatenate((original_profile, poly_inputs))

## making alpha a global variable
alpha = alpha1
## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
j_max = int(len(flux_init))
k_max = len(initial_inputs)
#k_max = int(len(initial_inputs)/(j_max+1))
print(k_max)

#poly_ord = input('Enter order of polynomial for continuum fit:')


if len(continuum_waves)<3:
    poly_inputs=continuumfit(flux_init,  wavelength_init/np.max(wavelength_init), flux_error_init, poly_ord)
else: poly_inputs=np.polyfit(continuum_waves/np.max(wavelength_init), continuum_flux, poly_ord)
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
initial_inputs = np.concatenate((initial_inputs, poly_inputs))

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

nll = lambda *args: -log_likelihood(*args)
soln = minimize(nll, initial_inputs, args = (x, y, yerr))
model_inputs = soln.x

mdlmdl = model_func(model_inputs, x)
plt.figure()
plt.plot(mdlmdl)
plt.show()
#print('model inputs created')
#print(model_inputs)
#initial_inputs = true_inputs
ndim = len(initial_inputs)
nwalkers= ndim*3
#pos = soln.x + 0.1 * np.random.randn(10000, len(initial_inputs))
rng = np.random.default_rng()
#pos=model_inputs[:k_max]+rng.normal(-0.01,0.01,(nwalkers, k_max))
#pos=initial_inputs[:]+rng.normal(-0.0001,0.0001,(nwalkers, ndim))
#pos = np.float128(pos)
#pos = list(pos)
#print('variation: 0.1*input')

##for second run
'''
hdu = fits.open('/home/lsd/Documents/first_run_outcome.fits')

continuum_coefs = poly_inputs=np.polyfit(hdu[2].data[0], hdu[2].data[1], poly_ord)
model_inputs = np.concatenate((hdu[0].data, continuum_coefs[::-1]))
'''
vary_amounts = []
pos = []
for i in range(0, ndim):
    if model_inputs[i]==0:pos2 = model_inputs[i]+rng.normal(-0.00001, 0.00001,(nwalkers, ))
    else:
        if i <ndim-poly_ord-1:
            pos2 = model_inputs[i]+rng.normal(-abs(model_inputs[i])*0.1,abs(model_inputs[i])*0.1,(nwalkers, ))
            vary_amounts.append(model_inputs[i]*0.1)
        else:
            pos2 = model_inputs[i]+rng.normal(-abs(model_inputs[i])*0.1,abs(model_inputs[i])*0.1,(nwalkers, ))
            vary_amounts.append(model_inputs[i]*0.1)
    pos.append(pos2)

#print(pos)
print(np.shape(pos))
pos = np.array(pos)
pos = np.transpose(pos)
'''
pos1 = np.array(pos1)
pos = np.transpose(pos)
pos = np.concatenate((pos, pos1))
pos = np.transpose(pos)
print(np.shape(pos))
'''
#pos1 = initial_inputs[:k_max] + rng.normal(-0.01, 0.01, (nwalkers, k_max))
#pos = np.concatenate((np.transpose(pos1), np.transpose(pos)))
#pos = np.transpose(pos)

#pos = initial_inputs + 0.0001 * np.random.randn(10000, len(initial_inputs))

#print(pos)
#print(soln.x)
#pos = initial_inputs + 1e-4 * np.random.randn(32, 2)

print(nwalkers, ndim)
steps_no = 250000 #must be greater than 10000

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
sampler.run_mcmc(pos, steps_no, progress=True);

fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
dis_no = int(np.floor(steps_no-10000))
#dis_no = 500
samples = sampler.get_chain(discard = dis_no)
#labels = ["sigma", "h", "mu", "offset_in_y"]
#for l in range(4, len(model_inputs)):
    #labels.append('poly_ord %s'%(l-3))
for i in range(5):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    #ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

#tau = sampler.get_autocorr_time()
#print("Tau: %s"%tau)

#dis_no = int(input("How many values to discard: "))

#av_tau = np.sum(tau)/len(tau)
flat_samples = sampler.get_chain(discard=dis_no, flat=True)
print(flat_samples.shape)

#fig = corner.corner(flat_samples, labels=labels);

plt.figure()

inds = np.random.randint(len(flat_samples), size=100)
for ind in inds:
    sample = flat_samples[ind]
    #plt.plot(x0, np.dot(np.vander(x0, 2), sample[:2]), "C1", alpha=0.1)
    mdl = model_func(sample, x)
    mdl1 = 0
    for i in np.arange(k_max, len(sample)):
        mdl1 = mdl1+sample[i]*(x/np.max(x))**(i-k_max)
    plt.plot(x, mdl1, "C1", alpha=0.1)
    plt.plot(x, mdl+1, "g", alpha=0.1)
    #print(sample[-1])
plt.scatter(x, y+1, color = 'k', marker = '.', label = 'data')
'''
for sample in flat_samples:
    mdl = model_func(sample, x)
    mdl1 = 0
    for i in np.arange(k_max, len(sample)):
        mdl1 = mdl1+sample[i]*(x/np.max(x))**(i-k_max)
    plt.plot(x, mdl1, color = 'r', alpha = 0.1)
    plt.plot(x, mdl+1, color = 'g', alpha = 0.1)
plt.scatter(x, y, color = 'k', marker = '.')
'''
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

for i in range(ndim):
    mcmc = np.median(flat_samples[:, i])
    if i<ndim-poly_ord-1:
        profile.append(mcmc)
    else:
        poly_cos.append(mcmc)
        #print(mcmc[1], poly_cos)
    #q = np.diff(mcmc)
    #txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
    #txt = txt.format(mcmc[1], q[0], q[1], labels[i])
mdl_1 = 0
for i in range(len(flat_samples)):
    sample = flat_samples[i]
    for j in range(k_max, ndim):
        mdl_1 = mdl_1+sample[j]*(x/np.max(x))**(j-k_max)
av_poly_fit = mdl_1/len(flat_samples)

fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]})
ax[0].plot(velocities, profile, color = 'r', label = 'mcmc')
ax[0].scatter(velocities, original_profile, color = 'k' ,marker = '.', label = 'original')
ax[1].scatter(velocities, original_profile-profile, color = 'r')
#plt.title('Profile directly from mcmc')
ax[0].legend()
plt.savefig('/home/lsd/Documents/mcmc_profile.png')

plt.figure('continuum fit from mcmc')
plt.plot(x, y+1, color = 'k', label = 'data')
mdl1 =0
for i in np.arange(0, len(poly_cos)):
    mdl1 = mdl1+poly_cos[i]*(x/np.max(x))**(i)
plt.plot(x, mdl1, label = 'mcmc continuum fit')
plt.scatter(continuum_waves, continuum_flux, label = 'continuum_points')
mdl =0
for i in np.arange(4,len(p0)):
    mdl = mdl+p0[i]*(x/np.max(x))**(i-4)
plt.plot(x, mdl, label = 'true continuum fit')
plt.plot(x, av_poly_fit, label = 'mcmc average')
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

fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'Forwards Model - Order: %s, seperate continuum fit'%(order), sharex = True)
ax[0].plot(x, true_flux, '--', color = 'orange', label = 'data')
ax[0].plot(x, m_flux, color = 'r', linestyle = '-', label = 'model')
ax[0].legend()
hline = [0]*len(x)
ax[1].plot(x, residuals, '.', color = 'red')
ax[1].plot(x, hline, linestyle = '--')
#ax[1].set_ylim([-0.3, 0.5])
#plt.savefig('/home/lsd/Documents/forward_model_%s.png'%order)
plt.show()


## calculating liklihood for real data:

true_mdl = model_func(true_inputs, x)
true_liklihood = log_probability(true_inputs, x, y, yerr)

## calculating likilihood for mcmc models
mcmc_inputs = np.concatenate((profile, poly_cos))
#mcmc_mdl = ((np.dot(alpha, profile)+1)*av_poly_fit)-1
mcmc_mdl = model_func(mcmc_inputs, x)
#mcmc_inputs = np.concatenate((mcmc_inputs, m))
mcmc_liklihood = log_probability(mcmc_inputs, x, y, yerr)

residuals_2 = (true_mdl+1) - (mcmc_mdl+1)

fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
ax[0].plot(x, true_mdl+1, 'r', alpha = 0.3, label = 'true spec')
ax[0].plot(x, true_fit, 'r', label = 'true continuum fit' )
ax[0].plot(x, mcmc_mdl+1, 'k', alpha =0.3, label = 'mcmc spec')
ax[0].plot(x, fit, 'k', label = 'mcmc continuum fit')
ax[0].legend()
ax[1].plot(x, residuals_2, '.')
plt.show()

print('True likelihood: %s\nMCMC likelihood: %s\n'%(true_liklihood, mcmc_liklihood))

print('reverted version, -0.01, 0.01, 0.001 in synthetic, using likelihood')
#print('Time Taken: %s minutes'%((t1-t0)/60))
