import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import  fits
import emcee
#import corner
import LSD_func_faster as LSD
import time
import synthetic_data as syn
import random

## for real data
fits_file = '/home/lsd/Documents/HD189733/August2007_master_out_ccfs.fits'
linelist = '/home/lsd/Documents/fulllinelist0001.txt'
#linelist = '/home/lsd/Documents/fulllinelist018.txt'
#linelist = '/Users/lucydolan/Starbase/fulllinelist004.txt'
#linelist = '/home/lsd/Documents/fulllinelist004.txt'
directory = '/home/lsd/Documents/HD189733/August2007/'
#directory = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/'

run_name = input('Input nickname for this version of code (for saving figures): ')
### adjusts continuum of the LSD profile to be at 0
## fluxes = profile, wavelengths = velocities - copied from another code so names don't make sense
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

## fits the continuum of the spectum - used to get the initial continuum coefficents
def continuumfit(fluxes1, wavelengths1, errors1, poly_ord):
        fluxes1 = fluxes1+1
        ## taking out masked areas
        idx = [errors1<1]
        print(idx)
        errors = errors1[tuple(idx)]
        fluxes = fluxes1[tuple(idx)]
        wavelengths = wavelengths1[tuple(idx)]

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
        fit = poly(wavelengths1)
        flux_obs = fluxes1/fit-1
        new_errors = errors1/fit

        '''
        fig = plt.figure('fit for initial continuum')
        plt.plot(wavelengths1, fluxes1)
        plt.plot(wavelengths1, fit)
        plt.plot(wavelengths1, flux_obs)
        #plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        plt.show()
        '''

        return coeffs, flux_obs, new_errors

### processes HARPS data file to produce a normalised spectrum (no continuum corection - only blaze correction), the initial LSD profile and the alpha matrix.
#sns = []
#sn_wave = []
def get_data(file, frame, order, poly_ord):

    fluxes, wavelengths, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'unmasked', run_name)
    #sns.append(sn)
    #sn_waves.append(mid_wave_order)
    plt.plot(wavelengths, fluxes)
    plt.show()

    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)
    poly_inputs, fluxes1, flux_error_order1 = continuumfit(fluxes,  (wavelengths*a)+b, flux_error_order, poly_ord)

    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name)
    velocities, profile, profile_errors = continuumfit_profile(profile, velocities, profile_errors, 1)
    profile = np.array(profile)
    #print(profile)
    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('optical depth')
    plt.title('Profile from LSD (intitial for mcmc)')
    plt.savefig('/home/lsd/Documents/LSD_Figures/initial_profiles/order%s_initprof_%s'%(order, run_name))
    #plt.show()

    return wavelengths, fluxes, flux_error_order, profile, alpha, velocities, continuum_waves, continuum_flux

## makes the synthetic spectrum, runs the LSD on it to produce the initial LSD profile
def get_synthetic_data(vgrid, linelist, p0, wavelengths):
    linelist1 = '/home/lsd/Documents/fulllinelist004.txt'
    print('little lines not fit for')
    fluxes, flux_error_order, original_profile = syn.make_spectrum(vgrid, p0, wavelengths, linelist1)

    ## adding noise
    for i in range(len(fluxes)):
        number = random.uniform(-0.1, 0.1)
        #print(number)
        fluxes[i] = fluxes[i]+number

    sn=1
    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux = LSD.LSD(wavelengths, fluxes, flux_error_order, linelist, 'False', poly_ord, sn)
    velocities, profile, profile_errors = continuumfit_profile(profile, velocities, profile_errors, 1)
    profile = np.array(profile)

    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('optical depth')
    plt.title('Profile from LSD (intitial for mcmc)')
    plt.savefig('/home/lsd/Documents/LSD_Figures/initial_profiles/order%s_originalprofsyn_%s'%(order, run_name))
    plt.show()

    return wavelengths, fluxes, flux_error_order, profile, alpha, velocities, continuum_waves, continuum_flux, original_profile, profile_errors

## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
def model_func(inputs, x):
    z = inputs[:k_max]

    mdl = np.dot(alpha, z) ##alpha has been declared a global variable after LSD is run.

    mdl = mdl+1

    ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
    a = 2/(np.max(x)-np.min(x))
    b = 1 - a*np.max(x)

    mdl1=0
    for i in range(k_max,len(inputs)):
        mdl1 = mdl1 + (inputs[i]*((x*a)+b)**(i-k_max))

    mdl = mdl * mdl1
    mdl = mdl-1

    return mdl

def convolve(profile, alpha):
    spectrum = np.dot(alpha, profile)
    return spectrum

## maximum likelihood estimation for the mcmc model.
def log_likelihood(theta, x, y, yerr):
    model = model_func(theta, x)

    lnlike = -0.5 * np.sum(((y+1) - (model+1)) ** 2 / yerr**2 + np.log(yerr**2)+ np.log(2*np.pi))

    return lnlike

## imposes the prior restrictions on the inputs - rejects if profile point if less thna 3 or greater than 1.5.
def log_prior(theta):

    check = 0
    z = theta[:k_max]

    for i in range(len(theta)):
        if i<k_max: ## must lie in z
            if -3<=theta[i]<=0.5: pass
            else:
                check = 1

    if check==0:
        ## penalty function for profile - not in use

        # excluding the continuum points in the profile
        z_cont = []
        v_cont = []
        for i in range(0, 3):
                z_cont.append(theta[len(theta)-i-1])
                v_cont.append(velocities[len(velocities)-i-1])
                z_cont.append(theta[i])
                v_cont.append(velocities[i])
        #print(z_cont)
        #print(velocities)
        z_cont = np.array(z_cont)
        '''
        plt.figure()
        plt.plot(velocities, theta[:k_max])
        plt.scatter(v_cont, z_cont)
        plt.show()
        '''
        # calcualte gaussian probability for each point in continuum
        p_pent = np.sum((1/np.sqrt(2*np.pi*p_var**2))*np.exp(-0.5*(z_cont/p_var)**2))

        return p_pent
        #return 0
    return -np.inf

## calculates log probability - used for mcmc
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

def residual_mask(wavelengths, forward, data_spec, data_err):
    #residuals=((data_spec+1)/(forward+1))/data_err
    residuals=abs(((data_spec+1)-(forward+1))/(data_spec+1))
    #residuals=abs((data_spec+1)-(forward+1))
    #print(residuals)
    #idx = tuple([residuals>=0.2])

    for i in range(len(residuals)):
        if residuals[i]>0.2:
            idx = np.logical_and(wavelengths>=wavelengths[i]-0.25, wavelengths<=wavelengths[i]+0.25)
            data_err[idx]=10000000000000000000

    return data_err


input1 = input('Use synthetic data? y or n: ')

if input1 == 'y':
    vgrid = np.arange(-25, 25, 0.8)
    #p0=[0.36, -0.6, 0, 0, 11.38 , -0.000422450, -0.000000408] ## for synthetic data
    #p0=[0.36, -0.6, 0, 0, 1, 0.05, -5, -0.5, -15, 19, -70, 100] ## for synthetic data
    poly_ord = int(input('Enter order of polynomial for synthetic continuum: '))

    ### for synthetic data ###
    ## setting up the 8th order polynomial for the synthetic data ####
    # creates a fake data set to fit a polynomial to - is just an easy way to get continuum coefficents that make sense for the wavelength range
    flux = [0.98, 1.048, 0.85, 1.03, 0.9, 1, 0.82, 1.037]
    waves = np. arange(4560, 4640, 10)

    ## used for adjusting wavelegnths to between -1 and 1
    a = 2/(np.max(waves)-np.min(waves))
    b = 1 - a*np.max(waves)

    poly_inputs=np.polyfit(((waves*a)+b), flux, poly_ord)
    poly_inputs = poly_inputs[::-1]
    ## the inputs for the gaussian used for profile
    p0= [0.36, -0.6, 0, 0]
    p0 = np.array(p0)
    p0 = np.concatenate((p0, poly_inputs))

    wavelengths = np.arange(4575, 4626, 0.01)
    wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1, velocities, continuum_waves, continuum_flux, original_profile, profile_errors = get_synthetic_data(vgrid, linelist, p0, wavelengths)
    true_inputs = np.concatenate((original_profile, poly_inputs))
    poly_ord = int(input('Enter order of polynomial for mcmc to fit: '))
else:
    #file = input('Enter path to data file: ')
    #frame = int(input('Enter frame: '))
    #order = int(input('Enter order: '))
    #poly_ord = int(input('Enter order of polynomial for mcmc to fit: '))

    file = '/home/lsd/Documents/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_e2ds_A.fits'
    #file = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_e2ds_A.fits'
    frame = 0
    #order = 26
    poly_ord = 3

    #profile_order = []
    #coeffs_order = []
    #for frame in range(, 10):
    #if input1!='y':

P=2.21857567 #Cegla et al, 2006 - days
T=2454279.436714 #Cegla et al, 2006

fits_file = fits.open(file)
phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1

for order in range(26, 27):
    plt.close('all')
    wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1, velocities, line_waves, line_depths = get_data(file, frame, order, poly_ord)

    print(order)

    t0 = time.time()

    ## parameters for working out continuum points in the LSD profile - if using penalty function
    p_var = 0.001
    v_min = -10
    v_max = 10

    ## making alpha a global variable
    alpha = alpha1
    ## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
    j_max = int(len(flux_init))
    k_max = len(initial_inputs)

    ## getting initial continuum coefficents
    a = 2/(np.max(wavelength_init)-np.min(wavelength_init))
    b = 1 - a*np.max(wavelength_init)

    poly_inputs, bin, bye=continuumfit(flux_init,  (wavelength_init*a)+b, flux_error_init, poly_ord)
    poly_inputs=poly_inputs[::-1]
    poly_inputs = np.array(poly_inputs)
    model_inputs = np.concatenate((initial_inputs, poly_inputs))

    ## setting x, y, yerr for emcee
    x = wavelength_init
    y = flux_init
    yerr = flux_error_init

    ## making initial model and masking areas with large residuals
    initial_mdl = model_func(model_inputs, x)
    #yerr = residual_mask(x, initial_mdl, y, yerr1)
    #print(yerr)

    ## setting these normalisation factors as global variables - used in the figures below
    a = 2/(np.max(x)-np.min(x))
    b = 1 - a*np.max(x)

    ## setting number of walkers and their start values(pos)
    ndim = len(model_inputs)
    nwalkers= ndim*3
    rng = np.random.default_rng()

    ## starting values of walkers vary from the model_inputs by 0.01*model_input - this means the bottom of the profile varies more than the continuum. Continuum coefficent vary by 1*model_input.
    vary_amounts = []
    pos = []
    for i in range(0, ndim):
        if model_inputs[i]==0:pos2 = model_inputs[i]+rng.normal(-0.0001, 0.0001,(nwalkers, ))
        else:
            if i <ndim-poly_ord-1:
                #pos2 = model_inputs[i]+rng.normal(-0.001, 0.001,(nwalkers, ))
                pos2 = model_inputs[i]+rng.normal(-abs(model_inputs[i])*0.01,abs(model_inputs[i])*0.01,(nwalkers, ))
                vary_amounts.append(abs(model_inputs[i])*0.01)
            else:
                pos2 = model_inputs[i]+rng.normal(-abs(model_inputs[i])*1,abs(model_inputs[i])*1,(nwalkers, ))
                vary_amounts.append(model_inputs[i]*1)
        pos.append(pos2)

    pos = np.array(pos)
    pos = np.transpose(pos)

    ## the number of steps is how long it runs for - if it doesn't look like it's settling at a value try increasing the number of steps
    steps_no = 10000

    # running the mcmc using python package emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, steps_no, progress=True);

    idx = tuple([yerr<10])

    x_nomask = x
    y_nomask = y
    yerr_nomask = yerr

    #x = x_nomask[idx]
    #y = y_nomask[idx]
    #yerr = yerr_nomask[idx]

    #x = np.array(x_masked)
    #y = np.array(y_masked)
    #yerr = np.array(yerr_masked)

    t1 = time.time()

    ## discarding all vales except the last 1000 steps.
    dis_no = int(np.floor(steps_no-5000))


    # plots the model for 'walks' of the all walkers for the first 5 profile points
    fig, axes = plt.subplots(5, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain(discard = dis_no)
    for i in range(5):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
    axes[-1].set_xlabel("step number");


    ## combining all walkers togather
    flat_samples = sampler.get_chain(discard=dis_no, flat=True)

    # plots random models from flat_samples - lets you see if it's converging
    plt.figure()
    inds = np.random.randint(len(flat_samples), size=100)
    for ind in inds:
        sample = flat_samples[ind]
        mdl = model_func(sample, x_nomask)
        #mdl = model_func(sample, x)
        #mdl = mdl[idx]
        mdl1 = 0
        for i in np.arange(k_max, len(sample)):
            mdl1 = mdl1+sample[i]*((a*x)+b)**(i-k_max)
        plt.plot(x, mdl1, "C1", alpha=0.1)
        plt.plot(x, mdl+1, "g", alpha=0.1)
    plt.scatter(x, y+1, color = 'k', marker = '.', label = 'data')
    plt.xlabel("wavelengths")
    plt.ylabel("optical depth")
    plt.title('mcmc models and data')
    #plt.savefig('/home/lsd/Documents/mcmc_and_data.png')
    plt.savefig('/home/lsd/Documents/LSD_Figures/mc_mdl/order%s_mc_mdl_%s'%(order, run_name))
    #plt.show()


    ## getting the final profile and continuum values - median of last 1000 steps
    profile = []
    poly_cos = []
    profile_err = []
    poly_cos_err = []

    for i in range(ndim):
        mcmc = np.median(flat_samples[:, i])
        error = np.std(flat_samples[:, i])
        if i<ndim-poly_ord-1:
            profile.append(mcmc)
            profile_err.append(error)
        else:
            poly_cos.append(mcmc)
            poly_cos_err.append(error)

    profile = np.array(profile)
    profile_err = np.array(profile_err)
    # plots the mcmc profile - will have extra panel if it's for data

    if input1 == 'y':
        fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]})
        ax[0].plot(velocities, profile, color = 'r', label = 'mcmc')
        ax[0].scatter(velocities, original_profile, color = 'k' ,marker = '.', label = 'original')
        ax[1].scatter(velocities, original_profile-profile, color = 'r')
        ax[0].legend()
    else:
        plt.figure()
        plt.plot(velocities, profile, color = 'r', label = 'mcmc')
        zero_line = [0]*len(velocities)
        plt.plot(velocities, zero_line)
        plt.plot(velocities, initial_inputs[:k_max], label = 'initial')
        plt.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
        plt.legend()
    plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/order%s_profile_%s'%(order, run_name))

    # plots mcmc continuum fit on top of data
    plt.figure('continuum fit from mcmc')
    plt.plot(x, y+1, color = 'k', label = 'data')
    mdl1 =0
    for i in np.arange(0, len(poly_cos)):
        mdl1 = mdl1+poly_cos[i]*((a*x)+b)**(i)
    plt.plot(x, mdl1, label = 'mcmc continuum fit')
    mdl1_poserr =0
    for i in np.arange(0, len(poly_cos)):
        mdl1_poserr = mdl1_poserr+(poly_cos[i]+poly_cos_err[i])*((a*x)+b)**(i)
    mdl1_neg =0
    for i in np.arange(0, len(poly_cos)):
        mdl1_neg = mdl1_neg+(poly_cos[i]-poly_cos_err[i])*((a*x)+b)**(i)
    plt.fill_between(x, mdl1_neg, mdl1_poserr, alpha = 0.3)

    #plt.scatter(continuum_waves, continuum_flux, label = 'continuum_points')
    if input1 == 'y':
        mdl =0
        for i in np.arange(4,len(p0)):
            mdl = mdl+p0[i]*((a*x)+b)**(i-4)
        plt.plot(x, mdl, label = 'true continuum fit')
    plt.legend()
    plt.title('continuum from mcmc')
    plt.xlabel("wavelengths")
    plt.ylabel("optical depth")
    plt.savefig('/home/lsd/Documents/LSD_Figures/continuum_fit/order%s_cont_%s'%(order, run_name))


    ## last section is a bit of a mess but plots the two forward models

    #fit = mdl1
    #flux_adjusted  = (flux_init+1)/fit-1
    #flux_error_adjusted = flux_error_init/fit
    ## calculating likilihood for mcmc models
    mcmc_inputs = np.concatenate((profile, poly_cos))
    mcmc_mdl = model_func(mcmc_inputs, x_nomask)
    #mcmc_mdl = mcmc_mdl[idx]
    mcmc_liklihood = log_probability(mcmc_inputs, x_nomask, y_nomask, yerr_nomask)

    print('Likelihood for mcmc: %s'%mcmc_liklihood)

    residuals_2 = (y+1) - (mcmc_mdl+1)

    fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
    non_masked = tuple([yerr<10])
    #ax[0].plot(x, y+1, color = 'r', alpha = 0.3, label = 'data')
    #ax[0].plot(x[non_masked], mcmc_mdl[non_masked]+1, color = 'k', alpha = 0.3, label = 'mcmc spec')
    ax[1].scatter(x[non_masked], residuals_2[non_masked], marker = '.')
    ax[0].plot(x, y+1, 'r', alpha = 0.3, label = 'data')
    ax[0].plot(x, mcmc_mdl+1, 'k', alpha =0.3, label = 'mcmc spec')
    residual_masks = tuple([yerr>10])
    '''
    for mask_pos in x[residual_masks]:
        ax[0].plot([mask_pos]*len(y), y+1, alpha = 0.3, color = 'w')
    '''
    #residual_masks = tuple([yerr>10])
    ax[0].scatter(x[residual_masks], y[residual_masks]+1, label = 'masked', color = 'b', alpha = 0.3)
    ax[0].legend(loc = 'lower right')
    #ax[0].set_ylim(0, 1)
    plotdepths = 1-np.array(line_depths)
    ax[0].vlines(line_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
    #ax[1].plot(x, residuals_2, '.')
    #ax[1].scatter(x[residual_masks], residuals_2[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    z_line = [0]*len(x)
    ax[1].plot(x, z_line, '--')
    plt.savefig('/home/lsd/Documents/LSD_Figures/forward_models/order%s_forward_%s'%(order, run_name))

    ## plots forward models for continuum corrected data and uncorrected data - only if using synthetic
    if input1 == 'y':
        true_fit = mdl
        true_flux = (flux_init+1)/true_fit-1

        m_flux = convolve(profile, alpha)
        residuals = flux_adjusted - m_flux

        fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'Forwards Model', sharex = True)
        ax[0].plot(x, true_flux, '--', color = 'orange', label = 'data')
        ax[0].plot(x, m_flux, color = 'r', linestyle = '-', label = 'model')
        ax[0].legend()
        hline = [0]*len(x)
        ax[1].plot(x, residuals, '.', color = 'red')
        ax[1].plot(x, hline, linestyle = '--')

        ## calculating liklihood for real data:
        true_mdl = model_func(true_inputs, x)
        true_liklihood = log_probability(true_inputs, x, y, yerr)

        residuals_2 = (true_mdl+1) - (mcmc_mdl+1)

        fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
        ax[0].plot(x, true_mdl+1, 'r', alpha = 0.3, label = 'true spec')
        ax[0].plot(x, true_fit, 'r', label = 'true continuum fit' )
        ax[0].plot(x, mcmc_mdl+1, 'k', alpha =0.3, label = 'mcmc spec')
        ax[0].plot(x, fit, 'k', label = 'mcmc continuum fit')
        ax[0].legend()
        ax[1].plot(x, residuals_2, '.')
        plt.savefig('/home/lsd/Documents/LSD_Figures/forward_models/order%s_forwardsyn_%s'%(order, run_name))


    print('Profile: %s\nContinuum Coeffs: %s\n'%(profile, poly_cos))
    #print('True likelihood: %s\nMCMC likelihood: %s\n'%(true_liklihood, mcmc_liklihood))
    #profile_order.append(profile)
    #coeffs_order.append(poly_cos)
    print('Time Taken: %s minutes'%((t1-t0)/60))

## asks before showing all the figures
input2 = input('View figures? y or n: ')
if input2 == 'y':
    plt.show()
else:
    plt.close('all')
