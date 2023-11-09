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
import glob

## for real data

## NEED TO CHNAGE FILE PATHS HERE AND LINE 832

fits_file = '/home/lsd/Documents/HD189733/August2007_master_out_ccfs.fits'
file_type = 'e2ds'
linelist = '/home/lsd/Documents/fulllinelist0001.txt'
#linelist = '/home/lsd/Documents/fulllinelist018.txt'
#linelist = '/Users/lucydolan/Starbase/fulllinelist004.txt'
#linelist = '/home/lsd/Documents/fulllinelist004.txt'
directory = '/home/lsd/Documents/HD189733/August2007/'
#directory = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/'
month = 'August2007'

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
        fig = plt.figure('Polynomial fit to profile - brings the continuum to zero')
        plt.title('Polynomial fit to profile - brings the continuum to zero')
        plt.plot(wavelengths, fluxes-1, label = 'orignial profile')
        plt.plot(wavelengths, fit-1, label = '1st order polynomial fit')
        plt.plot(wavelengths, flux_obs, label = 'adjusted profile')
        #plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        plt.xlabel('velocities, km/s')
        plt.ylabel('optical depth')
        plt.legend()
        #plt.show()
        '''

        return wavelengths, flux_obs, new_errors

## fits the continuum of the spectum - used to get the initial continuum coefficents
def continuumfit(fluxes1, wavelengths1, errors1, poly_ord):
        fluxes1 = fluxes1
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
        flux_obs = fluxes1/fit
        new_errors = errors1/fit
        '''
        fig = plt.figure('Continuum fit')
        plt.title('Continuum fit')
        plt.plot(wavelengths1, fluxes1, label = 'original')
        plt.plot(wavelengths1, fit, label = 'fit')
        plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        plt.ylabel('flux')
        plt.xlabel('wavelengths')
        plt.legend()

        fig = plt.figure('Continuum fit - adjusted')
        plt.title('Continuum fit')
        plt.plot(wavelengths1, flux_obs, label = 'adjusted')
        plt.ylabel('flux')
        plt.xlabel('wavelengths')
        plt.legend()
        #plt.show()
        '''
        coeffs=coeffs[::-1]
        coeffs = np.array(coeffs)

        if len(flux_obs[flux_obs<0])>0:
            return coeffs, fluxes1, errors1, fit

        return coeffs, flux_obs, new_errors, fit

def od2flux(x):
    return np.exp(x)-1

def flux2od(x):
    return np.log(x+1)

### processes HARPS data file to produce a normalised spectrum (no continuum corection - only blaze correction), the initial LSD profile and the alpha matrix.
def get_data(file, frame, order, poly_ord):

    fluxes, wavelengths, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'unmasked', run_name)
    if len(flux_error_order[flux_error_order<1])<5:
        print('discarded')
        return [], [], [], [], [], [], [], [], [], []

    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)
    poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes,  (wavelengths*a)+b, flux_error_order, poly_ord)

    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name)

    print('NUMBER OF LINES: %s'%no_line)
    '''
    plt.figure('First LSD profile from normalised, adjusted spectrum')
    plt.title('First LSD profile from normalised, adjusted spectrum')
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('optical depth')
    plt.show()

    plt.figure('First LSD profile from normalised, adjusted spectrum (flux)')
    plt.title('First LSD profile from normalised, adjusted spectrum (flux)')
    plt.plot(velocities, np.exp(profile)-1)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.show()
    '''

    velocities, profile, profile_errors = continuumfit_profile(profile, velocities, profile_errors, 1)
    profile = np.array(profile)

    '''
    fig, ax0 = plt.subplots()
    zero_line = [0]*len(velocities)
    ax0.plot(velocities, zero_line)
    ax0.plot(velocities, profile, label = 'initial')
    #ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
    ax0.set_xlabel('velocities')
    ax0.set_ylabel('optical depth')
    secax = ax0.secondary_yaxis('right', functions = (od2flux, flux2od))
    secax.set_ylabel('flux')
    #ax1 = ax0.twinx()
    #ax1.plot(velocities, prof_flux)
    #ax1.set_ylabel('flux')
    ax0.legend()
    '''
    '''
    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('optical depth')
    plt.title('Profile from LSD (intitial for mcmc)')
    '''
    #plt.savefig('/home/lsd/Documents/LSD_Figures/initial_profiles/order%s_initprof_%s'%(order, run_name))
    #plt.show()

    return wavelengths, fluxes, flux_error_order, profile, alpha, velocities, continuum_waves, continuum_flux, sn, no_line

## makes the synthetic spectrum, runs the LSD on it to produce the initial LSD profile
def get_synthetic_data(vgrid, linelist, p0, wavelengths):
    linelist1 = '/home/lsd/Documents/fulllinelist004.txt'
    fluxes, flux_error_order, original_profile = syn.make_spectrum(vgrid, p0, wavelengths, linelist1)

    ## adding noise
    for i in range(len(fluxes)):
        number = random.uniform(-0.1, 0.1)
        #print(number)
        fluxes[i] = fluxes[i]+number

    sn=1
    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line = LSD.LSD(wavelengths, fluxes, flux_error_order, linelist, 'False', poly_ord, sn, 20, run_name)
    #velocities, profile, profile_errors = continuumfit_profile(profile, velocities, profile_errors, 1)
    profile = np.array(profile)

    plt.figure()
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('optical depth')
    plt.title('Profile from LSD (intitial for mcmc)')
    plt.savefig('/home/lsd/Documents/LSD_Figures/initial_profiles/order%s_originalprofsyn_%s'%(20, run_name))
    #plt.show()

    return wavelengths, fluxes, flux_error_order, profile, alpha, velocities, continuum_waves, continuum_flux, original_profile, profile_errors

## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
def model_func(inputs, x):
    z = inputs[:k_max]

    mdl = np.dot(alpha, z) ##alpha has been declared a global variable after LSD is run.

    #converting model from optical depth to flux
    mdl = np.exp(mdl)

    #mdl = mdl +1

    ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
    a = 2/(np.max(x)-np.min(x))
    b = 1 - a*np.max(x)

    mdl1=0
    for i in range(k_max,len(inputs)):
        mdl1 = mdl1 + (inputs[i]*((x*a)+b)**(i-k_max))

    mdl = mdl * mdl1

    return mdl

def convolve(profile, alpha):
    spectrum = np.dot(alpha, profile)
    return spectrum

## maximum likelihood estimation for the mcmc model.
def log_likelihood(theta, x, y, yerr):
    model = model_func(theta, x)

    #lnlike = -0.5 * np.sum(((y) - (model)) ** 2 / yerr**2) - np.sum(np.log(yerr**2)) - len(y)*np.log(2*np.pi)
    lnlike = -0.5 * np.sum(((y) - (model)) ** 2 / yerr**2 + np.log(yerr**2)+ np.log(2*np.pi))

    return lnlike

## imposes the prior restrictions on the inputs - rejects if profile point if less thna 3 or greater than 1.5.
def log_prior(theta):

    check = 0
    z = theta[:k_max]


    for i in range(len(theta)):
        if i<k_max: ## must lie in z
            if -10<=theta[i]<=0.5: pass
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

## iterative residual masking - mask continuous areas first - then possibly progress to masking the narrow lines
def residual_mask(wavelengths, data_spec, data_err):

    poly_inputs, bin, bye, fit=continuumfit(data_spec,  (wavelengths*a)+b, data_err, poly_ord)
    forward = model_func(np.concatenate((initial_inputs, poly_inputs)), wavelengths)

    residuals=(data_spec+1)-(forward+1)

    flag = ['False']*len(residuals)
    flagged = []
    for i in range(len(residuals)):
        if abs(residuals[i])>0.25:
            flag[i] = 'True'
            if i>0 and flag[i-1] == 'True':
                flagged.append(i-1)
        else:
            if len(flagged)>0:
                flagged.append(i-1)
                if len(flagged)<20:
                    for no in flagged:
                        flag[no] = 'False'
                else:
                    idx = np.logical_and(wavelengths>=wavelengths[np.min(flagged)]-1, wavelengths<=wavelengths[np.max(flagged)]+1)
                    data_err[idx]=10000000000000000000
            flagged = []

    '''
    fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, sharex = True)
    non_masked = tuple([data_err<10])
    ax[1].scatter(wavelengths[non_masked], residuals[non_masked], marker = '.')
    ax[0].plot(wavelengths, data_spec, 'r', alpha = 0.3, label = 'data')
    ax[0].plot(wavelengths, forward, 'k', alpha =0.3, label = 'mcmc mdl')
    residual_masks = tuple([yerr>10])

    ax[0].scatter(wavelengths[residual_masks], data_spec[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    ax[0].legend(loc = 'lower right')
    plotdepths = -np.array(line_depths)
    #ax[0].vlines(line_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
    #ax[1].plot(x, residuals_2, '.')
    #ax[1].scatter(x[residual_masks], residuals_2[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    z_line = [0]*len(x)
    ax[1].plot(x, z_line, '--')

    plt.figure('Maksing based off residuals (>0.25 for >20 masked)')
    plt.plot(wavelengths, data_spec)
    residual_masks = tuple([data_err>10])
    plt.scatter(wavelengths[residual_masks], data_spec[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    plt.xlabel('wavelengths')
    plt.ylabel('flux')
    plt.show()
    '''
    poly_inputs, bin, bye, fit=continuumfit(data_spec,  (wavelengths*a)+b, data_err, poly_ord)
    velocities, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, bin, bye, linelist, 'False', poly_ord, sn, order, run_name)
    '''
    plt.figure('LSD profile - intital for mcmc')
    plt.title('LSD profile - intital for mcmc')
    plt.plot(velocities, profile)
    plt.xlabel('velocities km/s')
    plt.ylabel('optical depth')
    plt.show()

    plt.figure('LSD profile - intital for mcmc (flux)')
    plt.title('LSD profile - intital for mcmc (flux)')
    plt.plot(velocities, np.exp(profile)-1)
    plt.xlabel('velocities km/s')
    plt.ylabel('flux')
    plt.show()
    '''
    ## second round of masking - based of new profile - excluded as it made profile noiser due to reduction in number of lines
    '''
    forward = model_func(np.concatenate((profile, poly_inputs)), wavelengths)

    #need to re define continuum fit
    plt.figure('profile')
    plt.plot(velocities, profile)
    #plt.show()
    plt.figure()
    plt.plot(wavelengths, forward)
    plt.plot(wavelengths, data_spec)
    #plt.show()

    residuals=abs((data_spec+1)-(forward+1))

    plt.figure('residuals - after 1st resi mask')
    plt.scatter(x, residuals)
    #plt.show()

    limit=abs(max(velocities))*max(continuum_waves)/2.99792458e5

    flag = ['False']*len(residuals)
    for j in range(0, len(wavelengths)):
        flagged = []
        line_waves = []
        for i in (range(0,len(continuum_waves))):
            diff=wavelengths[j]-continuum_waves[i]
            if abs(diff)<=(limit):
                line_waves.append(wavelengths[j])
                if (residuals[j]/abs(continuum_flux[i]))>=10:
                    #print(residuals[j])
                    #print(continuum_flux[i])
                    flagged.append(j)
        if len(flagged)>len(line_waves)/2:
            #print(i)
            #print(len(flagged), len(line_waves)/2)
            idx = np.logical_and(wavelengths>=wavelengths[np.min(flagged)], wavelengths<=wavelengths[np.max(flagged)])
            data_err[idx]=10000000000000000000


    line_masked = 0
    for i in range(k_max):
        data_points = alpha[:, k]
        print(data_points)
        idx = tuple([data_points>0.1])
        resi_points = residuals[idx]
        if len(resi_points[resi_points>0.2])>=(len(resi_points)/2):
            data_err[idx]=10000000000000000000
            line_masked+=1
            plt.figure('line masking')
            plt.plot(wavelengths, data_spec)
            plt.scatter(x[idx], y[idx], label = 'masked', color = 'b', alpha = 0.3)
            plt.show()

    poly_inputs, bin, bye, fit=continuumfit(data_spec,  (wavelengths*a)+b, data_err, poly_ord)
    velocities, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(x, bin, bye, linelist, 'False', poly_ord, sn, order, run_name)
    #poly_inputs = [0.1]*(poly_ord+1)

    plt.figure()
    plt.plot(wavelengths, data_spec)
    residual_masks = tuple([data_err>10])
    plt.scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)

    plt.figure()
    plt.scatter(x, residuals)
    plt.plot(x, [0]*len(x))

    plt.figure('profile - after second masking')
    plt.plot(velocities, profile)
    plt.show()
    '''
    return data_err, np.concatenate((profile, poly_inputs))

def findfiles(directory, file_type):

    filelist1=glob.glob('%s/*/*%s**A_corrected*.fits'%(directory, file_type))    #finding corrected spectra
    filelist=glob.glob('%s/*/*%s**A*.fits'%(directory, file_type))               #finding all A band spectra

    filelist_final=[]

    for file in filelist:                                                        #filtering out corrected spectra
        count = 0
        for file1 in filelist1:
            if file1 == file:count=1
        if count==0:filelist_final.append(file)

    return filelist_final


input1 = input('Use synthetic data? y or n: ')

if input1 == 'y':
    vgrid = np.arange(-25, 25, 0.8)
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
    filelist=findfiles(directory, file_type)
    frame_list = [0] #set to only do one frame
    poly_ord = 3


P=2.21857567 #Cegla et al, 2006 - days
T=2454279.436714 #Cegla et al, 2006
t=0.076125 #Torres et al, 2008
deltaphi = t/(2*P)

sns=[]
no_lines = []

for frame in frame_list:
    file = filelist[frame]
    fits_file = fits.open(file)
    phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1

    hdu = fits.HDUList()
    hdr = fits.Header()
    for order in range(8, 9):
        print(order)
        plt.close('all')
        wavelength_init, flux_init, flux_error_init, initial_inputs, alpha1, velocities, line_waves, line_depths, sn, no_line = get_data(file, frame, order, poly_ord)
        sns.append(sn)

        if len(wavelength_init)==0:
            print('order %s discarded'%order)
            continue


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

        ## setting x, y, yerr for emcee
        x = wavelength_init
        y = flux_init
        yerr = flux_error_init

        ## setting these normalisation factors as global variables - used in the figures below
        a = 2/(np.max(x)-np.min(x))
        b = 1 - a*np.max(x)

        ## making initial model and masking areas with large residuals
        ## initial continuum fit done in this function
        yerr, model_inputs = residual_mask(x, y, yerr)

        if (len(yerr[yerr>1]))>=(len(yerr)/2):
            print('order %s discarded'%order)
            continue

        ## setting number of walkers and their start values(pos)
        ndim = len(model_inputs)
        nwalkers= ndim*3
        rng = np.random.default_rng()

        ## starting values of walkers vary from the model_inputs by 0.01*model_input - this means the bottom of the profile varies more than the continuum. Continuum coefficent vary by 1*model_input.
        vary_amounts = []
        pos = []
        for i in range(0, ndim):
            if model_inputs[i]==0:pos2 = model_inputs[i]+rng.normal(model_input[i], 0.0001,(nwalkers, ))
            else:
                if i <ndim-poly_ord-1:
                    #pos2 = model_inputs[i]+rng.normal(-0.001, 0.001,(nwalkers, ))
                    pos2 = model_inputs[i]+rng.normal(model_inputs[i], abs(model_inputs[i])*0.02,(nwalkers, ))
                    vary_amounts.append(abs(model_inputs[i])*0.02)
                else:
                    pos2 = model_inputs[i]+rng.normal(model_inputs[i],abs(model_inputs[i])*2,(nwalkers, ))
                    vary_amounts.append(model_inputs[i]*2)
            pos.append(pos2)

        pos = np.array(pos)
        pos = np.transpose(pos)

        ## the number of steps is how long it runs for - if it doesn't look like it's settling at a value try increasing the number of steps
        steps_no = 10000

        # check data before starting emcee fit
        plt.figure('Spectrum given to mcmc with errors')
        plt.title('Spectrum given to mcmc with errors')
        plt.errorbar(x, y, yerr=yerr)
        plt.ylabel('flux')
        plt.xlabel('wavelength')
        plt.show()

        # running the mcmc using python package emcee
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
        sampler.run_mcmc(pos, steps_no, progress=True);

        idx = tuple([yerr<10])

        t1 = time.time()

        ## discarding all vales except the last 1000 steps.
        dis_no = int(np.floor(steps_no-5000))

        # plots the model for 'walks' of the all walkers for the first 5 profile points
        samples = sampler.get_chain(discard = dis_no)
        fig, axes = plt.subplots(len(samples[0, 0, :]), figsize=(10, 7), sharex=True)
        for i in range(len(samples[0, 0, :])):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
        axes[-1].set_xlabel("step number");
        #plt.show()

        ## combining all walkers togather
        flat_samples = sampler.get_chain(discard=dis_no, flat=True)

        # plots random models from flat_samples - lets you see if it's converging
        plt.figure()
        inds = np.random.randint(len(flat_samples), size=100)
        for ind in inds:
            sample = flat_samples[ind]
            mdl = model_func(sample, x)
            mdl1 = 0
            for i in np.arange(k_max, len(sample)):
                mdl1 = mdl1+sample[i]*((a*x)+b)**(i-k_max)
            plt.plot(x, mdl1, "C1", alpha=0.1)
            plt.plot(x, mdl, "g", alpha=0.1)
        plt.scatter(x, y, color = 'k', marker = '.', label = 'data')
        plt.xlabel("wavelengths")
        plt.ylabel("flux")
        plt.title('mcmc models and data')
        #plt.savefig('/home/lsd/Documents/mcmc_and_data.png')
        #plt.savefig('/home/lsd/Documents/LSD_Figures/mc_mdl/order%s_mc_mdl_%s'%(order, run_name))
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

        prof_flux = np.exp(profile)-1
        # plots the mcmc profile - will have extra panel if it's for data

        if input1 == 'y':
            fig, ax = plt.subplots(2, gridspec_kw={'height_ratios': [4, 1]})
            ax[0].plot(velocities, profile, color = 'r', label = 'mcmc')
            ax[0].scatter(velocities, original_profile, color = 'k' ,marker = '.', label = 'original')
            ax[0].set_xlabel('velocities')
            ax[0].set_ylabel('optical depth')
            ax[1].scatter(velocities, original_profile-profile, color = 'r')
            ax[1].set_xlabel('velocities')
            ax[0].legend()
        else:
            fig, ax0 = plt.subplots()
            ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
            zero_line = [0]*len(velocities)
            ax0.plot(velocities, zero_line)
            ax0.plot(velocities, initial_inputs[:k_max], label = 'initial')
            ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
            ax0.set_xlabel('velocities')
            ax0.set_ylabel('optical depth')
            secax = ax0.secondary_yaxis('right', functions = (od2flux, flux2od))
            secax.set_ylabel('flux')
            ax0.legend()
        #plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/order%s_profile_%s'%(order, run_name))
        #plt.show()

        # plots mcmc continuum fit on top of data
        plt.figure('continuum fit from mcmc')
        plt.plot(x, y, color = 'k', label = 'data')
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

        mdl1_err =abs(mdl1-mdl1_neg)
        #plt.scatter(continuum_waves, continuum_flux, label = 'continuum_points')
        if input1 == 'y':
            mdl =0
            for i in np.arange(4,len(p0)):
                mdl = mdl+p0[i]*((a*x)+b)**(i-4)
            plt.plot(x, mdl, label = 'true continuum fit')
        plt.legend()
        plt.title('continuum from mcmc')
        plt.xlabel("wavelengths")
        plt.ylabel("flux")
        #plt.savefig('/home/lsd/Documents/LSD_Figures/continuum_fit/order%s_cont_%s'%(order, run_name))
        #plt.show()

        ## last section is a bit of a mess but plots the two forward models
        ## calculating likilihood for mcmc models
        mcmc_inputs = np.concatenate((profile, poly_cos))
        mcmc_mdl = model_func(mcmc_inputs, x)
        #mcmc_mdl = mcmc_mdl[idx]
        mcmc_liklihood = log_probability(mcmc_inputs, x, y, yerr)

        print('Likelihood for mcmc: %s'%mcmc_liklihood)

        residuals_2 = (y+1) - (mcmc_mdl+1)

        fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
        non_masked = tuple([yerr<10])
        ax[1].scatter(x[non_masked], residuals_2[non_masked], marker = '.')
        ax[0].plot(x, y, 'r', alpha = 0.3, label = 'data')
        ax[0].plot(x, mcmc_mdl, 'k', alpha =0.3, label = 'mcmc spec')
        residual_masks = tuple([yerr>10])

        ax[0].scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        ax[0].legend(loc = 'lower right')
        plotdepths = -np.array(line_depths)
        z_line = [0]*len(x)
        ax[1].plot(x, z_line, '--')
        #plt.savefig('/home/lsd/Documents/LSD_Figures/forward_models/order%s_forward_%s'%(order, run_name))
        #plt.show()

        ## plots forward models for continuum corrected data and uncorrected data - only if using synthetic
        if input1 == 'y':
            true_fit = mdl
            true_flux = (flux_init)/true_fit

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
            ax[0].plot(x, true_mdl, 'r', alpha = 0.3, label = 'true spec')
            ax[0].plot(x, true_fit, 'r', label = 'true continuum fit' )
            ax[0].plot(x, mcmc_mdl, 'k', alpha =0.3, label = 'mcmc spec')
            ax[0].plot(x, fit, 'k', label = 'mcmc continuum fit')
            ax[0].legend()
            ax[1].plot(x, residuals_2, '.')
            #plt.savefig('/home/lsd/Documents/LSD_Figures/forward_models/order%s_forwardsyn_%s'%(order, run_name))


        adjusted_spec_final = y/mdl1
        adjusted_err_final = yerr/mdl1

        plt.figure('adjusted spectrum')
        plt.errorbar(x, adjusted_spec_final, yerr = adjusted_err_final, ecolor = 'k')
        #plt.show()

        velocities, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(x, adjusted_spec_final, adjusted_err_final, linelist, 'False', poly_ord, sn, order, run_name)
        print(no_line)
        no_lines.append(no_line)

        fig, ax0 = plt.subplots()
        ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
        zero_line = [0]*len(velocities)
        ax0.plot(velocities, zero_line)
        ax0.plot(velocities, initial_inputs[:k_max], label = 'initial')
        ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
        ax0.set_xlabel('velocities')
        ax0.set_ylabel('optical depth')
        ax0.legend()
        #plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/order%s_final_profile_%s'%(order, run_name))
        #plt.show()

        profile_f = np.exp(profile)
        profile_errors_f = np.sqrt(profile_err**2/profile_f**2)
        profile_f = profile_f-1

        fig, ax0 = plt.subplots()
        ax0.plot(velocities, profile_f, color = 'r', label = 'LSD')
        zero_line = [0]*len(velocities)
        ax0.plot(velocities, zero_line)
        ax0.plot(velocities, np.exp(initial_inputs[:k_max])-1, label = 'initial')
        ax0.fill_between(velocities, profile_f-profile_errors_f, profile_f+profile_errors_f, alpha = 0.3, color = 'r')
        ax0.set_xlabel('velocities')
        ax0.set_ylabel('flux')
        ax0.legend()
        #plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/order%s_final_profile_%s'%(order, run_name))
        #plt.show()

        print('Profile: %s\nContinuum Coeffs: %s\n'%(profile, poly_cos))

        for i in range(len(poly_cos)):
            hdr['COEFF %s'%i]= poly_cos[i]

        hdr['ORDER'] = order
        hdr['PHASE'] = phi

        if phi<deltaphi:
            result = 'in'
        elif phi>1-deltaphi:
            result = 'in'
        else:
            result = 'out'

        hdr['result'] = result
        hdr['CRVAL1']=np.min(velocities)
        hdr['CDELT1']=velocities[1]-velocities[0]

        hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))

    hdu.writeto('/home/lsd/Documents/LSD_Figures/%s_%s_%s.fits'%(month, frame, run_name), output_verify = 'fix', overwrite = 'True')

## asks before showing all the figures - will only show figures for last order
input2 = input('View figures? y or n: ')
if input2 == 'y':
    plt.show()
else:
    plt.close('all')
