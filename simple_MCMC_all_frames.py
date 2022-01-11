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

def make_gauss(vgrid, A, linewidth, offset):
    """
    returns the line profile for the different points on the star
    as a 2d array with one axis being velocity and other axis position
    on the star
    npix - number of pixels along one axis of the star (assumes solid bosy rotation)
    rstar - the radius of the star in pixels
    xc - the midpoint of the star in pixels
    vgrid - the velocity grid for the spectrum you wish to make (1d array in km/s)
    A - the line depth of the intrinsic profile - the bottom is at (1 - A) is the max line depth (single value)
    veq - the equatorial velocity (the v sin i for star of inclination i) in km/s (single value)
    linewidth - the sigma of your Gaussian line profile in km/s (single value)
    """
    profile=1.-A*np.exp( -(vgrid-offset)**2/linewidth**2)
    return profile-1

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

def read_in_frames(order, filelist):
    frames = []
    errors = []
    frame_wavelengths = []
    sns = []
    max_sn = 0
    #read in all frames
    for file in filelist:
        fluxes, wavelengths, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'unmasked', run_name)
        frame_wavelengths.append(wavelengths)
        frames.append(fluxes)
        errors.append(flux_error_order)
        sns.append(sn)
        #find highest S/N frame
        if sn>max_sn:
            max_sn = sn
            reference_frame=fluxes
            reference_error=flux_error_order


    frames = np.array(frames)
    errors = np.array(errors)
    #divide all frames by this

    for n in range(len(frames)):
        div_frame = frames[n]/reference_frame

        #need to change to bin and sigma clip this before putting in
        ## sigma clip combined data set
        m = np.median(div_frame)
        sigma = np.std(div_frame)
        a = 1

        upper_clip = m+a*sigma
        lower_clip = m-a*sigma

        print(lower_clip)
        print(upper_clip)

        idx = np.logical_and(div_frame>=lower_clip, div_frame<=upper_clip)
        '''
        wavelengths1 = wavelengths.copy()
        plt.figure()
        plt.plot(wavelengths1, div_frame)
        plt.plot(wavelengths1[idx], div_frame[idx])
        plt.show()
        '''
        #bin data
        binned = np.zeros(int(len(div_frame)/2))
        binned_waves = np.zeros(int(len(div_frame)/2))
        for i in range(0, len(div_frame)-1, 2):
            pos = int(i/2)
            binned[pos] = (div_frame[i]+div_frame[i+1])/2
            binned_waves[pos] = (wavelengths[i]+wavelengths[i+1])/2

        #fit to div_frame
        coeffs=np.polyfit(binned_waves, binned, 2)
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)
        frames[n] = frames[n]/fit
        errors[n] = errors[n]/fit
        '''
        plt.figure()
        plt.plot(wavelengths, frames[n])
        plt.plot(wavelengths, fit)



        plt.figure()
        plt.plot(binned_waves, binned)
        plt.plot(binned_waves, poly(binned_waves))
        plt.show()
        '''

    return frame_wavelengths, frames, errors, sns

def continuumfit(fluxes1, wavelengths1, errors1, poly_ord):

        fluxes1 = fluxes1
        ## taking out masked areas
        if np.max(fluxes1)<1:
            idx = [errors1<1]
            print(idx)
            errors = errors1[tuple(idx)]
            fluxes = fluxes1[tuple(idx)]
            wavelengths = wavelengths1[tuple(idx)]
        else:
            errors = errors1
            fluxes = fluxes1
            wavelengths = wavelengths1

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

        idx = [flux_obs<=0]
        flux_obs[idx] = 0.00000000000000001
        new_errors[idx]=1000000000000000
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
        plt.show()
        '''
        coeffs=coeffs[::-1]
        coeffs = np.array(coeffs)

        return coeffs, flux_obs, new_errors, fit

def combine_spec(wavelengths, spectra, errors, sns):
    #combine all spectra to one spectrum
    length, width = np.shape(spectra)
    spectrum = np.zeros((width,))
    spec_errors = np.zeros((width,))

    for n in range(0,width):
        temp_spec = spectra[:, n]
        temp_err = errors[:, n]

        #plt.figure('temp spec')
        #plt.errorbar(np.arange(len(temp_spec)), temp_spec, yerr = temp_err, ecolor = 'k')

        weights = (1/temp_err**2)
        weights = weights/np.sum(weights)

        #plt.figure('temp spec weights')
        #plt.scatter(np.arange(len(temp_spec)), weights)

        print(np.std(temp_spec)**2)

        #weights = np.array(weights/sum(weights))
        spectrum[n]=sum(weights*temp_spec)
        sn = sum(weights*sns)/sum(weights)
        #spectrum[0,n]=sum(temp_spec)/len(temp_spec)
        spec_errors[n]=(np.std(temp_spec))*np.sqrt(sum(weights**2))
        #spec_errors[0, n] = stdev(temp_spec)
        #print(temp_spec)
        #print(temp_err)
        '''
        print(sum(weights)**2)
        print(np.sqrt(sum(weights**2)))
        print(spectrum[n])
        print(spec_errors[n])
        plt.show()
        '''
    '''
    plt.figure('frames')
    for n in range(0, len(spectra)):
        #plt.errorbar(wavelengths, frames[n], yerr=errors[n], ecolor = 'k')
        plt.plot(wavelengths, spectra[n], label = '%s'%n)
    plt.errorbar(wavelengths, spectrum, label = 'combined', color = 'b', yerr = spec_errors, ecolor = 'k')
    #plt.xlim(np.min(frames[n])-1, np.max(frames[n])+1)
    plt.legend()
    plt.show()
    '''
    return wavelengths, spectrum, spec_errors, sn

def od2flux(x):
    return np.exp(x)-1

def flux2od(x):
    return np.log(x+1)

no_line = 100
## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
def model_func(inputs, x):
    #z = inputs[:k_max]
    z = make_gauss(velocities, inputs[0], inputs[1], inputs[2])
    k_max = 3

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
    '''
    plt.figure('continuum fit and final mdl')
    plt.plot(x, mdl1)

    plt.plot(x, mdl)
    plt.errorbar(x, y, yerr = yerr)
    plt.show()
    '''
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


    if 0<=theta[0]<=5 and 0<theta[1]:
            # A                 lw
        return 0.0

    '''
    for i in range(len(theta)):
        if i<k_max: ## must lie in z
            if -10<=theta[i]<=0.5: pass
            else:
                check = 1
                '''

    if check==0:
        ## penalty function for profile - not in use
        '''
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

        # calcualte gaussian probability for each point in continuum
        p_pent = np.sum((1/np.sqrt(2*np.pi*p_var**2))*np.exp(-0.5*(z_cont/p_var)**2))

        return p_pent
        '''
        return 0

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
def residual_mask(wavelengths, data_spec_in, data_err, initial_inputs):
    #residuals=((data_spec+1)/(forward+1))/data_err
    #residuals=abs(((data_spec)-(forward))/(forward))
    forward = model_func(initial_inputs, wavelengths)

    #normalise
    data_spec = (data_spec_in - np.min(data_spec_in))/(np.max(data_spec_in)-np.min(data_spec_in))
    forward = (forward - np.min(forward))/(np.max(forward)-np.min(forward))

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

    fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, sharex = True)
    non_masked = tuple([data_err<1000000000000000000])
    ax[1].scatter(wavelengths[non_masked], residuals[non_masked], marker = '.')
    ax[0].plot(wavelengths, data_spec, 'r', alpha = 0.3, label = 'data')
    ax[0].plot(wavelengths, forward, 'k', alpha =0.3, label = 'mcmc mdl')
    residual_masks = tuple([data_err>1000000000000000000])

    ax[0].scatter(wavelengths[residual_masks], data_spec[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    ax[0].legend(loc = 'lower right')
    #plotdepths = -np.array(line_depths)
    #ax[0].vlines(line_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
    #ax[1].plot(x, residuals_2, '.')
    #ax[1].scatter(x[residual_masks], residuals_2[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    z_line = [0]*len(x)
    ax[1].plot(x, z_line, '--')
    '''
    plt.figure('Masking based off residuals (>0.25 for >20 masked)')
    plt.plot(wavelengths, data_spec)
    residual_masks = tuple([data_err>1000000000000000000])
    plt.scatter(wavelengths[residual_masks], data_spec[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    plt.xlabel('wavelengths')
    plt.ylabel('flux')
    plt.show()
    '''
    poly_inputs, bin, bye, fit=continuumfit(data_spec_in,  (wavelengths*a)+b, data_err, poly_ord)
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
    ## second round of masking - based of new profile
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

## finding continuum fit for each order
filelist=findfiles(directory, file_type)
order_range = np.arange(12,71)

P=2.21857567 #Cegla et al, 2006 - days
T=2454279.436714 #Cegla et al, 2006
t=0.076125 #Torres et al, 2008
deltaphi = t/(2*P)

for order in order_range:
    poly_ord = 3

    frame_wavelengths, frames, frame_errors, sns = read_in_frames(order, filelist)
    wavelengths, fluxes, flux_error_order, sn = combine_spec(frame_wavelengths[-1], frames, frame_errors, sns)

    ## normalise
    '''
    flux_error_order = (flux_error_order)/(np.max(fluxes)-np.min(fluxes))
    fluxes = (fluxes - np.min(fluxes))/(np.max(fluxes)-np.min(fluxes))
    '''
    # edit the errors
    flux_error_order = flux_error_order

    print("SN: %s"%sn)
    '''
    plt.figure()
    plt.title('Normalised Spectrum with errors')
    plt.errorbar(wavelengths, fluxes, yerr = flux_error_order, ecolor = 'k')
    #plt.xlim(np.min(fluxes)-1, np.max(fluxes)+1)
    plt.show()
    '''
    ## running the MCMC
    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)
    poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes,  (wavelengths*a)+b, flux_error_order, poly_ord)
    '''
    plt.figure()
    plt.title('Normalised Spectrum with errors')
    plt.errorbar(wavelengths, fluxes, yerr = flux_error_order, ecolor = 'k')
    #plt.xlim(np.min(fluxes)-1, np.max(fluxes)+1)
    plt.show()
    '''

    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name)
    print('VELOCITIES')
    print(velocities)

    if order == order_range[0]:
        all_frames = np.zeros((len(filelist), 71, 2, len(profile)))
    '''
    plt.figure('intial LSD')
    plt.plot(velocities, profile)
    plt.show()
    #poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes,  (wavelengths*a)+b, flux_error_order, poly_ord)
    '''


    ## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
    j_max = int(len(fluxes))
    k_max = 3

    profile = [0.559, 3.5, -2.2765]
    profile = np.array(profile)
    model_inputs = np.concatenate((profile, poly_inputs))

    plt.figure()
    plt.plot(velocities, make_gauss(velocities, model_inputs[0], model_inputs[1], model_inputs[2]))
    plt.show()
    ## setting x, y, yerr for emcee
    x = wavelengths
    y = fluxes
    yerr = flux_error_order

    ## setting these normalisation factors as global variables - used in the figures below
    a = 2/(np.max(x)-np.min(x))
    b = 1 - a*np.max(x)

    ## parameters for working out continuum points in the LSD profile - if using penalty function
    p_var = 0.001
    v_min = -10
    v_max = 10

    #masking based off residuals
    yerr, model_inputs_bin = residual_mask(x, y, yerr, model_inputs)

    ##masking frames also
    mask_idx = tuple([yerr>1000000000000000000])
    #frame_errors[:, idx]=1000000000000000000

    ## setting number of walkers and their start values(pos)
    ndim = len(model_inputs)
    nwalkers= ndim*3
    rng = np.random.default_rng()

    ## starting values of walkers vary from the model_inputs by 0.01*model_input - this means the bottom of the profile varies more than the continuum. Continuum coefficent vary by 1*model_input.
    vary_amounts = []
    pos = []
    for i in range(0, ndim):
        if model_inputs[i]==0:pos2 = model_inputs[i]+rng.normal(model_input[i], 0.0001, (nwalkers, ))
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
    '''
    plt.figure('Spectrum given to mcmc with errors')
    plt.title('Spectrum given to mcmc with errors')
    plt.errorbar(x, y, yerr=yerr, ecolor = 'k')
    plt.ylabel('flux')
    plt.xlabel('wavelength')
    plt.show()
    '''

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
        #mdl = model_func(sample, x)
        #mdl = mdl[idx]
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
    plt.savefig('/home/lsd/Documents/LSD_Figures/mc_mdl/order%s_mc_mdl_%s'%(order, run_name))
    #plt.show()

    ## getting the final profile and continuum values - median of last 1000 steps
    profile_out = []
    poly_cos = []
    profile_err_out = []
    poly_cos_err = []

    for i in range(ndim):
        mcmc = np.median(flat_samples[:, i])
        error = np.std(flat_samples[:, i])
        if i<ndim-poly_ord-1:
            profile_out.append(mcmc)
            profile_err_out.append(error)
        else:
            poly_cos.append(mcmc)
            poly_cos_err.append(error)

    #profile = np.array(profile)
    #profile_err = np.array(profile_err)

    profile = make_gauss(velocities, profile_out[0], profile_out[1], profile_out[2])
    profile_err = profile*0.001

    prof_flux = np.exp(profile)-1
    # plots the mcmc profile - will have extra panel if it's for data

    fig, ax0 = plt.subplots()
    ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
    zero_line = [0]*len(velocities)
    ax0.plot(velocities, zero_line)
    #ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
    ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
    ax0.set_xlabel('velocities')
    ax0.set_ylabel('optical depth')
    secax = ax0.secondary_yaxis('right', functions = (od2flux, flux2od))
    secax.set_ylabel('flux')
    ax0.legend()
    plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/order%s_profile_%s'%(order, run_name))
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
    plt.legend()
    plt.title('continuum from mcmc')
    plt.xlabel("wavelengths")
    plt.ylabel("flux")
    plt.savefig('/home/lsd/Documents/LSD_Figures/continuum_fit/order%s_cont_%s'%(order, run_name))
    #plt.show()

    ## last section is a bit of a mess but plots the two forward models

    mcmc_inputs = np.concatenate((profile, poly_cos))
    mcmc_mdl = model_func(mcmc_inputs, x)
    #mcmc_mdl = mcmc_mdl[idx]
    mcmc_liklihood = log_probability(mcmc_inputs, x, y, yerr)

    print('Likelihood for mcmc: %s'%mcmc_liklihood)

    residuals_2 = (y+1) - (mcmc_mdl+1)

    fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
    non_masked = tuple([yerr<1000000000000])
    #ax[0].plot(x, y+1, color = 'r', alpha = 0.3, label = 'data')
    #ax[0].plot(x[non_masked], mcmc_mdl[non_masked]+1, color = 'k', alpha = 0.3, label = 'mcmc spec')
    ax[1].scatter(x[non_masked], residuals_2[non_masked], marker = '.')
    ax[0].plot(x, y, 'r', alpha = 0.3, label = 'data')
    ax[0].plot(x, mcmc_mdl, 'k', alpha =0.3, label = 'mcmc spec')
    residual_masks = tuple([yerr>1000000000000])

    #residual_masks = tuple([yerr>10])
    ax[0].scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    ax[0].legend(loc = 'lower right')
    #ax[0].set_ylim(0, 1)
    #plotdepths = -np.array(line_depths)
    #ax[0].vlines(line_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
    #ax[1].plot(x, residuals_2, '.')
    #ax[1].scatter(x[residual_masks], residuals_2[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
    z_line = [0]*len(x)
    ax[1].plot(x, z_line, '--')
    plt.savefig('/home/lsd/Documents/LSD_Figures/forward_models/order%s_forward_%s'%(order, run_name))
    #plt.show()

    fig, ax0 = plt.subplots()
    ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
    zero_line = [0]*len(velocities)
    ax0.plot(velocities, zero_line)
    #ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
    ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
    ax0.set_xlabel('velocities')
    ax0.set_ylabel('optical depth')
    ax0.legend()
    plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/order%s_final_profile_%s'%(order, run_name))
    #plt.show()

    profile_f = np.exp(profile)
    profile_errors_f = np.sqrt(profile_err**2/profile_f**2)
    profile_f = profile_f-1

    fig, ax0 = plt.subplots()
    ax0.plot(velocities, profile_f, color = 'r', label = 'LSD')
    zero_line = [0]*len(velocities)
    ax0.plot(velocities, zero_line)
    #ax0.plot(velocities, np.exp(model_inputs[:k_max])-1, label = 'initial')
    ax0.fill_between(velocities, profile_f-profile_errors_f, profile_f+profile_errors_f, alpha = 0.3, color = 'r')
    ax0.set_xlabel('velocities')
    ax0.set_ylabel('flux')
    ax0.legend()
    #plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/order%s_final_profile_%s'%(order, run_name))
    #plt.show()

    '''
    profile = model_inputs[:k_max]
    poly_cos = model_inputs[k_max:]
    '''
    print('Profile: %s\nContinuum Coeffs: %s\n'%(profile, poly_cos))
    #print('True likelihood: %s\nMCMC likelihood: %s\n'%(true_liklihood, mcmc_liklihood))
    #profile_order.append(profile)
    #coeffs_order.append(poly_cos)
    #print('Time Taken: %s minutes'%((t1-t0)/60))

    mdl1 =0
    for i in np.arange(0, len(poly_cos)):
        mdl1 = mdl1+poly_cos[i]*((a*x)+b)**(i)


    profiles = []
    plt.figure('corrected spec, order 8')
    for counter in range(0, len(frames)):
        flux = frames[counter]
        error = frame_errors[counter]
        wavelengths = frame_wavelengths[counter]

        #masking based off residuals
        error[mask_idx]=10000000000000000

        ## normalise
        '''
        error = (error)/(np.max(flux)-np.min(flux))
        flux = (flux - np.min(flux))/(np.max(flux)-np.min(flux))
        '''
        flux = flux/mdl1
        error = error/mdl1

        idx = tuple([flux>0])
        error = error[idx]
        wavelengths = wavelengths[idx]
        flux = flux[idx]

        plt.plot(wavelengths, flux, label = '%s'%counter)

        velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, flux, error, linelist, 'False', poly_ord, sn, order, run_name)

        profile_f = np.exp(profile)
        profile_errors_f = np.sqrt(profile_err**2/profile_f**2)
        profile_f = profile_f-1

        all_frames[counter, order]=[profile_f, profile_errors_f]

        profiles.append(profile)

    plt.legend()

    count = 0
    plt.figure()
    plt.title("HARPS CCFs")
    file_list = findfiles(directory, 'ccf')
    for file in file_list[:-1]:
        ccf = fits.open(file)
        ccf_spec = ccf[0].data[order]
        velocities_ccf=ccf[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*ccf[0].header['CDELT1']
        plt.plot(velocities_ccf, ccf_spec/ccf_spec[0]-1, label = '%s'%count)
        count +=1
    plt.legend()
    plt.ylabel('flux')
    plt.xlabel('velocities km/s')
    #plt.show()

    plt.figure()
    plt.title('order %s, LSD profiles'%order)
    no=0

    print(velocities)

    for profile in profiles:
        plt.plot(velocities, np.exp(profile)-1, label = '%s'%no)
        no+=1
    plt.legend()
    plt.ylabel('flux')
    plt.xlabel('velocities km/s')
    plt.show()
    #plt.close('all')

# adding into fits files for each frame
for frame_no in range(0, len(frames)):
    file = filelist[frame_no]
    fits_file = fits.open(file)
    phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
    hdu = fits.HDUList()
    hdr = fits.Header()

    for order in range(0, 71):
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

        profile = all_frames[frame_no, order, 0]
        profile_err = all_frames[frame_no, order, 1]

        hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
    hdu.writeto('/home/lsd/Documents/LSD_Figures/%s_%s_%s.fits'%(month, frame_no, run_name), output_verify = 'fix', overwrite = 'True')
plt.show()
