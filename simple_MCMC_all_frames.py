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
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import multiprocessing as mp
from functools import partial

from math import log10, floor

def round_sig(x1, sig):
    return round(x1, sig-int(floor(log10(abs(x1))))-1)

from scipy.optimize import curve_fit

def gauss(x1, rv, sd, height, cont):
    y1 = height*np.exp(-(x1-rv)**2/(2*sd**2)) + cont
    return y1

## for real data
file_type = 'e2ds_A'
linelist = '/Users/lucydolan/Documents/HD189733/TEST/fulllinelist004.txt'
directory_p = '/Users/lucydolan/Documents/HD189733/TEST'
global velocities
if file_type == 'e2ds_A':
    velocities=np.arange(-21, 18, 0.82)
else: velocities = np.arange(-21, 18, 0.82) ## input alternative here
poly_ord = 3
months = [''] ## observation nights - subfolders in the directory
order_range = np.arange(10,70) ## orders to be covered
run_name = input('Input nickname for this version of code (for saving figures): ')
P=2.21857567 #Cegla et al, 2006 - days
T=2454279.436714 #Cegla et al, 2006
t=0.076125 #Torres et al, 2008
deltaphi = t/(2*P)

def findfiles(directory, file_type):

    filelist1=glob.glob('%s/%s*.fits'%(directory, file_type))    #finding spectra
    
    return filelist1

def read_in_frames(order, filelist):
    frames = []
    errors = []
    frame_wavelengths = []
    sns = []
    max_sn = 0
    ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
    for file in filelist:
        if file_type == 'e2ds_A' or file_type == 's1d':
            fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct(file_type, 'order', order, file, directory, 'unmasked', run_name, 'y')
        else:
            data_file = fits.open(file)
            sn = data_file[0].header['S/N']
            wavelengths = data_file[0].data
            fluxes = data_file[1].data
            flux_error_order = data_file[2].data

        frame_wavelengths.append(wavelengths)
        frames.append(fluxes)
        errors.append(flux_error_order)
        sns.append(sn)

        ### finding highest S/N frame, saves this as reference frame
        if sn>max_sn:
            max_sn = sn
            global reference_wave
            reference_wave = wavelengths
            reference_frame=fluxes
            reference_frame[reference_frame == 0]=0.001
            reference_error=flux_error_order
            reference_error[reference_frame == 0]=1000000000000000000

    frames = np.array(frames)
    errors = np.array(errors)

    global frames_unadjusted
    frames_unadjusted = frames
    global frame_errors_unadjusted
    frame_errors_unadjusted = errors

    ### each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
    for n in range(len(frames)):
        f2 = interp1d(frame_wavelengths[n], frames[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
        div_frame = f2(reference_wave)/reference_frame

        ### creating windows to fit polynomial to
        binned = np.zeros(int(len(div_frame)/2))
        binned_waves = np.zeros(int(len(div_frame)/2))
        for i in range(0, len(div_frame)-1, 2):
            pos = int(i/2)
            binned[pos] = (div_frame[i]+div_frame[i+1])/2
            binned_waves[pos] = (reference_wave[i]+reference_wave[i+1])/2

        ### fitting polynomial to div_frame
        coeffs=np.polyfit(binned_waves, binned, 2)
        poly = np.poly1d(coeffs)
        fit = poly(frame_wavelengths[n])
        frames[n] = frames[n]/fit
        errors[n] = errors[n]/fit

    return frame_wavelengths, frames, errors, sns

def continuumfit(fluxes1, wavelengths1, errors1, poly_ord):

        cont_factor = fluxes1[0]

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
        binsize = int(round(len(fluxes)/10, 1))
        for i in range(0, len(wavelength), binsize):
            if i+binsize<len(wavelength):
                waves = wavelength[i:i+binsize]
                flux = fluxe[i:i+binsize]
                indicies = flux.argsort()
                flux = flux[indicies]
                waves = waves[indicies]

                clipped_flux.append(flux[len(flux)-1])
                clipped_waves.append(waves[len(waves)-1])

        coeffs=np.polyfit(clipped_waves, clipped_flux/cont_factor, poly_ord)

        poly = np.poly1d(coeffs*cont_factor)
        fit = poly(wavelengths1)
        flux_obs = fluxes1/fit
        new_errors = errors1/fit

        idx = [flux_obs<=0]
        flux_obs[idx] = 0.00000000000000001
        new_errors[idx]=1000000000000000
        
        coeffs=coeffs[::-1]
        coeffs = list(coeffs)
        coeffs.append(cont_factor)
        coeffs = np.array(coeffs)
        
        return coeffs, flux_obs, new_errors, fit

def combine_spec(wavelengths_f, spectra_f, errors_f, sns_f):

    interp_spec = np.zeros(spectra_f.shape)
    #combine all spectra to one spectrum
    for n in range(len(wavelengths_f)):

        idx = np.where(wavelengths_f[n] != 0)[0]

        f2 = interp1d(wavelengths_f[n][idx], spectra_f[n][idx], kind = 'linear', bounds_error=False, fill_value = 'NaN')
        f2_err = interp1d(wavelengths_f[n][idx], errors_f[n][idx], kind = 'linear', bounds_error=False, fill_value = 'NaN')
        spectra_f[n] = f2(reference_wave)
        errors_f[n] = f2_err(reference_wave)

        ## mask out out extrapolated areas
        idx_ex = np.logical_and(reference_wave<np.max(wavelengths_f[n][idx]), reference_wave>np.min(wavelengths_f[n][idx]))
        idx_ex = tuple([idx_ex==False])

        spectra_f[n][idx_ex]=1.
        errors_f[n][idx_ex]=1000000000000

        ## mask out nans and zeros (these do not contribute to the main spectrum)
        where_are_NaNs = np.isnan(spectra_f[n])
        errors_f[n][where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spectra_f[n] == 0)[0]
        errors_f[n][where_are_zeros] = 1000000000000

        where_are_NaNs = np.isnan(errors_f[n])
        errors_f[n][where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(errors_f[n] == 0)[0]
        errors_f[n][where_are_zeros] = 1000000000000

    width = len(reference_wave)
    spectrum_f = np.zeros((width,))
    spec_errors_f = np.zeros((width,))

    for n in range(0,width):
        temp_spec_f = spectra_f[:, n]
        temp_err_f = errors_f[:, n]

        weights_f = (1/temp_err_f**2)

        idx = tuple([temp_err_f>=1000000000000])
        print(weights_f[idx])
        weights_f[idx] = 0.

        weights_f = weights_f/np.sum(weights_f)

        spectrum_f[n]=sum(weights_f*temp_spec_f)
        sn_f = sum(weights_f*sns_f)/sum(weights_f)

        spec_errors_f[n]=1/(sum(weights_f**2))
    
    return reference_wave, spectrum_f, spec_errors_f, sn_f

def od2flux(x):
    return np.exp(x)-1

def flux2od(x):
    return np.log(x+1)

no_line = 100
## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
def model_func(inputs, x):
    z = inputs[:k_max]

    mdl = np.dot(alpha, z) ##alpha has been declared a global variable after LSD is run.
   
    #converting model from optical depth to flux
    mdl = np.exp(mdl)

    ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
    a = 2/(np.max(x)-np.min(x))
    b = 1 - a*np.max(x)

    mdl1=0
    for i in range(k_max,len(inputs)-1):
        mdl1 = mdl1 + (inputs[i]*((x*a)+b)**(i-k_max))

    mdl1 = mdl1 * inputs[-1]

    mdl = mdl * mdl1
   
    return mdl

def convolve(profile, alpha):
    spectrum = np.dot(alpha, profile)
    return spectrum

## maximum likelihood estimation for the mcmc model.
def log_likelihood(theta, x, y, yerr):
    model = model_func(theta, x)

    lnlike = -0.5 * np.sum(((y) - (model)) ** 2 / yerr**2 + np.log(yerr**2)+ np.log(2*np.pi))

    return lnlike

## imposes the prior restrictions on the inputs - rejects if profile point if less than -10 or greater than 0.5.
def log_prior(theta):

    check = 0
    z = theta[:k_max]


    for i in range(len(theta)):
        if i<k_max: ## must lie in z
            if -10<=theta[i]<=0.5: pass
            else:
                check = 1


    if check==0:

        # excluding the continuum points in the profile (in flux)
        z_cont = []
        v_cont = []
        for i in range(0, 5):
                z_cont.append(np.exp(z[len(z)-i-1])-1)
                v_cont.append(velocities[len(velocities)-i-1])
                z_cont.append(np.exp(z[i])-1)
                v_cont.append(velocities[i])

        z_cont = np.array(z_cont)

        p_pent = np.sum((np.log((1/np.sqrt(2*np.pi*0.01**2)))-0.5*(z_cont/0.01)**2))

        return p_pent

    return -np.inf

## calculates log probability - used for mcmc
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    final = lp + log_likelihood(theta, x, y, yerr)
    return final

## iterative residual masking - mask continuous areas first - then possibly progress to masking the narrow lines
def residual_mask(wavelengths, data_spec_in, data_err, initial_inputs):

    forward = model_func(initial_inputs, wavelengths)

    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(x)

    mdl1=0
    for i in range(k_max,len(initial_inputs)-1):
        mdl1 = mdl1 + (initial_inputs[i]*((wavelengths*a)+b)**(i-k_max))
    mdl1 = mdl1 * initial_inputs[-1]

    residuals = data_spec_in/mdl1 -1
    ### finds consectuative sections where at least 40 points have no conitnuum points in between - these are masked
    flag = ['False']*len(residuals)
    flagged = []
    for i in range(len(residuals)):
        if abs(residuals[i])>0.1:
            flag[i] = 'True'
            if i>0 and flag[i-1] == 'True':
                flagged.append(i-1)
        else:
            if len(flagged)>0:
                flagged.append(i-1)
                print(abs(np.median(residuals[flagged[0]:flagged[-1]])))
                if len(flagged)<300:
                    for no in flagged:
                        flag[no] = 'False'
                else:
                    idx = np.logical_and(wavelengths>=wavelengths[np.min(flagged)]-1, wavelengths<=wavelengths[np.max(flagged)]+1)
                    data_err[idx]=10000000000000000000
            flagged = []

    residuals = (data_spec_in - np.min(data_spec_in))/(np.max(data_spec_in)-np.min(data_spec_in)) - (forward - np.min(forward))/(np.max(forward)-np.min(forward)) 

    ### finds consectuative sections where at least 20 points have residuals greater than 0.25 - these are masked
    flag = ['False']*len(residuals)
    flagged = []
    for i in range(len(residuals)):
        if abs(residuals[i])>0.25:
            flag[i] = 'True'
            if i>0 and (flag[i-1] == 'True' or flag[i-2] == 'True'):
                flagged.append(i-1)
        else:
            if len(flagged)>0:
                flagged.append(i-1)
                print(abs(np.median(residuals[flagged[0]:flagged[-1]])))
                if len(flagged)<20 or abs(np.mean(residuals[flagged[0]:flagged[-1]]))<0.45:
                    for no in flagged:
                        flag[no] = 'False'
                else:
                    idx = np.logical_and(wavelengths>=wavelengths[np.min(flagged)]-1, wavelengths<=wavelengths[np.max(flagged)]+1)
                    data_err[idx]=10000000000000000000
            flagged = []

    ###     masking tellurics      ###

    limit_wave =21/2.99792458e5 #needs multiplied by wavelength to give actual limit
    limit_pix = limit_wave/((max(wavelengths)-min(wavelengths))/len(wavelengths))
    
    tell_lines = [3820.33, 3933.66, 3968.47, 4327.74, 4307.90, 4383.55, 4861.34, 5183.62, 5270.39, 5889.95, 5895.92, 6562.81, 7593.70, 8226.96]
    for line in tell_lines:
        limit = limit_wave*line +3
        idx = np.logical_and((line-limit)<=wavelengths, wavelengths<=(limit+line))
        data_err[idx] = 1000000000000000000

    residual_masks = tuple([data_err>=1000000000000000000])
    
    plt.figure()
    plt.scatter(wavelengths[residual_masks], data_spec_in[residual_masks], color = 'r')
    plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/order%s_masking_%s'%(order, run_name))
    plt.close()
    
    ###      sigma clip masking     ###

    m = np.median(residuals)
    sigma = np.std(residuals)
    a = 1

    upper_clip = m+a*sigma
    lower_clip = m-a*sigma

    rcopy = residuals.copy()

    idx1 = tuple([rcopy<=lower_clip])
    idx2 = tuple([rcopy>=upper_clip])

    data_err[idx1]=10000000000000000000
    data_err[idx2]=10000000000000000000

    poly_inputs, bin, bye, fit=continuumfit(data_spec_in,  (wavelengths*a)+b, data_err, poly_ord)
    velocities1, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, bin, bye, linelist, 'False', poly_ord, sn, order, run_name, velocities)

    return data_err, np.concatenate((profile, poly_inputs)), residual_masks

def task(all_frames, counter):
    flux = frames[counter]
    error = frame_errors[counter]
    wavelengths = frame_wavelengths[counter]
    sn = sns[counter]
   
    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)

    mdl1 =0
    for i in np.arange(0, len(poly_cos)-1):
        mdl1 = mdl1+poly_cos[i]*((a*wavelengths)+b)**(i)
    mdl1 = mdl1*poly_cos[-1]

    #masking based off residuals (needs to be redone for each frame as wavelength grid is different) -- NO - the same masking needs to be applied to each frame
    #the mask therefore needs to be interpolated onto the new wavelength grid.
    mask_pos = np.ones(reference_wave.shape)
    mask_pos[mask_idx]=10000000000000000000
    f2 = interp1d(reference_wave, mask_pos, bounds_error = False, fill_value = np.nan)
    interp_mask_pos = f2(wavelengths)
    interp_mask_idx = tuple([interp_mask_pos>=10000000000000000000])

    error[interp_mask_idx]=1000000000000000000

    flux = flux/mdl1
    error = error/mdl1

    remove = tuple([flux<0])
    flux[remove]=1.
    error[remove]=10000000000000000000

    idx = tuple([flux>0])
    
    velocities1, profile1, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, flux, error, linelist, 'False', poly_ord, sn, order, run_name, velocities)

    profile_f = np.exp(profile1)
    profile_errors_f = np.sqrt(profile_errors**2/profile_f**2)
    profile_f = profile_f-1

    all_frames[counter, order]=[profile_f, profile_errors_f]
    
    return all_frames

for month in months:
    directory = '%s%s/'%(directory_p, month)
    filelist=findfiles(directory, file_type)
    global all_frames
    all_frames = np.zeros((len(filelist), 71, 2, len(velocities)))
    for order in order_range:

        ## reads in data
        frame_wavelengths, frames, frame_errors, sns = read_in_frames(order, filelist)

        ## combines spectra from each frame (weighted based of S/N), returns to S/N of combined spec
        fw = np.array(frame_wavelengths.copy())
        f = np.array(frames.copy())
        fe = np.array(frame_errors.copy())
        s = np.array(sns.copy())
        wavelengths, fluxes, flux_error_order, sn = combine_spec(fw, f, fe, s)
        
        ### getting the initial polynomial coefficents
        a = 2/(np.max(wavelengths)-np.min(wavelengths))
        b = 1 - a*np.max(wavelengths)
        poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes, (wavelengths*a)+b, flux_error_order, poly_ord)

        #### getting the initial profile
        velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name, velocities)

        ## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
        j_max = int(len(fluxes))
        k_max = len(profile)

        model_inputs = np.concatenate((profile, poly_inputs))

        ## setting x, y, yerr for emcee
        x = wavelengths
        y = fluxes
        yerr = flux_error_order

        #masking based off residuals
        yerr_unmasked = yerr
        yerr, model_inputs_resi, mask_idx = residual_mask(x, y, yerr, model_inputs)

        ###   MCMC Continuum Fit    ###

        ## setting number of walkers and their start values(pos)
        ndim = len(model_inputs)
        nwalkers= ndim*3
        rng = np.random.default_rng()

        ### starting values of walkers
        sigma = 0.8*0.005
        pos = []
        for i in range(0, ndim):
            if i <ndim-poly_ord-2:
                pos2 = rng.normal(model_inputs[i], sigma, (nwalkers, ))
            else:
                print(model_inputs[i])
                sigma = abs(round_sig(model_inputs[i], 1))/10
                print(sigma)
                # print(sigma_cont[i-k_max])
                pos2 = rng.normal(model_inputs[i], sigma, (nwalkers, ))
            pos.append(pos2)

        pos = np.array(pos)
        pos = np.transpose(pos)

        steps_no = 10000
    
        with mp.Pool(mp.cpu_count()) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), pool = pool)
            sampler.run_mcmc(pos, steps_no, progress=True)
        
        idx = tuple([yerr<=10000000000000000000])

        ## discarding all values except the last 1000 steps.
        dis_no = int(np.floor(steps_no-1000))

        ## combining all walkers together
        flat_samples = sampler.get_chain(discard=dis_no, flat=True)

        ## getting the final profile and continuum values - median of last 1000 steps
        profile = []
        poly_cos = []
        profile_err = []
        poly_cos_err = []

        for i in range(ndim):
            mcmc = np.median(flat_samples[:, i])
            error = np.std(flat_samples[:, i])
            mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
            error = np.diff(mcmc)
            if i<k_max:
                profile.append(mcmc[1])
                profile_err.append(np.max(error))
            else:
                poly_cos.append(mcmc[1])
                poly_cos_err.append(np.max(error))

        profile = np.array(profile)
        profile_err = np.array(profile_err)

        fig_opt = 'n'
        if fig_opt == 'y':
            # plots the mcmc profile - will have extra panel if it's for data
            fig, ax0 = plt.subplots()
            ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
            zero_line = [0]*len(velocities)
            ax0.plot(velocities, zero_line)
            ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
            ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
            ax0.set_xlabel('velocities')
            ax0.set_ylabel('optical depth')
            secax = ax0.secondary_yaxis('right', functions = (od2flux, flux2od))
            secax.set_ylabel('flux')
            ax0.legend()
            plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/profiles/order%s_profile_%s'%(order, run_name))
            plt.close()

            # plots mcmc continuum fit on top of data
            plt.figure('continuum fit from mcmc')
            plt.plot(x, y, color = 'k', label = 'data')
            mdl1 =0
            for i in np.arange(0, len(poly_cos)-1):
                mdl1 = mdl1+poly_cos[i]*((a*x)+b)**(i)
            mdl1 = mdl1*poly_cos[-1]
            plt.plot(x, mdl1, label = 'mcmc continuum fit')
            mdl1_poserr =0
            for i in np.arange(0, len(poly_cos)-1):
                mdl1_poserr = mdl1_poserr+(poly_cos[i]+poly_cos_err[i])*((a*x)+b)**(i)
            mdl1_poserr = mdl1_poserr*poly_cos[-1]
            mdl1_neg =0
            for i in np.arange(0, len(poly_cos)-1):
                mdl1_neg = mdl1_neg+(poly_cos[i]-poly_cos_err[i])*((a*x)+b)**(i)
            mdl1_neg = mdl1_neg*poly_cos[-1]
            plt.fill_between(x, mdl1_neg, mdl1_poserr, alpha = 0.3)

            mdl1_err =abs(mdl1-mdl1_neg)
            plt.legend()
            plt.title('continuum from mcmc')
            plt.xlabel("wavelengths")
            plt.ylabel("flux")
            plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/continuum_fit/order%s_cont_%s'%(order, run_name))
            plt.close()
        
            ## last section is a bit of a mess but plots the two forward models
            mcmc_inputs = np.concatenate((profile, poly_cos))
            mcmc_mdl = model_func(mcmc_inputs, x)
            mcmc_liklihood = log_probability(mcmc_inputs, x, y, yerr)

            print('Likelihood for mcmc: %s'%mcmc_liklihood)

            residuals_2 = (y+1) - (mcmc_mdl+1)

            fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
            non_masked = tuple([yerr<10])
            ax[1].scatter(x[non_masked], residuals_2[non_masked], marker = '.')
            ax[0].plot(x, y, 'r', alpha = 0.3, label = 'data')
            ax[0].plot(x, mcmc_mdl, 'k', alpha =0.3, label = 'mcmc spec')
            residual_masks = tuple([yerr>=100000000000000])

            #residual_masks = tuple([yerr>10])
            ax[0].scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
            ax[0].legend(loc = 'lower right')
            ax[1].plot(x, residuals_2, '.')
            z_line = [0]*len(x)
            ax[1].plot(x, z_line, '--')
            plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/forward_models/order%s_forward_%s'%(order, run_name))
            plt.close()

            fig, ax0 = plt.subplots()
            ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
            zero_line = [0]*len(velocities)
            ax0.plot(velocities, zero_line)
            ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
            ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
            ax0.set_xlabel('velocities')
            ax0.set_ylabel('optical depth')
            ax0.legend()
            plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/profiles/order%s_final_profile_%s'%(order, run_name))
            plt.close()

            # plots the model for 'walks' of the all walkers for the first 5 profile points
            samples = sampler.get_chain(discard = dis_no)
            fig, axes = plt.subplots(len(samples[0, 0, :]), figsize=(10, 7), sharex=True)
            for i in range(len(samples[0, 0, :10])):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
            axes[-1].set_xlabel("step number")

            # plots random models from flat_samples - lets you see if it's converging
            plt.figure()
            inds = np.random.randint(len(flat_samples), size=100)
            for ind in inds:
                sample = flat_samples[ind]
                mdl = model_func(sample, x)
                mdl1 = 0
                for i in np.arange(k_max, len(sample)-1):
                    mdl1 = mdl1+sample[i]*((a*x)+b)**(i-k_max)
                mdl1 = mdl1*sample[-1]
                plt.plot(x, mdl1, "C1", alpha=0.1)
                plt.plot(x, mdl, "g", alpha=0.1)
            plt.scatter(x, y, color = 'k', marker = '.', label = 'data')
            plt.xlabel("wavelengths")
            plt.ylabel("flux")
            plt.title('mcmc models and data')
            plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/mc_mdl/order%s_mc_mdl_%s'%(order, run_name))
            plt.close()

        profile_f = np.exp(profile)
        profile_errors_f = np.sqrt(profile_err**2/profile_f**2)
        profile_f = profile_f-1
       

        task_part = partial(task, all_frames)
        with mp.Pool(mp.cpu_count()) as pool:results=[pool.map(task_part, np.arange(len(frames)))]
        results = np.array(results[0])
        for i in range(len(frames)):
            all_frames[i]=results[i][i]
            
    plt.close('all')

    # adding into fits files for each frame
    phases = []
    for frame_no in range(0, len(frames)):
        file = filelist[frame_no]
        fits_file = fits.open(file)
        phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
        phases.append(phi)
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
        hdu.writeto('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/%s_%s_%s.fits'%(month, frame_no, run_name), output_verify = 'fix', overwrite = 'True')

