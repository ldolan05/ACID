import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import  fits
import emcee
#import corner
import LSD_func_faster as LSD
import time
# import synthetic_data as syn
import random
import glob
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.signal import find_peaks
import multiprocessing as mp
from functools import partial
import sys
from scipy.optimize import curve_fit
from math import log10, floor

def round_sig(x1, sig):
    return round(x1, sig-int(floor(log10(abs(x1))))-1)

def gauss(x1, rv, sd, height, cont):
    y1 = height*np.exp(-(x1-rv)**2/(2*sd**2)) + cont
    return y1

def findfiles(directory, file_type):
    filelist=glob.glob('%s*%s**A*.fits'%(directory, file_type)) 
    return filelist

K=6.3/1000 #km/s Boisse et al, 2009
#K=0.230
v0=27.4#-0.1875 #km/s Boisse et al, 2009
omega = (np.pi/2)
e = 0.
def remove_reflex(wavelengths, spectrum, errors, phi, K, e, omega, v0):
    velo = v0 + K*(e*np.cos(omega)+np.cos(2*np.pi*phi+omega))
    #print(velo)
    adjusted_wavelengths = wavelengths-wavelengths*velo/2.99792458e5
    f2 = interp1d(adjusted_wavelengths, spectrum, kind='linear', bounds_error=False, fill_value='extrapolate')
    adjusted_spectrum = f2(wavelengths)
    
    return wavelengths, adjusted_spectrum, errors

## test that calclates ccf based off frames and master out of transit frames - needs to be tested that it's working
def frame_ccf(wavelengths, spectra, phases):

    ## interpolating all into the same wavelength grid (first in spectrum)
    new_spectra = np.zeros(spectra.shape)
    for n in range(len(wavelengths)):
        f2 = interp1d(wavelengths[n], spectra[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
        new_spectra[n] = f2(wavelengths[0])
    
    ## isolating and combining out-of-transit spectra
    idx = [phases<deltaphi]
    for j in range(len(new_spectra)):
        w_discard, new_spectra[j], err_discard=remove_reflex(wavelengths[0], new_spectra[j], new_spectra[j]/100, phases[j], K, e, omega, v0)
        plt.plot(wavelengths[0], new_spectra[j])
    #plt.show()

    master_out_spec = np.mean(new_spectra[idx], axis = 0)
    plt.plot(wavelengths[0], master_out_spec, color = 'k')
    plt.show()

    ## calculate ccf between each frame - doesn't pick up on RVs as small as RM effect
    rvs = []
    for p in range(len(phases)):
        ccf = np.zeros(np.arange(-300, 300).shape)
        count = 0
        for v in np.arange(-300, 300):
            w = (np.array(wavelengths[0].copy()))#-wavelengths[0].copy()*0.01/2.99792458e5 ## purposly shifting spectrum to see if it gets picked up by ccf
            wshift = wavelengths[0].copy()*(v/100)/2.99792458e5
            # if v==-20:
            #     plt.figure()
            #     plt.plot(w, new_spectra[n])
            #     plt.show()
            print(w)
            print(wshift)
            f2 = interp1d(w+wshift, new_spectra[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
            ccf[count] = np.sum((f2(wavelengths[0])*master_out_spec))
            count+=1

        ##fit with curve fit
        # plt.figure()
        # ccf = ccf[10:-10]
        # plt.plot(np.arange(-300, 300)/100, 1-(ccf/ccf[0]))
        popt, pcov = curve_fit(gauss, np.arange(-300, 300)/100, 1-(ccf/ccf[0]))
        # plt.plot(np.arange(-300, 300)/100, gauss(np.arange(-300, 300)/100, popt[0], popt[1], popt[2], popt[3]), color = 'r')
        # plt.show()
        rvs.append(popt[0])

    plt.figure()
    plt.scatter(phases, rvs)
    plt.show()

def read_in_frames(order1, filelist):
    frames = []
    errors = []
    frame_wavelengths = []
    sns = []
    max_sn = 0

    ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
    ### also adds the blaze corrected overlap regions from the previous and next order to thier subsequent lists
    global overlap_flux
    global overlap_wave
    global overlap_error
    global overlap_sns
    global pha
    overlap_flux = []
    overlap_wave = []
    overlap_error = []
    overlap_sns = []
    pha = []
    # plt.figure('spectra after blaze_correct')
    for file in filelist:
        if file_type == 's1d':
            fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct('s1d', 'order', order, file, directory, 'unmasked', run_name, 'y')
        else:
            fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct('e2ds', 'order', order1, file, directory, 'unmasked', run_name, 'y')
            if len(np.array(overlap[0, 1]))>0:
                overlap_flux.append(np.array(overlap[0, 1]))
                overlap_wave.append(np.array(overlap[0, 0]))
                overlap_error.append(np.array(overlap[0, 2]))
                overlap_sns.append(sn)
            if len(np.array(overlap[1, 1]))>0:
                overlap_flux.append(np.array(overlap[1, 1]))
                overlap_wave.append(np.array(overlap[1, 0]))
                overlap_error.append(np.array(overlap[1, 2]))
                overlap_sns.append(sn)

        ## saves the phase for the frame to the llist 'pha' 
        f = fits.open(file)
        try:phi = (((f[0].header['ESO DRS BJD'])-T)/P)%1
        except:phi = (((f[0].header['TNG DRS BJD'])-T)/P)%1
        f.close()
        pha.append(phi)

        ## reads in ccf RV - this was used to for test
        # try: ccf = fits.open(file.replace(file_type, 'ccf_K5'))
        # except: ccf = fits.open(file.replace(file_type, 'ccf_G2'))
        # try:ccf_rvs.append([ccf[0].header['ESO DRS CCF RV']])
        # except:ccf_rvs.append([ccf[0].header['TNG DRS CCF RV']])
        # plt.plot(wavelengths, fluxes)

        frame_wavelengths.append(list(wavelengths))
        frames.append(list(fluxes))
        errors.append(list(flux_error_order))
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

    frame_wavelengths = np.array(frame_wavelengths)
    frames = np.array(frames)
    errors = np.array(errors)

    # frame_ccf(frame_wavelengths, frames, np.array(pha)) # - ccf test

    # save a copy of unadjusted frames - this can then be used to check if the continuum corrected phases are shifted in comparison
    global frames_unadjusted
    frames_unadjusted = frames.copy()
    global frame_errors_unadjusted
    frame_errors_unadjusted = errors.copy()

    ### each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
    #plt.figure()
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

        # plt.plot(frame_wavelengths[n], frames[n])
    # plt.show()

    # frame_ccf(frame_wavelengths, frames, np.array(pha))

    ##adjusting overlap region in the same way that individual frames were adjusted
    for n in range(len(overlap_flux)):

        f2 = interp1d(overlap_wave[n], overlap_flux[n], kind = 'linear', bounds_error=False, fill_value=np.nan)
        div_frame = f2(reference_wave)

        #removing extrapolated regions
        idx = np.logical_and(reference_wave<np.max(overlap_wave[n]), reference_wave>np.min(overlap_wave[n]))
        div_frame=div_frame[idx]
        print(len(div_frame))
        reference_frame1 = reference_frame[idx]
        reference_wave1 = reference_wave[idx]
        div_frame = div_frame/reference_frame1

        ### fitting polynomial to div_frame
        coeffs=np.polyfit(reference_wave1, div_frame, 1)
        poly = np.poly1d(coeffs)
        fit = poly(overlap_wave[n])
        overlap_flux[n] = overlap_flux[n]/fit
        overlap_error[n] = overlap_error[n]/fit

        ## overlap regions are padded out with zeros to remain the same shape as the full spectra 
        ## these padded region have a weight of zero in the weighted mean and so do not count towards the combined spectrum.
        filled_flux = np.zeros((len(frame_wavelengths[0]),))
        filled_wave = np.zeros((len(frame_wavelengths[0]),))
        filled_error = np.zeros((len(frame_wavelengths[0]),))
        
        filled_flux[:len(overlap_flux[n])]=overlap_flux[n]
        filled_wave[:len(overlap_wave[n])]=overlap_wave[n]
        filled_error[:len(overlap_error[n])]=overlap_error[n]

        overlap_flux[n] = filled_flux
        overlap_wave[n] = filled_wave
        overlap_error[n] = filled_error

    return frame_wavelengths, frames, errors, sns, telluric_spec

## performs a rough continuum fit to a given spectrum by windowing the data and fitting a polynomail to the maximum data point in each window
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

        ## regions that go below zero are replaced and masked out - values <0 cause an error if this is fed into the LSD function
        idx = [flux_obs<=0]
        flux_obs[idx] = 0.00000000000000001
        new_errors[idx]=1000000000000000
        
        coeffs=coeffs[::-1]
        coeffs = list(coeffs)
        coeffs.append(cont_factor)
        coeffs = np.array(coeffs)
        
        return coeffs, flux_obs, new_errors, fit

### YOU ARE HERE
def combine_spec(wavelengths_f, spectra_f, errors_f, sns_f):

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
    #mdl = mdl+1
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
def residual_mask(wavelengths, data_spec_in, data_err, initial_inputs, telluric_spec, order):
    forward = model_func(initial_inputs, wavelengths)
    
    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)

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

    ##############################################
    #                  TELLURICS                 #   
    ##############################################

    ## shallow tellurics
    limit_wave =21/2.99792458e5 #needs multiplied by wavelength to give actual limit
    limit_pix = limit_wave/((max(wavelengths)-min(wavelengths))/len(wavelengths))  

    peaks = find_peaks(1-telluric_spec, height=0.01)
    for peak in peaks[0]:
        limit = int(limit_pix*wavelengths[peak])
        st = peak-limit
        end = peak+limit
        if st <0:st =0
        if end>len(data_err):end=len(data_err)
        for i in range(st, end):
            data_err[i] = 1000000000000000000

    ## deep telluric lines
    tell_lines = [3820.33, 3933.66, 3968.47, 4327.74, 4307.90, 4383.55, 4861.34, 5183.62, 5270.39, 5889.95, 5895.92, 6562.81, 7593.70, 8226.96]
    for line in tell_lines:
        limit = limit_wave*line +3
        idx = np.logical_and((line-limit)<=wavelengths, wavelengths<=(limit+line))
        data_err[idx] = 1000000000000000000

    residual_masks = tuple([data_err>=1000000000000000000])
    
    plt.figure()
    plt.plot(wavelengths, data_spec_in, 'k')
    plt.scatter(wavelengths[residual_masks], data_spec_in[residual_masks], color = 'r')
    plt.savefig('%sorder%s_masking_%s'%(save_path, order, run_name))
    plt.close()

    ###################################
    ###      sigma clip masking     ###
    ###################################

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

def task(all_frames, frames, counter):
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

    error[interp_mask_idx]=10000000000000000000
    # error[idx_overlap]=10000000000000000000
    
    poly2 = np.poly1d([ 1.23455938e-05, -1.58205893e-01,  6.75786834e+02, -9.62216391e+05])
    
    flux = (flux/mdl1)#/poly2(wavelengths)+0.5
    error = (error/mdl1)#/poly2(wavelengths)

    remove = tuple([flux<0])
    flux[remove]=1.
    error[remove]=10000000000000000000

    idx = tuple([flux>0])
    
    if len(flux[idx])==0:
        plt.plot(flux)
        plt.show()
        print('continuing... frame %s'%counter)
    
    else:
        frames[counter]=flux

        velocities1, profile1, profile_errors, alpha_here, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, flux, error, linelist, 'False', poly_ord, sn, order, run_name, velocities)

        profile_f = np.exp(profile1)
        profile_errors_f = np.sqrt(profile_errors**2/profile_f**2)
        profile_f = profile_f-1
        
        res = (flux) - (np.exp(convolve(profile1, alpha_here)))
        fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, sharex = True)
        ax[1].scatter(wavelengths, res, marker = '.')
        ax[0].plot(wavelengths, flux, 'r', alpha = 0.3, label = 'data')
        ax[0].plot(wavelengths, np.exp(convolve(profile1, alpha_here)), 'k', alpha =0.3, label = 'mcmc spec')
        residual_masks = tuple([error>=100000000000000])
        ax[0].scatter(wavelengths[residual_masks], flux[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        ax[0].legend(loc = 'lower right')
        z_line = [0]*len(wavelengths)
        ax[1].plot(wavelengths, z_line, '--')
        plt.savefig('%sorder%s_ACIDprofforward_%s%s'%(save_path, order, counter, run_name))

        all_frames[counter, order]=[profile_f, profile_errors_f]

        return all_frames, frames

def main():
    ## for real data
    global file_type, linelist, directory, save_path, run_name, ccf_rvs, overlap_wave, overlap_flux, overlap_error, overlap_sns, k_max, alpha, deltaphi, T, P
    file_type = 'e2ds'
    linelist = '/home/lsd/Documents/55 Cnc/55Cnc_lines.txt'
    directory = '/home/lsd/Documents/55 Cnc/group 2/*/*/*/'
    save_path = '/home/lsd/Documents/55 Cnc/results/'
    # linelist = '/home/lsd/Documents/55 Cnc/55Cnc_lines.txt'
    # directory = '/home/lsd/Documents/55 Cnc/N/'
    # save_path = '/home/lsd/Documents/55 Cnc/N/results/'
    P=0.737 #Cegla et al, 2006 - days
    T=2455962.0697 #Cegla et al, 2006
    t=1.6/24 #Torres et al, 2008
    deltaphi = t/(2*P)

    run_name = input('Input nickname for this version of code (for saving figures): ')

    ccf_rvs = []

    # order_range = np.arange(18,19)
    order_range = np.arange(28,29)

    filelist=findfiles(directory, file_type)
    print(filelist)
    phasess=[]
    poptss=[]
    global velocities
    velocities=np.arange(6+21, 50+21, 0.82)
    global all_frames
    all_frames = np.zeros((len(filelist), 71, 2, len(velocities)))
    global order
    for order in order_range:
        global poly_ord
        poly_ord = 3
        global frames, frame_wavelengths, frame_errors, sns
        frame_wavelengths, frames, frame_errors, sns, telluric_spec = read_in_frames(order, filelist)

        ## combines spectra from each frame (weighted based of S/N), returns to S/N of combined spec
        frame_wavelengths = np.array(frame_wavelengths)
        frames = np.array(frames)
        frame_errors = np.array(frame_errors)
        sns = np.array(sns)

        overlap_wave = np.array(overlap_wave)
        overlap_flux= np.array(overlap_flux)
        overlap_error = np.array(overlap_error)
        overlap_sns = np.array(overlap_sns)

        print(frame_wavelengths.shape)
        print(overlap_wave.shape)

        if file_type == 's1d':include_overlap = 'n'
        else:include_overlap = 'y'
        print(include_overlap)
        #include_overlap = 'n'
        if include_overlap =='y':
            fw = np.concatenate((frame_wavelengths.copy(), overlap_wave.copy()))
            f = np.concatenate((frames.copy(), overlap_flux.copy()))
            fe = np.concatenate((frame_errors.copy(), overlap_error.copy()))
            s = np.concatenate((sns.copy(), overlap_sns.copy()))
        else:
            fw = frame_wavelengths.copy()
            f = frames.copy()
            fe = frame_errors.copy()
            s = sns.copy()
        
        global sn
        wavelengths, fluxes, flux_error_order, sn = combine_spec(fw, f, fe, s)

        idx = np.isnan(fluxes)
        fluxes[idx]=1.
        flux_error_order[idx]=10000000000000000000.
        print("SN of combined spectrum, order %s: %s"%(order, sn))

        #### running MCMC #####

        ### getting the initial polynomial coefficents
        a = 2/(np.max(wavelengths)-np.min(wavelengths))
        b = 1 - a*np.max(wavelengths)
        poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes,  (wavelengths*a)+b, flux_error_order, poly_ord)

        #### getting the initial profile
        velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name, velocities)

        plt.figure('initial profile')
        plt.plot(velocities, profile)
        plt.savefig('%sorder%s_firstprofile_%s'%(save_path, order, run_name))

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
        global mask_idx
        yerr, model_inputs_resi, mask_idx = residual_mask(x, y, yerr, model_inputs, telluric_spec, order)

        ## setting number of walkers and their start values(pos)
        ndim = len(model_inputs)
        nwalkers= ndim*3
        rng = np.random.default_rng()

        ### starting values of walkers with indpendent variation
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

        ## the number of steps is how long it runs for - if it doesn't look like it's settling at a value try increasing the number of steps
        steps_no = 10000
        with mp.Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr), pool = pool)
            sampler.run_mcmc(pos, steps_no, progress=True)

        ## discarding all vales except the last 1000 steps.
        dis_no = int(np.floor(steps_no-1000))

        # plots the model for 'walks' of the all walkers for the first 5 profile points
        # samples = sampler.get_chain(discard = dis_no)
        # fig, axes = plt.subplots(len(samples[0, 0, :]), figsize=(10, 7), sharex=True)
        # for i in range(len(samples[0, 0, :10])):
        #     ax = axes[i]
        #     ax.plot(samples[:, :, i], "k", alpha=0.3)
        #     ax.set_xlim(0, len(samples))
        # axes[-1].set_xlabel("step number")
        # #plt.show()

        ## combining all walkers together
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
            for i in np.arange(k_max, len(sample)-1):
                mdl1 = mdl1+sample[i]*((a*x)+b)**(i-k_max)
            mdl1 = mdl1*sample[-1]
            plt.plot(x, mdl1, "C1", alpha=0.1)
            plt.plot(x, mdl, "g", alpha=0.1)
        plt.scatter(x, y, color = 'k', marker = '.', label = 'data')
        plt.xlabel("wavelengths")
        plt.ylabel("flux")
        plt.title('mcmc models and data')
        #plt.savefig('/home/lsd/Documents/mcmc_and_data.png')
        plt.savefig('%sorder%s_mc_mdl_%s'%(save_path, order, run_name))
        # plt.show()
        plt.close()
        #plt.show()

        ## getting the final profile and continuum values - median of last 1000 steps
        global poly_cos
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

        prof_flux = np.exp(profile)-1
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
        plt.savefig('%sorder%s_profile_%s'%(save_path, order, run_name))
        plt.close()
        #plt.show()

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

        plt.legend()
        plt.title('continuum from mcmc')
        plt.xlabel("wavelengths")
        plt.ylabel("flux")
        plt.savefig('%sorder%s_cont_%s'%(save_path, order, run_name))
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
        ax[0].scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        ax[0].legend(loc = 'lower right')
        ax[1].plot(x, residuals_2, '.')
        z_line = [0]*len(x)
        ax[1].plot(x, z_line, '--')
        plt.savefig('%sorder%s_forward_%s'%(save_path, order, run_name))
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
        plt.savefig('%sorder%s_final_profile_%s'%(save_path, order, run_name))
        plt.close()

        profile_f = np.exp(profile)
        profile_errors_f = np.sqrt(profile_err**2/profile_f**2)
        profile_f = profile_f-1

        fig, ax0 = plt.subplots()
        ax0.plot(velocities, profile_f, color = 'r', label = 'LSD')
        zero_line = [0]*len(velocities)
        ax0.plot(velocities, zero_line)
        ax0.plot(velocities, np.exp(model_inputs[:k_max])-1, label = 'initial')
        ax0.fill_between(velocities, profile_f-profile_errors_f, profile_f+profile_errors_f, alpha = 0.3, color = 'r')
        ax0.set_xlabel('velocities')
        ax0.set_ylabel('flux')
        ax0.legend()
        plt.close()

        print('Profile: %s\nContinuum Coeffs: %s\n'%(profile, poly_cos))

        task_part = partial(task, all_frames, frames)
        with mp.Pool(mp.cpu_count()) as pool:results=[pool.map(task_part, np.arange(len(frames)))]
        for i in range(len(frames)):
            all_frames[i]=results[0][i][0][i]
            frames[i]=results[0][i][1][i]

        # plt.figure()
        # for n in range(len(frames)):
        #     print(pha[n])
        #     plt.plot(frame_wavelengths[n], frames[n])
        # plt.show()
        # inp = input('Enter to continue...')
        #frame_ccf(frame_wavelengths, frames, np.array(pha))

    plt.close('all')

    # adding into fits files for each frame
    phases = []
    for frame_no in range(0, len(frames)):
        file = filelist[frame_no]
        fits_file = fits.open(file)
        try:phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
        except:phi = (((fits_file[0].header['TNG DRS BJD'])-T)/P)%1
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

            new_velocities = np.arange(6, 50, 0.82)
            f2 = interp1d(velocities+file[0].header['ESO DRS BERV'], all_frames[frame_no, order, 0], kind='linear', bounds_error=False, fill_value='extrapolate')
            profile = f2(new_velocities)
            profile_err = all_frames[frame_no, order, 1]

            hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
        hdu.writeto('%s%s_%s.fits'%(save_path, frame_no, run_name), output_verify = 'fix', overwrite = 'True')


main()
