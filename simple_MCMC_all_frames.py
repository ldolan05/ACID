import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import  fits
import emcee
#import corner
import LSD_func_faster as LSD
import time

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
fits_file = '/home/lsd/Documents/Starbase/novaprime/Documents/HD189733/August2007_master_out_ccfs.fits'
file_type = 'e2ds'
linelist = '/home/lsd/Documents/Starbase/novaprime/Documents/fulllinelist0001.txt'
#linelist = '/home/lsd/Documents/norm.txt'
#linelist = '/home/lsd/Documents/sme_linelist_result_48.txt'
#linelist = '/home/lsd/Documents/HD189733b_tloggmvsini.txt'
#linelist = '/home/lsd/Documents/HD189733_vary_abund.txt'
#linelist = '/home/lsd/Documents/fulllinelist018.txt'
#linelist = '/Users/lucydolan/Starbase/fulllinelist004.txt'
#linelist = '/home/lsd/Documents/fulllinelist004.txt'
directory_p = '/home/lsd/Documents/Starbase/novaprime/Documents/HD189733 old/HD189733/'
#directory = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/'
month = 'August2007'

run_name = input('Input nickname for this version of code (for saving figures): ')

ccf_rvs = []

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

    ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
    global overlap_flux
    global overlap_wave
    global overlap_error
    global overlap_sns
    overlap_flux = []
    overlap_wave = []
    overlap_error = []
    overlap_sns = []
    plt.figure('spectra after blaze_correct')
    for file in filelist:
        print('e2ds')
        # fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct('s1d', 'order', order, file.replace('e2ds', 's1d'), directory, 'unmasked', run_name, 'y')
        fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'unmasked', run_name, 'y')
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

        # plt.figure()
        # plt.title('overlaps stuff')
        # plt.plot(wavelengths, fluxes)
        # plt.plot(overlap[0, 0], overlap[0, 1])
        # plt.plot(overlap[1, 0], overlap[1, 1])
        # plt.show()
        # for i in range(len(fluxes)):
        #     if i ==0:
        #         fluxes[i] = fluxes[i]*(0.01/(2.99792458e5*(wavelengths[1]-wavelengths[0])))
        #     else:
        #         fluxes[i] = fluxes[i]*(0.01/(2.99792458e5*(wavelengths[i]-wavelengths[i-1])))
        try: ccf = fits.open(file.replace('e2ds', 'ccf_K5'))
        except: ccf = fits.open(file.replace('e2ds', 'ccf_G2'))
        ccf_rvs.append([ccf[0].header['ESO DRS CCF RV']])
        plt.plot(wavelengths, fluxes)
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

    # plt.figure('divided spectra (by reference frame)')
    ### each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
    popts = []
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

        print(len(binned_waves))
        # plt.figure()
        # plt.plot(binned_waves, binned)
        # plt.show()

        # plt.plot(frame_wavelengths[n], frames_unadjusted[n], color = 'b', label = 'unadjusted')
        # plt.figure()
        # plt.plot(frame_wavelengths[n], frames[n])
        # plt.show()

        mid_line = 4594.08
        ### fitting polynomial to div_frame
        coeffs=np.polyfit(binned_waves, binned, 2)
        poly = np.poly1d(coeffs)
        fit = poly(frame_wavelengths[n])
        frames[n] = frames[n]/fit
        errors[n] = errors[n]/fit
       
    #     id = np.logical_and(frame_wavelengths[n]<mid_line+0.5, frame_wavelengths[n]>mid_line-0.5)
    #     w = ((frame_wavelengths[n] - mid_line)*2.99792458e5)/frame_wavelengths[n]
    #     p = frames_unadjusted[n]/frames_unadjusted[n, 0]

    #     try:   
    #         popt, pcov = curve_fit(gauss, w[id], p[id])
    #         popts.append(popt[0])
    #     except: 
    #         plt.figure()
    #         plt.plot(w[id], p[id])
    #         plt.show()

    #     plt.plot(w[id], p[id])
    #     # plt.plot(frame_wavelengths[n], frames[n], color = 'g', label = 'adjusted')
    #     if n==0:
    #         plt.legend()
    # plt.xlim(np.min(w[id]), np.max(w[id]))
    
    # plt.figure()
    # plt.scatter(np.arange(len(frames)), popts - np.median(popts), label = 'e2ds')
    # plt.scatter(np.arange(len(frames)), ccf_rvs - np.median(ccf_rvs), label = 'ccf')
    # plt.legend()
    # plt.show()
    # # plt.show()
    plt.close()
    plt.figure()
    for n in range(len(frames)):
        plt.plot(reference_wave, frames[n], color = 'b')
    ##adjusting overlap region in the same way that individual frames were adjusted
    for n in range(len(overlap_flux)):
        # print(overlap_wave[n], overlap_flux[n])
        # print(len(overlap_wave[n]))
        # print(len(overlap_flux[n]))
        # plt.figure()
        # plt.plot(overlap_wave[n], overlap_flux[n])
        # plt.show()

        f2 = interp1d(overlap_wave[n], overlap_flux[n], kind = 'linear', bounds_error=False, fill_value=np.nan)
        div_frame = f2(reference_wave)

        # plt.figure()
        # plt.plot(reference_wave, reference_frame, label = 'ref')
        # plt.plot(overlap_wave[n], overlap_flux[n], label = 'overlap')
        # plt.plot(reference_wave, div_frame, label = 'div_frame')
        #plt.show()

        # print(len(reference_frame))
        # print(len(div_frame))

        # plt.figure()
        # plt.plot(reference_wave, div_frame, label = 'full div frame')
        dfcop = div_frame.copy()
        #remove extrapolated regions
        idx = np.logical_and(reference_wave<np.max(overlap_wave[n]), reference_wave>np.min(overlap_wave[n]))
        div_frame=div_frame[idx]
        print(len(div_frame))
        reference_frame1 = reference_frame[idx]
        reference_wave1 = reference_wave[idx]
        # plt.plot(reference_wave1, div_frame, label = 'part of div frame')
        # plt.legend()
        # plt.close()
        # plt.show()

        # plt.figure()
        # plt.plot(reference_wave1, reference_frame1, label = 'ref')
        # plt.plot(overlap_wave[n], overlap_flux[n], label = 'overlap')
        # plt.plot(reference_wave1, div_frame, label = 'div_frame')
        # plt.show()

        div_frame = div_frame/reference_frame1

        # plt.figure()
        # plt.plot(reference_wave1, div_frame)
        # plt.plot(reference_wave, dfcop/reference_frame)
        # plt.show()
        # ### creating windows to fit polynomial to
        # binned = np.zeros(int(len(div_frame)/2))
        # binned_waves = np.zeros(int(len(div_frame)/2))
        # for i in range(0, len(div_frame)-1, 2):
        #     pos = int(i/2)
        #     binned[pos] = (div_frame[i]+div_frame[i+1])/2
        #     binned_waves[pos] = (reference_wave[i]+reference_wave[i+1])/2
        
        # print(div_frame[div_frame==np.isnan])
        # print(len(div_frame))
        # plt.figure()
        # plt.plot(reference_wave1, div_frame)
        # plt.show()

        ### fitting polynomial to div_frame
        coeffs=np.polyfit(reference_wave1, div_frame, 1)
        poly = np.poly1d(coeffs)
        fit = poly(overlap_wave[n])
        overlap_flux[n] = overlap_flux[n]/fit
        overlap_error[n] = overlap_error[n]/fit

        filled_flux = np.zeros((len(frame_wavelengths[0]),))
        filled_wave = np.zeros((len(frame_wavelengths[0]),))
        filled_error = np.zeros((len(frame_wavelengths[0]),))
        
        filled_flux[:len(overlap_flux[n])]=overlap_flux[n]
        filled_wave[:len(overlap_wave[n])]=overlap_wave[n]
        filled_error[:len(overlap_error[n])]=overlap_error[n]

        overlap_flux[n] = filled_flux
        overlap_wave[n] = filled_wave
        overlap_error[n] = filled_error
        
        idx = tuple([filled_flux!=0])
        # plt.errorbar(filled_wave[idx], filled_flux[idx], filled_error[idx], color = 'k')
    # plt.close()

    return frame_wavelengths, frames, errors, sns, telluric_spec

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

                # print(np.max(flux))
                # plt.figure()
                # plt.scatter(waves, flux)
                # plt.show()

                clipped_flux.append(flux[len(flux)-1])
                clipped_waves.append(waves[len(waves)-1])
        
        ## trying to find bug - delete after
        # print(clipped_waves, clipped_flux/cont_factor, poly_ord)
        # plt.figure()
        # plt.plot(clipped_waves, clipped_flux)

        # plt.figure()
        # plt.plot(clipped_waves, clipped_flux/cont_factor)
        # plt.show()

        coeffs=np.polyfit(clipped_waves, clipped_flux/cont_factor, poly_ord)

        poly = np.poly1d(coeffs*cont_factor)
        fit = poly(wavelengths1)
        flux_obs = fluxes1/fit
        new_errors = errors1/fit

        idx = [flux_obs<=0]
        flux_obs[idx] = 0.00000000000000001
        new_errors[idx]=1000000000000000

        
        # fig = plt.figure('Continuum fit')
        # plt.title('Continuum fit')
        # plt.plot(wavelengths1, fluxes1, label = 'original')
        # plt.plot(wavelengths1, fit, label = 'fit')
        # plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        # plt.ylabel('flux')
        # plt.xlabel('wavelengths')
        # plt.legend()
        # plt.show()

        # fig = plt.figure('Continuum fit - adjusted')
        # plt.title('Continuum fit')
        # plt.plot(wavelengths1, flux_obs, label = 'adjusted')
        # plt.ylabel('flux')
        # plt.xlabel('wavelengths')
        # plt.legend()
        # plt.show()
        
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

        # print(spectra_f[n])
        # print(errors_f[n])

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
   
    # plt.figure()
    # plt.title('spectra for each frame and combined spectrum')
    # for n in range(0, len(spectra_f)):
    #     #plt.errorbar(wavelengths, frames[n], yerr=errors[n], ecolor = 'k')
    #     plt.plot(reference_wave, spectra_f[n], label = '%s'%n)
    # plt.errorbar(reference_wave, spectrum_f, label = 'combined', color = 'b', yerr = spec_errors_f, ecolor = 'k')
    # #plt.xlim(np.min(frames[n])-1, np.max(frames[n])+1)
    # #plt.legend()
    # plt.show()
    
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

    # print(inputs)

    # plt.figure()
    # plt.plot(x, mdl1, 'r')
    # plt.plot(x, y)
    # plt.show()
    
    mdl = mdl * mdl1
    '''
    plt.figure()
    plt.plot(x, mdl, 'r')
    plt.plot(x, y)
    plt.show()
    '''
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

        # plt.figure()
        # plt.plot(velocities, z)

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

        # plt.scatter(v_cont, z_cont)
        '''
        plt.figure()
        plt.plot(velocities, theta[:k_max])
        plt.scatter(v_cont, z_cont)
        plt.show()
        '''
        # calcualte gaussian probability for each point in continuum
        # print(z_cont)
        # print((1/np.sqrt(2*np.pi*p_var**2)))
        # print(np.exp(-0.5*(abs(z_cont)/p_var)**2))
        # print((1/np.sqrt(2*np.pi*p_var**2))*np.exp(-0.5*(z_cont/p_var)**2))
        # print(z_cont/p_var)
        # print(-0.5*(z_cont/p_var)**2)
        # print(np.exp(-0.5*(z_cont/p_var)**2))

        # z_cont = []
        # v_cont = []
        # for i in range(10, 20):
        #         z_cont.append(np.exp(z[len(z)-i-1])-1)
        #         v_cont.append(velocities[len(velocities)-i-1])
        #         z_cont.append(np.exp(z[i])-1)
        #         v_cont.append(velocities[i])

        # z_cont = np.array(z_cont)

        # p_pent = p_pent+np.sum((np.log((1/np.sqrt(2*np.pi*0.01**2)))-0.5*(z_cont/0.01)**2))
        # print(p_pent)
        # plt.show()
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
def residual_mask(wavelengths, data_spec_in, data_err, initial_inputs, telluric_spec):
    #residuals=((data_spec+1)/(forward+1))/data_err
    #residuals=abs(((data_spec)-(forward))/(forward))
    forward = model_func(initial_inputs, wavelengths)
    data_spec = data_spec_in.copy()

    ### (roughly) normalise the data (easier to set standard threshold for residuals)
    #residuals=(data_spec/data_spec[0]-forward/forward[0])
    # residuals = residuals-np.max(residuals)
    # plt.figure()
    # plt.plot(wavelengths, data_spec_in)
    # plt.plot(wavelengths, forward)
    # plt.show()

    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(x)

    # plt.figure()
    mdl1=0
    for i in range(k_max,len(initial_inputs)-1):
        mdl1 = mdl1 + (initial_inputs[i]*((wavelengths*a)+b)**(i-k_max))

    mdl1 = mdl1 * initial_inputs[-1]
    # plt.plot(wavelengths, data_spec_in/mdl1)
    # plt.show()

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

    # residuals = data_spec_in/forward-np.median(data_spec_in/forward)
    # plt.figure()
    # plt.plot(wavelengths, data_spec_in)
    # plt.scatter(wavelengths[data_err>=10000000000000000000], data_spec_in[data_err>=10000000000000000000])
    # plt.show()

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

    # plt.figure()
    # plt.plot(wavelengths, data_spec_in)
    # plt.scatter(wavelengths[data_err>=10000000000000000000], data_spec_in[data_err>=10000000000000000000])
    
    # plt.figure()
    # plt.scatter(wavelengths, residuals)
    # plt.show()
    ##############################################
    #                  TELLURICS                 #   
    ##############################################

    ## masking tellurics
    ## use find peaks on telluric spec to find the lines
    ## mask out region around these telluric lines
    ## these will take care of shallow lines
    ## create seperate section to mask out very deep lines (think there is a section in LSD that deals with these - Franhofer lines (definetly not spelt like that))

    limit_wave =21/2.99792458e5 #needs multiplied by wavelength to give actual limit
    limit_pix = limit_wave/((max(wavelengths)-min(wavelengths))/len(wavelengths))
    
    # plt.figure()
    # plt.plot(wavelengths, 1-telluric_spec)

    peaks = find_peaks(1-telluric_spec, height=0.01)
    # print(peaks[0])
    for peak in peaks[0]:
        limit = int(limit_pix*wavelengths[peak])
        # print('the limit is...')
        # print(limit)
        st = peak-limit
        end = peak+limit
        if st <0:st =0
        if end>len(data_err):end=len(data_err)
        for i in range(st, end):
            # print(wavelengths[i])
            data_err[i] = 1000000000000000000

    tell_lines = [3820.33, 3933.66, 3968.47, 4327.74, 4307.90, 4383.55, 4861.34, 5183.62, 5270.39, 5889.95, 5895.92, 6562.81, 7593.70, 8226.96]
    for line in tell_lines:
        limit = limit_wave*line +3
        idx = np.logical_and((line-limit)<=wavelengths, wavelengths<=(limit+line))
        data_err[idx] = 1000000000000000000

    residual_masks = tuple([data_err>=1000000000000000000])
    
    plt.figure()
    plt.plot(wavelengths, data_spec_in, 'k')
    # print(residual_masks)
    # print(wavelengths[residual_masks]) 
    # print(data_spec_in[residual_masks])
    plt.scatter(wavelengths[residual_masks], data_spec_in[residual_masks], color = 'r')
    plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/order%s_masking_%s'%(order, run_name))
    plt.close()

    ###################################
    ###      sigma clip masking     ###
    ###################################

    m = np.median(residuals)
    sigma = np.std(residuals)
    a = 1

    upper_clip = m+a*sigma
    lower_clip = m-a*sigma

    # print(lower_clip)
    # print(upper_clip)

    rcopy = residuals.copy()

    idx1 = tuple([rcopy<=lower_clip])
    idx2 = tuple([rcopy>=upper_clip])
    # print(residuals[idx1])
    # print(residuals[idx2])

    plt.figure()
    plt.plot(wavelengths, residuals, 'k')
    plt.plot(wavelengths, [0]*len(wavelengths), '--')
    plt.scatter(wavelengths[idx1], residuals[idx1], color = 'r')
    plt.scatter(wavelengths[idx2], residuals[idx2], color = 'r')
    plt.close()
    #plt.show()

    data_err[idx1]=10000000000000000000
    data_err[idx2]=10000000000000000000

    poly_inputs, bin, bye, fit=continuumfit(data_spec_in,  (wavelengths*a)+b, data_err, poly_ord)
    velocities1, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, bin, bye, linelist, 'False', poly_ord, sn, order, run_name, velocities)

    ## uncomment if you would like to keep sigma clipping masking in for final LSD run 
    residual_masks = tuple([data_err>=1000000000000000000])

    return data_err, np.concatenate((profile, poly_inputs)), residual_masks

def task(all_frames, counter):
    flux = frames[counter]
    error = frame_errors[counter]
    wavelengths = frame_wavelengths[counter]
    sn = sns[counter]
    # idx_overlap = idx_overlaps[counter]

    # a = 2/(np.max(wavelengths)-np.min(wavelengths))
    # b = 1 - a*np.max(wavelengths)
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
    # yerr_resi, model_inputs_resi, mask_idx = residual_mask(wavelengths, flux, error, model_inputs, telluric_spec)

    error[interp_mask_idx]=10000000000000000000
    # error[idx_overlap]=10000000000000000000

    ## normalise
    '''
    error = (error)/(np.max(flux)-np.min(flux))
    flux = (flux - np.min(flux))/(np.max(flux)-np.min(flux))
    '''
    flux_b = flux
    error_b = error

    # plt.figure('flux before continuum correction')
    # plt.plot(wavelengths, flux)

    # plt.figure('flux before continuum correction - with continuum fit')
    # plt.plot(wavelengths, flux)
    # plt.plot(wavelengths, mdl1)

    conts = []
    for ind in inds:
        sample = flat_samples[ind]
        mdl = model_func(sample, wavelengths)
        #mdl = model_func(sample, x)
        #mdl = mdl[idx]
        mdl1 = 0
        for i in np.arange(k_max, len(sample)-1):
            mdl1 = mdl1+sample[i]*((a*x)+b)**(i-k_max)
        mdl1 = mdl1*sample[-1]
        conts.append(mdl1)
    
    continuum_error = np.std(np.array(conts), axis = 0)

    # plt.show()

    error = (error/flux) + (continuum_error/mdl1)
    flux = flux/mdl1
    error  = flux*error

    remove = tuple([flux<0])
    flux[remove]=1.
    error[remove]=10000000000000000000


    idx = tuple([flux>0])
    
    if len(flux[idx])==0:
        plt.plot(flux)
        plt.show()
        print('continuing... frame %s'%counter)
    
    else:
        #plt.plot(wavelengths, flux, label = '%s'%counter)
        #print(counter, order)

        velocities1, profile1, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, flux, error, linelist, 'False', poly_ord, sn, order, run_name, velocities)

        p = np.exp(profile1)-1
        popt, pcov = curve_fit(gauss, velocities, p)
        popts_new.append(popt[0])
        

        profile_f = np.exp(profile1)
        profile_errors_f = np.sqrt(profile_errors**2/profile_f**2)
        profile_errors_f = profile_errors*profile_f
        profile_f = profile_f-1

        # print(profile)
        # print(profile_f)

        # inp = input('(od profile and flux profile above) Enter to continue...')
        #print(profile_f)
        all_frames[counter, order]=[profile_f, profile_errors_f]
        #print(all_frames[counter, order, 0])

        #plt.show()
        #plt.legend()
        #plt.show()

        # count = 0
        # plt.figure()
        # plt.title("HARPS CCFs")
        # file_list = findfiles(directory, 'ccf')
        # for file in file_list[:-1]:
        #     ccf = fits.open(file)
        #     ccf_spec = ccf[0].data[order]
        #     velocities_ccf=ccf[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*ccf[0].header['CDELT1']
        #     plt.plot(velocities_ccf, ccf_spec/ccf_spec[0]-1, label = '%s'%count)
        #     count +=1
        # plt.legend()
        # plt.ylabel('flux')
        # plt.xlabel('velocities km/s')
        # #plt.show()

        # berv_corr = ccf[0].header['ESO DRS BERV']

        # plt.figure()
        # adjusted_velocities = velocities+berv_corr
        # f2 = interp1d(adjusted_velocities, profile_f, kind='linear', bounds_error=False, fill_value=np.nan)
        # velocity_grid = np.linspace(-15,15,len(profile_f))
        # adjusted_prof = f2(velocity_grid)
        # plt.plot(velocity_grid, adjusted_prof, label = 'adjusted_profile')
        # plt.plot(velocities_ccf, ccf_spec/ccf_spec[0]-1, label = 'ccf') 
        # plt.plot(velocities, profile_f, label = 'profile (pre-adjusted)')
        # plt.show()

        # plt.figure()
        # plt.title('order %s, LSD profiles'%order)
        # no=0

        # print(velocities)
        
        # ##########################################################################
        # ##########################################################################
        # ### testing rvs against ccfs - take out after test

        # #for profile in profiles:
        # plt.figure('profiles')
        # plt.plot(velocities, np.exp(profile)-1, label = '%s'%counter)
        #     # p = np.exp(profile)-1
        #     # popt, pcov = curve_fit(gauss, velocities[15:-15], p[15:-15])
        #     # print(popt[0], ccf_rvs[no])
            
        #     #plt.show()
        # # plt.legend()
        # plt.ylabel('flux')
        # plt.xlabel('velocities km/s')
        # if counter == len(frames)-1:
        #     print('I THINK ITS THE LAST FRAME')
        #     print(counter)
        #     print(len(frames))
        #     plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/profiles/order%s_FINALprof_%s'%(order, run_name))
        #     plt.close()
        #     #plt.show()

        # # inp = input('Check rvs stated above ^^')
        # #############################################################################
        # #############################################################################
        
        # mcmc_mdl = model_func(np.concatenate((profile, poly_cos)), wavelengths)
        # f2 = interp1d(velocities_ccf, ccf_spec/ccf_spec[0]-1, kind='linear', bounds_error=False, fill_value=np.nan)
        # ccf_profile = f2(velocities)
        # ccf_mdl = model_func(np.concatenate((ccf_profile, poly_cos)), wavelengths)


        # residuals_2 = (flux_b) - (mcmc_mdl)

        # fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'final LSD forward and true model', sharex = True)
        # non_masked = tuple([error_b<1000000000000000000])
        # #ax[0].plot(x, y+1, color = 'r', alpha = 0.3, label = 'data')
        # #ax[0].plot(x[non_masked], mcmc_mdl[non_masked]+1, color = 'k', alpha = 0.3, label = 'mcmc spec')
        # ax[1].scatter(wavelengths[non_masked], residuals_2[non_masked], marker = '.')
        # ax[0].plot(wavelengths, flux_b, 'r', alpha = 0.3, label = 'data')
        # ax[0].plot(wavelengths, mcmc_mdl, 'k', alpha =0.3, label = 'LSD spec')
        # ax[0].plot(wavelengths, ccf_mdl, 'g', alpha =0.3, label = 'CCF spec')
        # residual_masks = tuple([error_b>=1000000000000000000])

        # print(wavelengths[residual_masks])
        # #residual_masks = tuple([yerr>10])
        # ax[0].scatter(wavelengths[residual_masks], flux_b[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        # ax[0].legend(loc = 'lower right')
        # #ax[0].set_ylim(0, 1)
        # #plotdepths = -np.array(line_depths)
        # #ax[0].vlines(line_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
        # #ax[1].plot(x, residuals_2, '.')
        # #ax[1].scatter(wavelengths[residual_masks], residuals_2[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        # #z_line = [0]*len(wavelengths)
        # #ax[1].plot(wavelengths, z_line, '--')
        # plt.savefig('/home/lsd/Documents/LSD_Figures/forward_models/order%s_FINALforward_%s'%(order, run_name))
        #plt.show()
        return all_frames

months1 = ['August2007']
months = ['July2007', 'August2007', 'July2006', 'Sep2006']
#filelist = filelist[0]
order_range = np.arange(10,70)
# order_range = np.arange(28,29)

P=2.21857567 #Cegla et al, 2006 - days
T=2454279.436714 #Cegla et al, 2006
t=0.076125 #Torres et al, 2008
deltaphi = t/(2*P)

month_spec = []
for month in months:
    directory = '%s%s/'%(directory_p, month)
    print(directory)
    filelist=findfiles(directory, file_type)
    print(filelist)
    phasess=[]
    poptss=[]
    global velocities
    velocities=np.arange(-25, 25, 0.82)
    global all_frames
    all_frames = np.zeros((len(filelist), 71, 2, len(velocities)))
    for order in order_range:

        #plt.close('all')
        poly_ord = 3

        ### read in spectra for each frame, along with S/N of each frame
        # for file in filelist:
        #     fits_file = fits.open(file)
        #     phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
        #     print(phi)
        #     print(file, order)
        #     fluxes, wavelengths, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory1, 'unmasked', run_name, 'y')
        #     fluxes1, wavelengths1, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory2, 'unmasked', run_name, 'n')

        frame_wavelengths, frames, frame_errors, sns, telluric_spec = read_in_frames(order, filelist)

        #  ################## TEST - get the RV from each frame - see how it compares to CCF rv and unadjusted spectrum rv #####################
        # plt.figure('rvs after basic continuum fit')
        # popts = []
        # popts_unad = []
        # line_popts = []
        # line_popts_unad = []

        # for n in range(len(frame_wavelengths)):
        #     wavelengths1 = frame_wavelengths[n]
        #     a = 2/(np.max(wavelengths1)-np.min(wavelengths1))
        #     b = 1 - a*np.max(wavelengths1)
        #     poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(frames[n],  wavelengths1, frame_errors[n], poly_ord)

        #     #### getting the initial profile
        #     velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths1, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sns[n], order, run_name)

        #     p = np.exp(profile)-1
        #     # p=profile
        #     popt, pcov = curve_fit(gauss, velocities, p)
        #     popts.append(popt[0])
        #     # plt.figure()
        #     # plt.plot(velocities, p, color = 'k')
        #     # plt.plot(velocities, gauss(velocities, popt[0], popt[1], popt[2], popt[3]), color = 'r')
        #     print(popt[0], ccf_rvs[n])
            
        #     ## test for all lines rv #####
        #     # print('fitting to un-continuum adjusted spectrum')
        #     mid_lines = [4575.7795, 4580.4365, 4584.5967, 4588.7294, 4594.1158, 4597.7472, 4601.134 , 4606.2251, 4616.2463, 4621.941 ]
            
        #     # line_popts1 = []
        #     # for mid_line in mid_lines:
        #         ### fitting polynomial to div_frame
        #     mid_line = mid_lines[0]
        #     id = np.logical_and(frame_wavelengths[n]<mid_line+0.5, frame_wavelengths[n]>mid_line-0.5)
        #     w = ((frame_wavelengths[n] - mid_line)*2.99792458e5)/frame_wavelengths[n]
        #     p = fluxes1
        #     try:   
        #         popt, pcov = curve_fit(gauss, w[id], p[id])
        #         line_popts.append(popt[0])
        #     except: 
        #         print(w[id])
        #         print(p[id])
        #         plt.figure()
        #         plt.plot(w[id], p[id])
        #         plt.show()

        #     # line_popts.append(line_popts1)
        #     ## end of sub-test ##

        #     # poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(frames_unadjusted[n],  wavelengths1, frame_errors_unadjusted[n], poly_ord)

        #     # #### getting the initial profile
        #     # velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths1, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sns[n], order, run_name)

        #     # # p=profile
        #     # p = np.exp(profile)-1
        #     # popt, pcov = curve_fit(gauss, velocities, p)
        #     # popts_unad.append(popt[0])
        #     # print(popt[0])

        #     # mid_line = 4594.08
        #     # ### fitting polynomial to div_frame
        #     # id = np.logical_and(frame_wavelengths[n]<mid_line+0.5, frame_wavelengths[n]>mid_line-0.5)
        #     # w = ((frame_wavelengths[n] - mid_line)*2.99792458e5)/frame_wavelengths[n]
        #     # p = frames_unadjusted[n]/frames_unadjusted[n,0]

        #     # try:   
        #     #     popt, pcov = curve_fit(gauss, w[id], p[id])
        #     #     line_popts_unad.append(popt[0])
        #     # except: 
        #     #     plt.figure()
        #     #     plt.plot(w[id], p[id])
        #     #     plt.show()
    
        # # line_popts = np.array(line_popts)

        # # plt.figure()
        # # for i in range(len(mid_lines)):    
        # #     plt.scatter(np.arange(len(frames)), line_popts[:, i] - np.median(line_popts[:, i]), label = 'line e2ds - basic continuum - line %s'%i)
        # # plt.scatter(np.arange(len(frames)), ccf_rvs - np.median(ccf_rvs), label = 'ccf')
        # # plt.legend()
        # # plt.show()

        # plt.figure()
        # plt.title('RV Curve for Synthetic e2ds Spectrum')
        # plt.scatter(np.arange(len(frames)), line_popts, label = 'RV from line')
        # plt.scatter(np.arange(len(frames)), popts, label = 'RV from LSD profile')
        # plt.scatter(np.arange(len(frames)), ccf_rvs, label = 'RV from CCF (input into synthetic spectrum)')
        # plt.xlabel('Frame Number')
        # plt.ylabel('RV')
        # plt.legend()
        # #plt.show()

        # plt.figure()
        # plt.title('RV-median(RVs) Curve for Synthetic e2ds Spectrum')
        # plt.scatter(np.arange(len(frames)), line_popts-np.median(line_popts), label = 'RV from line')
        # plt.scatter(np.arange(len(frames)), popts-np.median(popts), label = 'RV from LSD profile')
        # plt.scatter(np.arange(len(frames)), ccf_rvs-np.median(ccf_rvs), label = 'RV from CCF (input into synthetic spectrum)')
        # plt.xlabel('Frame Number')
        # plt.ylabel('RV - median(RVs)')
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.title('RV difference for Synthetic e2ds Spectrum')
        # popts = np.array(popts)
        # ccf_rvs = np.array(ccf_rvs)
        # ccf_rvs = ccf_rvs.reshape(popts.shape)
        # plt.scatter(np.arange(len(frames)), (popts-np.median(popts))-(ccf_rvs-np.median(ccf_rvs)), label = 'RV Diff from line')
        # #plt.scatter(np.arange(len(frames)), popts-np.median(popts), label = 'RV from LSD profile')
        # # plt.scatter(np.arange(len(frames)), ccf_rvs-np.median(ccf_rvs), label = 'RV from CCF (input into synthetic spectrum)')
        # plt.xlabel('Frame Number')
        # plt.ylabel('RV Difference (LSD RV - CCF RV)')
        # plt.legend()
        # plt.show()

        # plt.figure()
        # plt.scatter(np.arange(len(frames)), line_popts_unad - np.median(line_popts_unad), label = 'line e2ds - basic continuum unadjusted')
        # plt.scatter(np.arange(len(frames)), ccf_rvs - np.median(ccf_rvs), label = 'ccf')
        # plt.legend()
    
        # plt.figure()
        # plt.scatter(np.arange(len(popts)), popts-np.median(popts), label = 'basic continuum')
        # plt.scatter(np.arange(len(ccf_rvs)), ccf_rvs-np.median(ccf_rvs), label = 'ccf rvs')
        # plt.legend()

        # plt.figure()
        # plt.scatter(np.arange(len(popts_unad)), popts_unad-np.median(popts_unad), label = 'basic continuum, unadjusted')
        # plt.scatter(np.arange(len(ccf_rvs)), ccf_rvs-np.median(ccf_rvs), label = 'ccf rvs')
        # plt.legend()
        # plt.show()

        ############# END OF TEST ####################
        # inp = input('END of Test 1 - rvs with basic continuum fit')

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

        include_overlap = 'y'
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
            
        wavelengths, fluxes, flux_error_order, sn = combine_spec(fw, f, fe, s)

        # plt.figure()
        # plt.errorbar(wavelengths, fluxes, flux_error_order, color = 'b', ecolor = 'k')
        # plt.show()
        # plt.figure()
        # plt.plot(wavelengths, fluxes)

        # plt.figure()
        # plt.plot(wavelengths, flux_error_order)
        # plt.show()

        print("SN of combined spectrum, order %s: %s"%(order, sn))
        
        # month_spec.append([wavelengths, wavelengths1, fluxes, fluxes1])
                        #corr_wave, uncorr wave, corr flux, uncorr flux 
        ## normalise
        '''
        flux_error_order = (flux_error_order)/(np.max(fluxes)-np.min(fluxes))
        fluxes = (fluxes - np.min(fluxes))/(np.max(fluxes)-np.min(fluxes))
        '''

        # ################## TEST - get the RV from each frame - see how it compares to CCF rv and unadjusted spectrum rv #####################
        # plt.figure('rvs after basic continuum fit')
        # popts = []
        # popts_unad = []
        # line_popts = []
        # line_popts_unad = []
        # for n in range(len(frame_wavelengths)):
        #     wavelengths1 = frame_wavelengths[n]
        #     a = 2/(np.max(wavelengths1)-np.min(wavelengths1))
        #     b = 1 - a*np.max(wavelengths1)
        #     poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(frames[n],  wavelengths1, frame_errors[n], poly_ord)

        #     #### getting the initial profile
        #     velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths1, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sns[n], order, run_name)

        #     p = np.exp(profile)-1
        #     # p=profile
        #     popt, pcov = curve_fit(gauss, velocities, p)
        #     popts.append(popt[0])
        #     # plt.figure()
        #     # plt.plot(velocities, p, color = 'k')
        #     # plt.plot(velocities, gauss(velocities, popt[0], popt[1], popt[2], popt[3]), color = 'r')
        #     print(popt[0], ccf_rvs[n])
            
        #     print('fitting to un-continuum adjusted spectrum')
        #     mid_line = 4594.08
        #     ### fitting polynomial to div_frame
        #     id = np.logical_and(frame_wavelengths[n]<mid_line+0.5, frame_wavelengths[n]>mid_line-0.5)
        #     w = ((frame_wavelengths[n] - mid_line)*2.99792458e5)/frame_wavelengths[n]
        #     p = frames[n]/frames[n,0]

        #     try:   
        #         popt, pcov = curve_fit(gauss, w[id], p[id])
        #         line_popts.append(popt[0])
        #     except: 
        #         plt.figure()
        #         plt.plot(w[id], p[id])
        #         plt.show()

        #     poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(frames_unadjusted[n],  wavelengths1, frame_errors_unadjusted[n], poly_ord)

        #     #### getting the initial profile
        #     velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths1, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sns[n], order, run_name)

        #     # p=profile
        #     p = np.exp(profile)-1
        #     popt, pcov = curve_fit(gauss, velocities, p)
        #     popts_unad.append(popt[0])
        #     print(popt[0])

        #     mid_line = 4594.08
        #     ### fitting polynomial to div_frame
        #     id = np.logical_and(frame_wavelengths[n]<mid_line+0.5, frame_wavelengths[n]>mid_line-0.5)
        #     w = ((frame_wavelengths[n] - mid_line)*2.99792458e5)/frame_wavelengths[n]
        #     p = frames_unadjusted[n]/frames_unadjusted[n,0]

        #     try:   
        #         popt, pcov = curve_fit(gauss, w[id], p[id])
        #         line_popts_unad.append(popt[0])
        #     except: 
        #         plt.figure()
        #         plt.plot(w[id], p[id])
        #         plt.show()
    
        # plt.figure()
        # plt.scatter(np.arange(len(frames)), line_popts - np.median(line_popts), label = 'line e2ds - basic continuum')
        # plt.scatter(np.arange(len(frames)), ccf_rvs - np.median(ccf_rvs), label = 'ccf')
        # plt.legend()

        # plt.figure()
        # plt.scatter(np.arange(len(frames)), line_popts_unad - np.median(line_popts_unad), label = 'line e2ds - basic continuum')
        # plt.scatter(np.arange(len(frames)), ccf_rvs - np.median(ccf_rvs), label = 'ccf')
        # plt.legend()
    
        # plt.figure()
        # plt.scatter(np.arange(len(popts)), popts-np.median(popts), label = 'basic continuum')
        # plt.scatter(np.arange(len(ccf_rvs)), ccf_rvs-np.median(ccf_rvs), label = 'ccf rvs')
        # plt.legend()

        # plt.figure()
        # plt.scatter(np.arange(len(popts_unad)), popts_unad-np.median(popts_unad), label = 'basic continuum, unadjusted')
        # plt.scatter(np.arange(len(ccf_rvs)), ccf_rvs-np.median(ccf_rvs), label = 'ccf rvs')
        # plt.legend()
        # plt.show()

        # ############# END OF TEST ####################
        # inp = input('END of Test 2 - rvs with basic continuum fit - after combine spec function')


        #### running the MCMC #####

        ### getting the initial polynomial coefficents
        a = 2/(np.max(wavelengths)-np.min(wavelengths))
        b = 1 - a*np.max(wavelengths)
        poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes,  (wavelengths*a)+b, flux_error_order, poly_ord)

        print(poly_inputs)

        #### getting the initial profile
        velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name, velocities)

        # plt.figure('forward model after first LSD')
        # plt.plot(wavelengths, np.log(fluxes1))
        # plt.plot(wavelengths, np.dot(alpha, profile))
        # plt.show()

        # plt.figure('initial profile')
        # plt.plot(velocities, profile)
        # plt.show()

        # plt.figure()
        # plt.title('SPEC BEFORE 1st LSD')
        # plt.plot(wavelengths, fluxes1, 'r')
        # plotdepths = 1 - np.array(continuum_flux)
        # plt.vlines(continuum_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
        # plt.show()

        # plt.figure()
        # plt.plot(velocities, profile)
        # plt.title('LSD - basic continuuum correction')
        # plt.xlabel('Velocities (km/s)')
        # plt.ylabel('Normalised Flux')
        # plt.show()
            

        ## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
        j_max = int(len(fluxes))
        k_max = len(profile)

        model_inputs = np.concatenate((profile, poly_inputs))

        ## setting x, y, yerr for emcee
        x = wavelengths
        y = fluxes
        yerr = flux_error_order

        ## setting these normalisation factors as global variables - used in the figures below
        a = 2/(np.max(x)-np.min(x))
        b = 1 - a*np.max(x)
        '''
        plt.figure()
        plt.title('continuum(--) fit on top of data')
        plt.plot(x, y)
        check_fit=0
        for i in range(k_max,len(model_inputs)):
            check_fit = check_fit + (model_inputs[i]*((x*a)+b)**(i-k_max))
        plt.plot(x, check_fit, '--')
        plt.show()
        '''
        ## parameters for working out continuum points in the LSD profile - if using penalty function
        p_var = 0.001
        v_min = -10
        v_max = 10


        ##plots
        forward = model_func(model_inputs, x)
        # plt.figure(figsize=(16,9))
        # plt.title('LSD optical depth forward model')
        # plt.plot(x, y, 'k', alpha = 0.3, label = 'data')
        # plt.plot(x, forward, 'r', alpha =0.3, label = 'forward model')
        # plt.vlines(continuum_waves, plotdepths, 1, label = 'line depths from VALD linelist', color = 'c', alpha = 0.2)
        # #plt.xlim(5460, 5480)
        # plt.legend()
        # plt.show()

        #masking based off residuals
        yerr_unmasked = yerr
        yerr, model_inputs_resi, mask_idx = residual_mask(x, y, yerr, model_inputs, telluric_spec)

        # #################### TEST - get the RV from each frame after residual masking - see how it compares to CCF rv and unadjusted spectrum rv #####################
        # plt.figure('rvs after basic continuum fit and residual mask')
        # popts = []
        # popts_unad = []
        # for n in range(len(frame_wavelengths)):
        #     wavelengths1 = frame_wavelengths[n]
        #     a = 2/(np.max(wavelengths1)-np.min(wavelengths1))
        #     b = 1 - a*np.max(wavelengths1)

        #     mask_pos = np.ones(reference_wave.shape)
        #     mask_pos[mask_idx]=10000000000000000000
        #     f2 = interp1d(reference_wave, mask_pos, bounds_error = False, fill_value = np.nan)
        #     interp_mask_pos = f2(wavelengths)
        #     interp_mask_idx = tuple([interp_mask_pos>=10000000000000000000])

        #     err = frame_errors[n]
        #     err[interp_mask_idx] = 10000000000000000000
        
        #     poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(frames[n],  (wavelengths1*a)+b, err, poly_ord)

        #     #### getting the initial profile
        #     velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths1, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name)

        #     p = np.exp(profile)-1
        #     popt, pcov = curve_fit(gauss, velocities[15:-15], p[15:-15])
        #     popts.append(popt[0])
        #     print(popt[0], ccf_rvs[n])
            
        #     err = frame_errors_unadjusted[n]
        #     err[interp_mask_idx] = 10000000000000000000

        #     poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(frames_unadjusted[n],  (wavelengths1*a)+b, err, poly_ord)

        #     #### getting the initial profile
        #     velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths1, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name)

        #     p = np.exp(profile)-1
        #     popt, pcov = curve_fit(gauss, velocities[15:-15], p[15:-15])
        #     popts_unad.append(popt[0])
        #     print(popt[0])

            
        # plt.scatter(np.arange(len(popts)), popts)
        # plt.scatter(np.arange(len(ccf_rvs)), ccf_rvs)
        # plt.show()
        # ############## END OF TEST ####################
        # inp = input('END of Test 2 - rvs with basic continuum fit and masking applied')


        ##masking frames also
        #mask_idx = tuple([yerr>1000000000000000000])
        #frame_errors[:, idx]=1000000000000000000

        ## setting number of walkers and their start values(pos)
        ndim = len(model_inputs)
        nwalkers= ndim*3
        rng = np.random.default_rng()

        '''
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
        print('DEPENDENT ON INPUT')
        print(pos)

        '''
        print('MODEL INPUTS')
        model_func(model_inputs, x)

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

        print('INDEPENDENT OF INPUT')
        print(pos)

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

        # plt.figure()
        # plt.plot(x, y)
        # plt.show()

        # running the mcmc using python package emcee

        # plt.figure()
        # plt.errorbar(x, y, yerr, color = 'b', ecolor = 'k')
        # plt.show()
        # with mp.Pool(mp.cpu_count()) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
        sampler.run_mcmc(pos, steps_no, progress=True)
        
        idx = tuple([yerr<=10000000000000000000])

        t1 = time.time()

        ## discarding all vales except the last 1000 steps.
        dis_no = int(np.floor(steps_no-1000))

        # plots the model for 'walks' of the all walkers for the first 5 profile points
        samples = sampler.get_chain(discard = dis_no)
        fig, axes = plt.subplots(len(samples[0, 0, :]), figsize=(10, 7), sharex=True)
        for i in range(len(samples[0, 0, :10])):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
        axes[-1].set_xlabel("step number")
        #plt.show()

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
        plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/mc_mdl/order%s_mc_mdl_%s'%(order, run_name))
        # plt.show()
        plt.close()
        #plt.show()

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
        plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/profiles/order%s_profile_%s'%(order, run_name))
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

        mdl1_err =abs(mdl1-mdl1_neg)
        #plt.scatter(continuum_waves, continuum_flux, label = 'continuum_points')
        plt.legend()
        plt.title('continuum from mcmc')
        plt.xlabel("wavelengths")
        plt.ylabel("flux")
        plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/continuum_fit/order%s_cont_%s'%(order, run_name))
        plt.close()
        #plt.show()

        ## last section is a bit of a mess but plots the two forward models

        mcmc_inputs = np.concatenate((profile, poly_cos))
        mcmc_mdl = model_func(mcmc_inputs, x)
        #mcmc_mdl = mcmc_mdl[idx]
        mcmc_liklihood = log_probability(mcmc_inputs, x, y, yerr)

        print('Likelihood for mcmc: %s'%mcmc_liklihood)

        residuals_2 = (y+1) - (mcmc_mdl+1)

        fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
        non_masked = tuple([yerr<10])
        #ax[0].plot(x, y+1, color = 'r', alpha = 0.3, label = 'data')
        #ax[0].plot(x[non_masked], mcmc_mdl[non_masked]+1, color = 'k', alpha = 0.3, label = 'mcmc spec')
        ax[1].scatter(x[non_masked], residuals_2[non_masked], marker = '.')
        ax[0].plot(x, y, 'r', alpha = 0.3, label = 'data')
        ax[0].plot(x, mcmc_mdl, 'k', alpha =0.3, label = 'mcmc spec')
        residual_masks = tuple([yerr>=100000000000000])

        #residual_masks = tuple([yerr>10])
        ax[0].scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        ax[0].legend(loc = 'lower right')
        #ax[0].set_ylim(0, 1)
        #plotdepths = -np.array(line_depths)
        #ax[0].vlines(line_waves, plotdepths, 1, label = 'line list', color = 'c', alpha = 0.5)
        ax[1].plot(x, residuals_2, '.')
        #ax[1].scatter(x[residual_masks], residuals_2[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        z_line = [0]*len(x)
        ax[1].plot(x, z_line, '--')
        plt.savefig('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/forward_models/order%s_forward_%s'%(order, run_name))
        # plt.show()
        plt.close()
        #plt.show()

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
        #plt.show()

        profile_f = np.exp(profile)
        profile_errors_f = profile_err*profile_f
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
        #plt.close('all')

        profiles = []
        popts_new = []
        corrected_spec = []
        phases = []
        #plt.figure()
        # task_part = partial(task, all_frames)
        # with mp.Pool(mp.cpu_count()) as pool:results=[pool.map(task_part, np.arange(len(frames)))]
        # results = np.array(results[0])
        # for i in range(len(frames)):
        #     all_frames[i]=results[i][i]
        for counter in range(len(frames)):
                all_frames = task(all_frames, counter)
            
    # plt.show()
    plt.close('all')

    # print('We are done')

    # inp = input('Enter...')

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

# plt.figure()
# plt.scatter(phases, popts - np.median(popts), label = 'basic continuum correction')
# plt.scatter(phases, popts_new-np.median(popts_new), label = 'mcmc continuum correction (LSD profile)')
# plt.scatter(phases, ccf_rvs[:len(popts)]-np.median(ccf_rvs[:len(popts)]), label = 'ccf rvs')
# plt.legend()
# plt.show()

# month_spec = np.array(month_spec)

# plt.figure()
# plt.plot(month_spec[0, 0, :], month_spec[0, 2, :], label = '%s, BERV corrected'%months[0], color = 'b', linestyle= '-')
# plt.plot(month_spec[1, 0, :], month_spec[1, 2, :], label = '%s, BERV corrected'%months[1], color = 'b', linestyle='--')
# plt.plot(month_spec[0, 1, :], month_spec[0, 3, :], label = '%s, not corrected'%months[0], color = 'orange', linestyle= '-')
# plt.plot(month_spec[1, 1, :], month_spec[1, 3, :], label = '%s, not corrected'%months[1], color = 'orange', linestyle='--')
# plt.legend()

# plt.figure()
# plt.title('BERV Corrected')
# plt.plot(month_spec[0, 0, :], month_spec[0, 2, :], label = '%s'%months[0], linestyle= '-')
# plt.plot(month_spec[1, 0, :], month_spec[1, 2, :], label = '%s'%months[1], linestyle='--')
# plt.legend()

# plt.figure()
# plt.title('Not corrected')
# plt.plot(month_spec[1, 1, :], month_spec[1, 3, :], label = '%s'%months[1],  linestyle='--')
# plt.plot(month_spec[0, 1, :], month_spec[0, 3, :], label = '%s'%months[0],  linestyle= '-')
# plt.legend()

# plt.show()


# for no in range(len(corrected_spectrum_e2ds)):
#     e2ds_spec, e2ds_wave, e2ds_err = corrected_spectrum_e2ds[no]
#     s1d_spec, s1d_wave, s1d_err = corrected_spectrum_s1d[no]
#     plt.figure()
#     plt.errorbar(e2ds_spec, e2ds_wave, e2ds_err, label = 'e2ds')
#     plt.errorbar(s1d_spec, s1d_wave, s1d_err, label = 's1d')
#     plt.legend()
#     plt.savefig('/home/lsd/Documents/LSD_Figures/e2ds_vs_s1d/%s.png'%no)

# plt.figure()
# plt.title('e2ds frames')
# plt.plot(cse2ds[0, 0], cse2ds[0, 1])
# plt.plot(cse2ds[3, 0], cse2ds[3, 1])

## figure with insets
# fig = plt.figure(figsize = [20,12])
# gs = fig.add_gridspec(3, 6)
# ax = fig.add_subplot(gs[1:, 1:5])
# ax.plot(corrected_spec[0, 0, :]-0.0121, corrected_spec[0, 1, :], label='s1d - shifted by 0.0121 Angstroms')
# ax.plot(corrected_spec_e2ds[0, 0, :], corrected_spec_e2ds[0, 1, :], label='e2ds')
# ax.scatter(corrected_spec_e2ds[0, 0, idx[0]], corrected_spec_e2ds[0, 1, idx[0]], label ='overlap region', alpha = 0.1)
# plt.legend(loc=[1.1, -0.1])

# ## middle
# axins = ax.inset_axes([0.25, 1.1, 0.47, 0.47])
# axins.plot(corrected_spec[0, 0, :]-0.0121, corrected_spec[0, 1, :], label='s1d - shifted by 0.0121 Angstroms')
# axins.plot(corrected_spec_e2ds[0, 0, :], corrected_spec_e2ds[0, 1, :], label='e2ds')
# axins.scatter(corrected_spec_e2ds[0, 0, idx[0]], corrected_spec_e2ds[0, 1, idx[0]], label ='overlap region', alpha = 0.1)
# axins.set_xlim(4601.5, 4604.1)
# ax.indicate_inset_zoom(axins, edgecolor = 'black')
# ##start
# axins2 = ax.inset_axes([-0.3, 1.1, 0.47, 0.47])
# axins2.plot(corrected_spec[0, 0, :]-0.0121, corrected_spec[0, 1, :], label='s1d - shifted by 0.0121 Angstroms')
# axins2.plot(corrected_spec_e2ds[0, 0, :], corrected_spec_e2ds[0, 1, :], label='e2ds')
# axins2.scatter(corrected_spec_e2ds[0, 0, idx[0]], corrected_spec_e2ds[0, 1, idx[0]], label ='overlap region', alpha = 0.1)
# axins2.set_xlim(4575.1, 4577.5)
# ax.indicate_inset_zoom(axins2, edgecolor = 'black')
# ## end
# axins3 = ax.inset_axes([0.8, 1.1, 0.47, 0.47])
# axins3.plot(corrected_spec[0, 0, :]-0.0121, corrected_spec[0, 1, :], label='s1d - shifted by 0.0121 Angstroms')
# axins3.plot(corrected_spec_e2ds[0, 0, :], corrected_spec_e2ds[0, 1, :], label='e2ds')
# axins3.scatter(corrected_spec_e2ds[0, 0, idx[0]], corrected_spec_e2ds[0, 1, idx[0]], label ='overlap region', alpha = 0.1)
# axins3.set_xlim(4622.2, 4624.7)
# ax.indicate_inset_zoom(axins3, edgecolor = 'black')

# plt.show()

