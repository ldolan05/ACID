import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.io import  fits
import emcee
import LSD_func_faster as LSD
import glob
from scipy.interpolate import interp1d
from math import log10, floor
import multiprocessing as mp
from time import time


def ACID_order(order, poly_ord, filelist):
    ## read in spectra for each frame, along with S/N of each frame
    frame_wavelengths, frames, frame_errors, sns = read_in_frames(order, filelist)
    ## combines spectra from each frame (weighted based of S/N), returns to S/N of combined spec
    wavelengths, fluxes, flux_error_order, sn = combine_spec(frame_wavelengths[-1], frames, frame_errors, sns)

    #### running the MCMC #####

    ## setting these normalisation factors as global variables - normalises wavelengths to keep polynomial coefficents small
    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)

    ## getting the initial polynomial coefficents
    poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes,  (wavelengths*a)+b, flux_error_order, poly_ord)

    #### getting the initial profile   
    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, sn)

    ## Setting the number of point in the spectrum (j_max) and vgrid (k_max)
    j_max = int(len(fluxes))
    k_max = len(profile)

    model_inputs = np.concatenate((profile, poly_inputs))

    ## setting x, y, yerr for emcee
    x = wavelengths
    y = fluxes
    yerr = flux_error_order

    ## limit, min velocity and max velocity for penalty function
    p_var = 0.001 
    v_min = -10
    v_max = 10

    ## masking based of resiudals from initial input forward model 
    forward = model_func(model_inputs, x)
    yerr_unmasked = yerr
    yerr, model_inputs_resi, mask_idx = residual_mask(x, y, yerr, model_inputs)

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
            sigma = abs(round_sig(model_inputs[i], 1))/10
            pos2 = rng.normal(model_inputs[i], sigma, (nwalkers, ))
        pos.append(pos2)

    pos = np.array(pos)
    pos = np.transpose(pos)

    ## the number of steps is how long it runs for - if it doesn't look like it's settling at a value try increasing the number of steps
    steps_no = 10000

    # running the mcmc using python package emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
    sampler.run_mcmc(pos, steps_no, progress=True)

    idx = tuple([yerr<=10000000000000000000])

    ## discarding all vales except the last 1000 steps.
    dis_no = int(np.floor(steps_no-5000))
    
    ## combining all walkers togather
    flat_samples = sampler.get_chain(discard=dis_no, flat=True)

    ## getting the final profile and continuum values
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
    
    ## calculates model continuum fit
    mdl1 =0
    for i in np.arange(0, len(poly_cos)-1):
        mdl1 = mdl1+poly_cos[i]*((a*x)+b)**(i)
    mdl1 = mdl1*poly_cos[-1]

    ## LSD returns opacity profile - this converts back into flux
    profile_f = np.exp(profile)
    profile_errors_f = np.sqrt(profile_err**2/profile_f**2)
    profile_f = profile_f-1

    profiles = []
    for counter in range(0, len(frames)):
        flux = frames[counter]
        error = frame_errors[counter]
        wavelengths = frame_wavelengths[counter]

        #masking based off residuals
        error[mask_idx]=10000000000000000000

        ## setting these normalisation factors for new wavelength range
        a = 2/(np.max(wavelengths)-np.min(wavelengths))
        b = 1 - a*np.max(wavelengths)

        ## copying original flux and error
        flux_b = flux.copy()
        error_b = error.copy()

        # plt.show()
        flux = flux/mdl1
        error = error/mdl1

        ## checks for negative flux
        idx = tuple([flux>0])
        if len(flux[idx])==0:
            print('continuing... frame %s'%counter)
            continue
        
        velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, sn)
        
        profile_f = np.exp(profile)
        profile_errors_f = np.sqrt(profile_err**2/profile_f**2)
        profile_f = profile_f-1

        all_frames[counter, order]=[profile_f, profile_errors_f]
    
    return all_frames[:, order]

file_type = 'e2ds'

# run_name = input('Input nickname for this version of code (for saving figures): ')
# linelist = input('Enter file path to line list (VALD format):')
# directory = input('Enter path to directory containing data files:')
# fig_ans = input('Save Figures? (y/n):')
# fig_path = input('Enter file path for directory to save figures and result files:')
# if fig_path == '':
#     fig_path = '/Users/lucydolan/Documents/ACID/ACID_RESULTS/'

# if linelist == '':
#     linelist = '/home/lsd/Documents/fulllinelist0001.txt'
# if directory == '':
#     print('default directory set')
#     directory = '/Users/lucydolan/Starbase/HD189733/August2007/*/*/*/'

run_name = 'HDtest'
linelist = '/Users/lucydolan/Documents/ACID/TEST/fulllinelist0001.txt'
directory = '/Users/lucydolan/Documents/ACID/TEST/*/*/*/'
fig_ans = 'n'
fig_path = '/Users/lucydolan/Documents/ACID/TEST/'

# run_name = 'Sun'
# linelist = '/Users/lucydolan/Documents/ACID/AM_LSD/MM-LSD/VALD_files/Sun.txt'
# directory = '/Users/lucydolan/Documents/ACID/AM_LSD/MM-LSD/data/Sun/data/'
# fig_ans = 'y'
# fig_path = '/Users/lucydolan/Documents/ACID/AM_LSD/MM-LSD/ACID/'

def round_sig(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def findfiles(directory, file_type):

    filelist=glob.glob('%s*%s_A.fits'%(directory, file_type))               #finding all A band e2ds spectra
    if len(filelist)==0:
        filelist=glob.glob('%s/*%s_A.fits'%(directory, 'S2D'))
        if len(filelist)==0:
            print('Empty file folder please check path: "%s*%s_A.fits"'%(directory, file_type))
    return filelist

def read_in_frames(order, filelist):
    frames = []
    errors = []
    frame_wavelengths = []
    sns = []
    max_sn = 0

    ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
    for file in filelist:
        fluxes, wavelengths, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'unmasked', run_name)
        frame_wavelengths.append(wavelengths)
        frames.append(fluxes)
        errors.append(flux_error_order)
        sns.append(sn)
        ### finding highest S/N frame, saves this as reference frame
        if sn>max_sn:
            max_sn = sn
            reference_frame=fluxes

    frames = np.array(frames)
    errors = np.array(errors)

    ### each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
    for n in range(len(frames)):
        div_frame = frames[n]/reference_frame

        ### creating windows to fit polynomial to
        binned = np.zeros(int(len(div_frame)/2))
        binned_waves = np.zeros(int(len(div_frame)/2))
        for i in range(0, len(div_frame)-1, 2):
            pos = int(i/2)
            binned[pos] = (div_frame[i]+div_frame[i+1])/2
            binned_waves[pos] = (wavelengths[i]+wavelengths[i+1])/2

        ### fitting polynomial to div_frame
        coeffs=np.polyfit(binned_waves, binned, 2)
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)
        frames[n] = frames[n]/fit
        errors[n] = errors[n]/fit

    return frame_wavelengths, frames, errors, sns

def continuumfit(fluxes1, wavelengths1, errors1, poly_ord):

        cont_factor = fluxes1[0]

        fluxes1 = fluxes1
        ## taking out masked areas
        if np.max(fluxes1)<1:
            idx = [errors1<1]
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
        coeffs=np.polyfit(clipped_waves, clipped_flux/cont_factor, poly_ord)

        poly = np.poly1d(coeffs*cont_factor)
        fit = poly(wavelengths1)
        flux_obs = fluxes1/fit
        new_errors = errors1/fit

        ## replacing negative values with a very small number and masking it out to avoid error in optical depth conversion
        ## this function is only used to get emcee inputs and so this does not affect the final profiles
        idx = tuple([flux_obs<=0])
        flux_obs[idx] = 0.00000000000000001
        new_errors[idx]=1000000000000000
        
        coeffs=coeffs[::-1]
        coeffs = list(coeffs)
        coeffs.append(cont_factor)
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

        weights = (1/temp_err**2)
        weights = weights/np.sum(weights)

        spectrum[n]=sum(weights*temp_spec)
        sn = sum(weights*sns)/sum(weights)

        spec_errors[n]=(np.std(temp_spec))*np.sqrt(sum(weights**2))
   
    return wavelengths, spectrum, spec_errors, sn

def od2flux(x):
    return np.exp(x)-1

def flux2od(x):
    return np.log(x+1)

no_line = 100
## model for emcee code - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
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
        # excluding the continuum points in the profile
        z_cont = []
        v_cont = []
        for i in range(0, 3):
                z_cont.append(theta[len(theta)-i-1])
                v_cont.append(velocities[len(velocities)-i-1])
                z_cont.append(theta[i])
                v_cont.append(velocities[i])

        z_cont = np.array(z_cont)
       
        # calculate penalty function for the continuum points - encourages continuum of profile to remain at 0.
        p_pent = np.sum((1/np.sqrt(2*np.pi*p_var**2))*np.exp(-0.5*(z_cont/p_var)**2))

        return p_pent

    return -np.inf

## calculates log probability - used for mcmc
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    final = lp + log_likelihood(theta, x, y, yerr)
    return final

## iterative residual masking - mask continuous areas first
def residual_mask(wavelengths, data_spec_in, data_err, initial_inputs):
    
    forward = model_func(initial_inputs, wavelengths)

    ### (roughly) normalise the data (easier to set standard threshold for residuals)
    residuals = (data_spec_in - np.min(data_spec_in))/(np.max(data_spec_in)-np.min(data_spec_in)) - (forward - np.min(forward))/(np.max(forward)-np.min(forward))

    ### finds consectuative sections where at least 20 points have resiudals greater than 0.25 - these are masked
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


    residual_masks = tuple([data_err>=1000000000000000000])

    ### sigma clip masking  - masks lines where the line depth in the data is very different to what the line list predicts ##
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
    velocities, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, sn)

    return data_err, np.concatenate((profile, poly_inputs)), residual_masks

# min_order = int(input('Enter minimum order (Please exclude orders with negative/zero flux):'))
# max_order = int(input('Enter maximum order (Please exclude orders with negative/zero flux):'))

# if min_order == '':
#     min_order = 8
# if max_order == '':
#     max_order = 71

min_order = 8
max_order = 71

order_range = np.arange(min_order, max_order)

phase_calc = 'n' #input('Include phase in fits header (y/n):')

if phase_calc == 'y' or '':
    result = input('Enter Period, Mid-Tranist Time and Transit Duration:')
    if result == '':
        P=2.21857567 #Cegla et al, 2006 - days
        T=2454279.436714 #Cegla et al, 2006
        t=0.076125 #Torres et al, 2008
    deltaphi = t/(2*P)


filelist=findfiles(directory, file_type)
# poly_ord = int(input('Enter order of polynomial for continuum fit (recommended is 3):'))
# if poly_ord == '':
#     poly_ord = 3

poly_ord = 3

#### setting up empty array for final profiles to go into
all_frames = np.zeros((len(filelist), 71, 2, 48))

print('We have gotten to here')

t0 = time()
print(t0)

pool = mp.Pool(1)

results = pool.map(ACID_order, [order for order in order_range])

pool.close()

print('are we here?')

# adding into fits files for each frame
for frame_no in range(0, len(all_frames[:, 0])):
    file = filelist[frame_no]
    fits_file = fits.open(file)
    phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
    hdu = fits.HDUList()
    hdr = fits.Header()

    for order in range(0, 71):
        if phase_calc == 'y':
            hdr['PHASE'] = phi

            if phi<deltaphi:
                result = 'in'
            elif phi>1-deltaphi:
                result = 'in'
            else:
                result = 'out'
            
            hdr['result'] = result

        hdr['ORDER'] = order+1

        profile = all_frames[frame_no, order, 0]
        profile_err = all_frames[frame_no, order, 1]

        hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
    hdu.writeto('%s%s_frame%s.fits'%(fig_path, run_name, frame_no), output_verify = 'fix', overwrite = 'True')

t1 = time()

print('ACID Complete. Time taken: %ss'%t1-t0)