import sys, emcee, warnings, os, time, importlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
from . import utils
from . import LSD

warnings.filterwarnings("ignore")
importlib.reload(LSD)
importlib.reload(utils)


def continuumfit(fluxes, wavelengths, errors, poly_ord):
        
        cont_factor = fluxes[0]
        if cont_factor == 0: 
            cont_factor = np.mean(fluxes)
        idx = wavelengths.argsort()
        wavelength = wavelengths[idx]
        fluxe = fluxes[idx] / cont_factor
        clipped_flux = []
        clipped_waves = []
        binsize = 100
        for i in range(0, len(wavelength), binsize):
            waves = wavelength[i:i+binsize]
            flux = fluxe[i:i+binsize]
            indicies = flux.argsort()
            flux = flux[indicies]
            waves = waves[indicies]
            clipped_flux.append(flux[len(flux)-1])
            clipped_waves.append(waves[len(waves)-1])
        coeffs = np.polyfit(clipped_waves, clipped_flux, poly_ord)
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths) * cont_factor
        flux_obs = fluxes / fit
        new_errors = errors / fit

        return np.concatenate((np.flip(coeffs), [cont_factor])), flux_obs, new_errors, fit

def read_in_frames(order, filelist, file_type, directory=None):
    
    # read in first frame
    fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct(
        file_type, 'order', order, filelist[0], directory, 'unmasked', run_name, 'y')
    
    frames = np.zeros((len(filelist), len(wavelengths)))
    errors = np.zeros((len(filelist), len(wavelengths)))
    frame_wavelengths = np.zeros((len(filelist), len(wavelengths)))
    sns = np.zeros((len(filelist), ))

    frames[0] = fluxes
    errors[0] = flux_error_order
    frame_wavelengths[0] = wavelengths
    sns[0] = sn

    def task_frames(frames, errors, frame_wavelengths, sns, i):
        file = filelist[i]
        frames[i], frame_wavelengths[i], errors[i], sns[i], mid_wave_order, telluric_spec, overlap = LSD.blaze_correct(file_type, 'order', order, file, directory, 'unmasked', run_name, 'y')
        # print(i, frames)
        return frames, frame_wavelengths, errors, sns
    
    ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
    for i in range(len(filelist[1:])+1):
        # print(i)
        frames, frame_wavelengths, errors, sns = task_frames(frames, errors, frame_wavelengths, sns, i)
        
    ### finding highest S/N frame, saves this as reference frame

    idx = (sns==np.max(sns))
    global reference_wave
    reference_wave = frame_wavelengths[idx][0]
    reference_frame = frames[idx][0]
    reference_frame[reference_frame == 0] = 0.001
    reference_error = errors[idx][0]
    reference_error[reference_frame == 0] = 1000000000000000000

    global frames_unadjusted
    frames_unadjusted = frames
    global frame_errors_unadjusted
    frame_errors_unadjusted = errors

    ### each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
    for n in range(len(frames)):
        f2 = interp1d(frame_wavelengths[n], frames[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
        div_frame = f2(reference_wave)/reference_frame

        idx_ref = (reference_frame<=0)
        div_frame[idx_ref]=1

        binned = []
        binned_waves = []
        binsize = int(round(len(div_frame)/5, 1))
        for i in range(0, len(div_frame), binsize):
            if i+binsize<len(reference_wave):
                waves = reference_wave[i:i+binsize]
                flux = div_frame[i:i+binsize]
                waves = waves[abs(flux-np.median(flux))<0.1]
                flux = flux[abs(flux-np.median(flux))<0.1]
                binned.append(np.median(flux))
                binned_waves.append(np.median(waves))

        binned = np.array(binned)
        binned_waves = np.array(binned_waves)
    
        ### fitting polynomial to div_frame
        try:coeffs = np.polyfit(binned_waves, binned, 4)
        except:coeffs = np.polyfit(binned_waves, binned, 2)
        poly = np.poly1d(coeffs)
        fit = poly(frame_wavelengths[n])
        frames[n] = frames[n]/fit
        errors[n] = errors[n]/fit
        idx = (frames[n] == 0)
        frames[n][idx] = 0.00001
        errors[n][idx] = 1000000000

    return frame_wavelengths, frames, errors, sns, telluric_spec

def calc_deltav(wavelengths):
    """Calculates velocity pixel size

    Calculates the velocity pixel size for the LSD velocity grid based off the spectral wavelengths.

    Args:
        wavelengths (array): Wavelengths for ACID input spectrum (in Angstroms).
        
    Returns:
        float: Velocity pixel size in km/s
    """ 
    resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    return resol1 / (wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2)) * LSD.ckms

def combine_spec(wavelengths_f, spectra_f, errors_f, sns_f):

    interp_spec = np.zeros(spectra_f.shape)
    # combine all spectra to one spectrum
    for n in range(len(wavelengths_f)):

        global reference_wave
        idx_ref = (sns_f == np.max(sns_f))
        reference_wave = wavelengths_f[idx_ref][0]

        idx = np.where(wavelengths_f[n] != 0)[0]

        f2 = interp1d(wavelengths_f[n][idx], spectra_f[n][idx], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
        f2_err = interp1d(wavelengths_f[n][idx], errors_f[n][idx], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
        spectra_f[n] = f2(reference_wave)
        errors_f[n] = f2_err(reference_wave)

        # print(spectra_f[n])
        # print(errors_f[n])

        ## mask out out extrapolated areas
        idx_ex = np.logical_and(reference_wave<=np.max(wavelengths_f[n][idx]), reference_wave>=np.min(wavelengths_f[n][idx]))
        idx_ex = tuple([idx_ex==False])

        spectra_f[n][idx_ex] = 1.
        errors_f[n][idx_ex] = 1000000000000

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
        # print(weights_f[idx])
        weights_f[idx] = 0.

        if sum(weights_f) > 0:
            weights_f = weights_f / np.sum(weights_f)

            spectrum_f[n] = sum(weights_f * temp_spec_f)
            sn_f = sum(weights_f * sns_f) / sum(weights_f)

            spec_errors_f[n] = 1 / (sum(weights_f ** 2))
        
        else: 
            spectrum_f[n] = np.mean(temp_spec_f)
            spec_errors_f[n] = 1000000000000

    
    return reference_wave, spectrum_f, spec_errors_f, sn_f

def model_func(inputs, x):
    ## model for the mcmc - takes the profile(z) and the continuum coefficents(inputs[k_max:]) to create a model spectrum.
    z = inputs[:k_max]

    mdl = np.dot(alpha, z) ##alpha has been declared a global variable after LSD is run.

    #converting model from optical depth to flux
    mdl = np.exp(mdl)

    ## these are used to adjust the wavelengths to between -1 and 1 - makes the continuum coefficents smaller and easier for emcee to handle.
    a = 2/(np.max(x)-np.min(x))
    b = 1 - a*np.max(x)

    mdl1 = 0
    for i in range(k_max, len(inputs) - 1):
        mdl1 = mdl1 + (inputs[i] * ((x * a) + b) ** (i - k_max))

    # coefs = np.asarray(inputs[k_max:-1], dtype=float) # Potential improvement - Ben
    # X = (a * x) + b
    # if coefs.size:
    #     powers = np.arange(coefs.size)
    #     # X[:, None] ** powers -> shape (len(x), coefs.size); dot with coefs -> (len(x),)
    #     mdl1 = np.dot(X[:, None] ** powers[None, :], coefs)
    # else:
    #     mdl1 = 0.0

    mdl1 = mdl1 * inputs[-1]
    
    mdl = mdl * mdl1
   
    return mdl

def log_likelihood(theta, x, y, yerr):
    ## maximum likelihood estimation for the mcmc model.
    model = model_func(theta, x)

    lnlike = -0.5 * np.sum(((y) - (model)) ** 2 / yerr**2 + np.log(yerr**2)+ np.log(2*np.pi))

    return lnlike

def log_prior(theta):
    ## imposes the prior restrictions on the inputs - rejects if profile point is less than -10 or greater than 0.5.
    check = 0
    z = theta[:k_max]

    for i in range(len(theta)):
        if i < k_max: ## must lie in z
            if -10 <= theta[i] <= 0.5:
                pass
            else:
                check = 1

    if check == 0:

        # excluding the continuum points in the profile (in flux)
        z_cont = []
        v_cont = []
        for i in range(0, 5):
                z_cont.append(np.exp(z[len(z)-i-1])-1)
                v_cont.append(velocities[len(velocities)-i-1])
                z_cont.append(np.exp(z[i])-1)
                v_cont.append(velocities[i])

        z_cont = np.array(z_cont)
        v_cont = np.array(v_cont)

        p_pent = np.sum((np.log((1/np.sqrt(2*np.pi*0.01**2)))-0.5*(z_cont/0.01)**2))

        return p_pent

    return -np.inf

def log_probability(theta):
    ## calculates log probability - used for mcmc
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    final = lp + log_likelihood(theta, x, y, yerr)
    return final

def residual_mask(wavelengths, data_spec_in, data_err, initial_inputs, poly_ord, linelist,
                  velocities=np.arange(-25, 25, 0.82), pix_chunk=20, dev_perc=25, 
                  tell_lines=None, n_sig=1, alpha=None):
    ## iterative residual masking - mask continuous areas first - then possibly progress to masking the narrow lines

    forward = model_func(initial_inputs, wavelengths)

    a = 2 / (np.max(wavelengths) - np.min(wavelengths))
    b = 1 - a * np.max(wavelengths)

    mdl1 = 0
    for i in range(k_max, len(initial_inputs) - 1):
        mdl1 = mdl1 + (initial_inputs[i] * ((wavelengths * a) + b) ** (i - k_max))

    mdl1 = mdl1 * initial_inputs[-1]

    data_normalised = (data_spec_in - np.min(data_spec_in)) / (np.max(data_spec_in) - np.min(data_spec_in))
    forward_normalised = (forward - np.min(forward)) / (np.max(forward) - np.min(forward))
    residuals = data_normalised - forward_normalised

    data_err_compare = data_err.copy()
    
    ### finds consectuative sections where at least pix_chunk points have residuals greater than 0.25 - these are masked
    idx = (abs(residuals) > dev_perc / 100)

    flag_min = 0
    flag_max = 0
    for value in range(len(idx)):
        if idx[value] == True and flag_min <= value:
            flag_min = value
            flag_max = value
        elif idx[value] == True and flag_max < value:
            flag_max = value
        elif idx[value] == False and flag_max - flag_min >= pix_chunk:
            data_err[flag_min:flag_max] = 10000000000000000000
            flag_min = value
            flag_max = value

    ##############################################
    #                  TELLURICS                 #   
    ##############################################

    # data_err_compare = data_err.copy()

    ## masking tellurics
    for line in tell_lines:
        limit = (21/2.99792458e5)*line +3
        idx = np.logical_and((line-limit) <= wavelengths, wavelengths <= (limit+line))
        data_err[idx] = 10000000000000000000

    residual_masks = tuple([data_err >= 10000000000000000000])

    ###################################
    ###      sigma clip masking     ###
    ###################################

    m = np.median(residuals)
    sigma = np.std(residuals)
    a = 1

    upper_clip = m + a * sigma
    lower_clip = m - a * sigma

    rcopy = residuals.copy()

    idx1 = tuple([rcopy <= lower_clip])
    idx2 = tuple([rcopy >= upper_clip])

    data_err[idx1] = 10000000000000000000
    data_err[idx2] = 10000000000000000000

    poly_inputs, bin, bye, fit = continuumfit(data_spec_in, (wavelengths * a) + b, data_err, poly_ord)
    velocities1, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line = LSD.LSD(
        wavelengths, bin, bye, linelist, 'False', poly_ord, 100, 30, run_name, velocities)

    # ## comment if you would like to keep sigma clipping masking in for final LSD run 
    # residual_masks = tuple([data_err>=1000000000000000000])

    return data_err, np.concatenate((profile, poly_inputs)), residual_masks

def get_profiles(all_frames, order, poly_cos, continuum_error, counter):
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

    #masking based off residuals interpolated onto new wavelength grid
    if len(frame_wavelengths)>1:
        reference_wave = frame_wavelengths[sns==max(sns)][0]
    else:
        reference_wave = frame_wavelengths[0]
    mask_pos = np.ones(reference_wave.shape)
    mask_pos[mask_idx]=10000000000000000000
    f2 = interp1d(reference_wave, mask_pos, bounds_error = False, fill_value = np.nan)
    interp_mask_pos = f2(wavelengths)
    interp_mask_idx = tuple([interp_mask_pos>=10000000000000000000])

    error[interp_mask_idx]=10000000000000000000

    # corrrecting continuum
    error = (error/flux) + (continuum_error/mdl1)
    flux = flux/mdl1
    error  = flux*error

    remove = tuple([flux<0])
    flux[remove]=1.
    error[remove]=10000000000000000000

    idx = tuple([flux>0])
    
    if len(flux[idx])==0:
        print('continuing... frame %s'%counter)
    
    else:
        velocities1, profile1, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(
            wavelengths, flux, error, linelist, 'False', poly_ord, sn, 10, 'test', velocities)

        p = np.exp(profile1)-1

        profile_f = np.exp(profile1)
        profile_errors_f = np.sqrt(profile_errors**2/profile_f**2)
        profile_f = profile_f-1

        all_frames[counter, order]=[profile_f, profile_errors_f]
        
        return all_frames

def combineprofiles(spectra, errors):
    spectra = np.array(spectra)
    idx = np.isnan(spectra)
    shape_og = spectra.shape
    if len(spectra[idx])>0:
        spectra = spectra.reshape((len(spectra)*len(spectra[0]), ))
        for n in range(len(spectra)):
            if spectra[n] == np.nan:
                spectra[n] = (spectra[n+1]+spectra[n-1])/2
                if spectra[n] == np.nan:
                    spectra[n] = 0.
    spectra = spectra.reshape(shape_og)
    errors = np.array(errors)

    
    spectra_to_combine = []
    weights=[]
    for n in range(0, len(spectra)):
        if np.sum(spectra[n])!=0:
            spectra_to_combine.append(list(spectra[n]))
            temp_err = np.array(errors[n, :])
            weight = (1/temp_err**2)
            weights.append(np.mean(weight))
    weights = np.array(weights/sum(weights))

    spectra_to_combine = np.array(spectra_to_combine)

    length, width = np.shape(spectra_to_combine)
    spectrum = np.zeros((1,width))
    spec_errors = np.zeros((1,width))

    for n in range(0, width):
        temp_spec = spectra_to_combine[:, n]
        spectrum[0,n]=sum(weights*temp_spec)/sum(weights)
        spec_errors[0,n] = (np.std(temp_spec, ddof=1)**2) * np.sqrt(sum(weights**2))

    spectrum = list(np.reshape(spectrum, (width,)))
    spec_errors = list(np.reshape(spec_errors, (width,)))

    return  spectrum, spec_errors

def ACID(input_wavelengths, input_spectra, input_spectral_errors, line, frame_sns,
         vgrid, all_frames='default', poly_or=3, pix_chunk=20, dev_perc=25, name="test",
         n_sig=1, telluric_lines=None, order=0, verbose=True, parallel=True, cores=None,
         nsteps=8000):
    """Accurate Continuum fItting and Deconvolution

    Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra, returning an LSD profile for each spectrum given. 
    Spectra must cover a similiar wavelength range.

    Parameters
    ----------
    input_wavelengths : list or array
        Wavelengths for each frame (in Angstroms).
    input_spectra : list or array
        Spectral frames (in flux).
    input_spectral_errors : list or array
        Errors for each frame (in flux).
    line : str
        Path to linelist. Takes VALD linelist in long or short format as input. Minimum line depth input into VALD must
        be less than 1/(3*SN) where SN is the highest signal-to-noise ratio of the spectra. 
    frame_sns : list or array
        Average signal-to-noise ratio for each frame (used to calculate minimum line depth to consider from line list. 
    vgrid : array
        Velocity grid for LSD profiles (in km/s).
    all_frames : str or array, optional
        Output array for resulting profiles. Only neccessary if looping ACID function over many wavelength
        regions or order (in the case of echelle spectra). General shape needs to be (no. of frames, 1, 2, no. of velocity pixels)., by default 'default'
    poly_or : int, optional
        Order of polynomial to fit as the continuum, by default 3
    pix_chunk : int, optional
        Size of 'bad' regions in pixels. 'bad' areas are identified by the residuals between an inital model
        and the data. If a residual deviates by a specified percentage (dev_perc) for a specified number of pixels, by default 20
    dev_perc : int, optional
        Allowed deviation percentage. 'bad' areas are identified by the residuals between an inital model
        and the data. If a residual deviates by a specified percentage (dev_perc) for a specified number of pixels, by default 25
    n_sig : int, optional
        Number of sigma to clip in sigma clipping. Ill fitting lines are identified by sigma-clipping the
        residuals between an inital model and the data. The regions that are clipped from the residuals will
        be masked in the spectra. This masking is only applied to find the continuum fit and is removed when
        LSD is applied to obtain the final profiles, by default 1
    telluric_lines : list, optional
        List of wavelengths (in Angstroms) of telluric lines to be masked. This can also include problematic
        lines/features that should be masked also. For each wavelengths in the list ~3Å eith side of the line is masked., by default None
    order : int, optional
        Only applicable if an all_frames output array has been provided as this is the order position in that
        array where the result should be input. i.e. if order = 5 the output profile and errors would be inserted in all_frames[:, 5]., by default 0
    verbose : bool, optional
        If True prints out time taken for each section of the code. Defaults to True., by default True
    parallel : bool, optional
        If True uses multiprocessing to calculate the profiles for each frame in parallel, by default True
    cores : int, optional
        Number of cores to use if parallel=True. If None (default) all available cores will be used, by default None
    nsteps : int, optional
        nsteps (int, optional): Number of steps for the MCMC to run, try increasing if it doesn't converge, by default 8000

    Returns
    -------
    array
        Resulting profiles and errors for spectra.

    Raises
    ------
    TypeError
        If the input types are not as expected.
    """
    ### Setup

    # Ensure inputs are lists, np.arrays are converted to lists
    input_wavelengths, input_spectra, input_spectral_errors, frame_sns = [
        utils.ensure_list(v) for v in (input_wavelengths, input_spectra, input_spectral_errors, frame_sns)]

    # Define tell_lines if not input, check type if it is
    if telluric_lines is None:
        telluric_lines = [3820.33, 3933.66, 3968.47, 4327.74, 4307.90, 4383.55, 4861.34,
                          5183.62, 5270.39, 5889.95, 5895.92, 6562.81, 7593.70, 8226.96]
    elif not (isinstance(telluric_lines, list) or isinstance(telluric_lines, int)):
        raise TypeError("telluric_lines must be a list or integer")

    global velocities
    velocities = vgrid.copy()
    global linelist
    linelist = line
    global poly_ord
    poly_ord = poly_or
    global run_name
    run_name = name
    global frames
    global frame_wavelengths
    global frame_errors
    global sns
    frame_wavelengths = np.array(input_wavelengths)
    frames = np.array(input_spectra)
    frame_errors = np.array(input_spectral_errors)
    sns = np.array(frame_sns)

    if type(all_frames) != np.ndarray:
        if all_frames == 'default':
            all_frames = np.zeros((len(frames), 1, 2, len(velocities)))

    fw = frame_wavelengths.copy()
    f = frames.copy()
    fe = frame_errors.copy()
    s = sns.copy()

    if verbose:
        t0 = time.time()
        print('Initialising...')

    ## combines spectra from each frame (weighted based of S/N), returns to S/N of combined spec
    if len(fw)>1:
        wavelengths, fluxes, flux_error_order, sn = combine_spec(fw, f, fe, s)
    else:
        wavelengths, fluxes, flux_error_order, sn = fw[0], f[0], fe[0], s[0]

    ### getting the initial polynomial coefficents
    a = 2 / (np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a * np.max(wavelengths)
    poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(
        fluxes, (wavelengths*a)+b, flux_error_order, poly_ord)

    # if verbose:
    #     t2 = time.time()
    #     print('Set up before LSD %s'%(t2-t0))

    #### getting the initial profile
    global alpha
    velocities, profile, profile_errors, alpha, continuum_waves, continuum_flux, no_line = LSD.LSD(
        wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, 30, run_name,
        velocities, verbose=verbose)

    # if verbose:
    #     t3 = time.time()
    #     print('LSD run takes: %s'%(t3-t2))

    ## Setting the number of points in vgrid (k_max)
    global k_max
    k_max = len(profile)
    model_inputs = np.concatenate((profile, poly_inputs))

    ## Setting x, y, yerr for emcee
    global x, y, yerr
    x = wavelengths
    y = fluxes
    yerr = flux_error_order

    ## Setting these normalisation factors as global variables - used in the figures below
    a = 2 / (np.max(x) - np.min(x))
    b = 1 - a * np.max(x)

    # Masking based off residuals
    global mask_idx
    
    if verbose:
        print('Residual masking...')
    yerr, model_inputs_resi, mask_idx = residual_mask(x, y, yerr, model_inputs, poly_ord,
                                                      linelist, pix_chunk=pix_chunk,
                                                      dev_perc=dev_perc, tell_lines=telluric_lines,
                                                      n_sig=n_sig, alpha=alpha)

    # if verbose:
    #     t4 = time.time()
    #     print('residual masking takes: %s' %(t4-t3))

    ## Setting number of walkers and their start values(pos)
    ndim = len(model_inputs)
    nwalkers = ndim * 3
    rng = np.random.default_rng()

    ### starting values of walkers with independent variation
    sigma = 0.8 * 0.005
    pos = []
    for i in range(0, ndim):
        if i < ndim - poly_ord - 2:
            pos2 = rng.normal(model_inputs[i], sigma, (nwalkers, ))
        else:
            sigma = abs(utils.round_sig(model_inputs[i], 1)) / 10
            pos2 = rng.normal(model_inputs[i], sigma, (nwalkers, ))
        pos.append(pos2)

    pos = np.array(pos)
    pos = np.transpose(pos)

    if verbose:
        t5 = time.time()
        # print('MCMC set up takes: %s'%(t5-t4))
        print('Initialised in %ss'%round((t5-t0), 2))

    print('Fitting the continuum using emcee...')

    if parallel:

        # Choose a sensible number of workers
        if cores is None:
            cores = os.cpu_count()  # Use all available cores

        if verbose:
            print(f"Using {cores} out of {os.cpu_count()} cores for MCMC")

        # For some reason, unspecified pooling as was before (as in case of windows in the else statement)
        # leds to a hung computer. So specify mp.get_context required, default is spawn, but spawn
        # causes multiple instances of this script to rerun, causing alpha matrix calculation to be redone
        # in each child process. Therefore, fork, which is legacy mp behavior on unix, is used.
        if sys.platform != "win32":
            ctx = mp.get_context("fork")
            with ctx.Pool(processes = cores) as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool)
                sampler.run_mcmc(pos, nsteps, progress=True, store=True)

            # with Pool() as pool: # Original code
            #         sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc_class.log_probability, pool=pool)
            #         sampler.run_mcmc(pos, nsteps, progress=True)
        
        else: # Untested. Now tested, this doesn't work, needs serious modifications to make work
            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, mcmc_class.log_probability, pool=pool)
                sampler.run_mcmc(pos, nsteps, progress=verbose)


    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
        sampler.run_mcmc(pos, nsteps, progress=True)

    print('MCMC run takes: %s'%(time.time()-t5))

    ## discarding all vales except the last 1000 steps.
    dis_no = int(np.floor(nsteps-1000))

    global flat_samples
    ## combining all walkers together
    flat_samples = sampler.get_chain(discard=dis_no, flat=True)

    ## getting the final profile and continuum values - median of last 1000 steps
    profile = []
    global poly_cos
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
    if fig_opt =='y':

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
        plt.savefig('figures/mcmc_and_data.png')

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
        secax = ax0.secondary_yaxis('right', functions = (utils.od2flux, utils.flux2od))
        secax.set_ylabel('flux')
        ax0.legend()
        plt.savefig('figures/profile_%s'%(run_name))

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
        plt.savefig('figures/cont_%s'%(run_name))
    
        mcmc_inputs = np.concatenate((profile, poly_cos))
        mcmc_mdl = model_func(mcmc_inputs, x)

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
        plt.savefig('figures/forward_%s'%(run_name))
        

        fig, ax0 = plt.subplots()
        ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
        zero_line = [0]*len(velocities)
        ax0.plot(velocities, zero_line)
        ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
        ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
        ax0.set_xlabel('velocities')
        ax0.set_ylabel('optical depth')
        ax0.legend()
        plt.savefig('figures/final_profile_%s'%(run_name))

    print('Getting the final profiles...')

    # finding error for the continuuum fit
    inds = np.random.randint(len(flat_samples), size=50)
    conts = []
    for ind in inds:
        sample = flat_samples[ind]
        mdl = model_func(sample, wavelengths)
        #mdl = model_func(sample, x)
        #mdl = mdl[idx]
        mdl1_temp = 0
        for i in np.arange(k_max, len(sample)-1):
            mdl1_temp = mdl1_temp+sample[i]*((a*wavelengths)+b)**(i-k_max)
        mdl1_temp = mdl1_temp*sample[-1]
        conts.append(mdl1_temp)

    continuum_error = np.std(np.array(conts), axis = 0)

    task_part = partial(get_profiles, all_frames, order, poly_cos, continuum_error)
    if len(frames)>1:
        # ctx = mp.get_context("fork") # Changed from spawn to fork to avoid multiple reruns of page
        # with ctx.Pool(processes = mp.cpu_count()) as pool:
        #     results=[pool.map(task_part, np.arange(len(frames)))]
        # results = np.array(results[0])
        # for i in range(len(frames)):
        #     all_frames[i]=results[i][i]
        for counter in range(len(frames)):
            all_frames = get_profiles(all_frames, order, poly_cos, continuum_error, counter)  
    else:
        all_frames = get_profiles(all_frames, order, poly_cos, continuum_error, 0)

    return all_frames

def ACID_HARPS(filelist, line, vgrid, order_range=None, save_path = './',
               file_type = 'e2ds', name="test", **kwargs):
    """Accurate Continuum fItting and Deconvolution for HARPS e2ds and s1d spectra (DRS pipeline 3.5)

    Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra,
    returning an LSD profile for each file given. Files must all be kept in the same folder as well
    as their corresponding blaze files. If 's1d' are being used their e2ds equivalents must also be
    in this folder. Result files containing profiles and associated errors for each order (or
    corresponding wavelength range in the case of 's1d' files) will be created and saved to a
    specified folder. It is recommended that this folder is seperate to the input files.

    Parameters
    ----------
    filelist : list of strings
        List of files. Files must come from the same observation night as continuum is fit for a combined
        spectrum of all frames. A profile and associated errors will be produced for each file specified.
    line : str
        Path to linelist. Takes VALD linelist in long or short format as input. Minimum line depth input into VALD must
        be less than 1/(3*SN) where SN is the highest signal-to-noise ratio of the spectra. 
    vgrid : array
        Velocity grid for LSD profiles (in km/s).
    order_range : array, optional
        Orders to be included in the final profiles. If s1d files are input, the corresponding wavelengths 
        will be considered, by default None, the function generates np.arange(10,70) in the body.
    save_path : str, optional
        Path to the directory where output files will be saved, by default './'
    file_type : str, optional
        Type of the input files, either "e2ds" or "s1d", by default 'e2ds'
    **kwargs
        Additional arguments to be passed to the ACID function. See ACID function for details.

    Returns
    -------
    list
        Barycentric Julian Date for files
    list
        Profiles (in normalised flux)
    list
        Errors on profiles (in normalised flux)
    """

    global velocities
    velocities = vgrid.copy()
    global all_frames
    all_frames = np.zeros((len(filelist), len(order_range), 2, len(velocities)))
    global linelist
    linelist = line

    global frames
    global frame_wavelengths
    global frame_errors
    global sns
    global run_name
    run_name = name

    if order_range is None:
        order_range = np.arange(10, 70)
    
    for order in order_range:
    
        print('Running for order %s/%s...'%(order-min(order_range)+1, max(order_range)-min(order_range)+1))

        frame_wavelengths, frames, frame_errors, sns, telluric_spec = read_in_frames(order, filelist, file_type)

        all_frames = ACID(frame_wavelengths, frames, frame_errors, linelist, sns, velocities, all_frames, order=order-min(order_range), **kwargs)

    # adding into fits files for each frame
    BJDs = []
    profiles = []
    errors = []
    for frame_no in range(0, len(frames)):
        file = filelist[frame_no]
        fits_file = fits.open(file)
        hdu = fits.HDUList()
        hdr = fits.Header()
        
        for order in order_range:
            hdr['ORDER'] = order
            hdr['BJD'] = fits_file[0].header['ESO DRS BJD']
            if order == order_range[0]:
                BJDs.append(fits_file[0].header['ESO DRS BJD'])
            hdr['CRVAL1']=np.min(velocities)
            hdr['CDELT1']=velocities[1]-velocities[0]

            profile = all_frames[frame_no, order-min(order_range), 0]
            profile_err = all_frames[frame_no, order-min(order_range), 1]

            hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
            if save_path != 'no save':
                month = 'August2007'
                hdu.writeto('%s%s_%s_%s.fits'%(save_path, month, frame_no, run_name), output_verify = 'fix', overwrite = 'True')

        result1, result2 = combineprofiles(all_frames[frame_no, :, 0], all_frames[frame_no, :, 1])
        profiles.append(result1)
        errors.append(result2)

    return BJDs, profiles, errors
