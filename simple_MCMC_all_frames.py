import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.io import  fits
import emcee
import LSD_func_faster as LSD
import glob
from scipy.interpolate import interp1d
from math import log10, floor

file_type = 'e2ds'

run_name = input('Input nickname for this version of code (for saving figures): ')
linelist = input('Enter file path to line list (VALD format):')
directory = input('Enter path to directory containing data files:')
fig_ans = input('Save Figures? (y/n):')
fig_path = input('Enter file path for directory to save figures and result files:')
if fig_path == '':
    fig_path = '/Users/lucydolan/Documents/ACID/ACID_RESULTS/'

if linelist == '':
    linelist = '/home/lsd/Documents/fulllinelist0001.txt'
if directory == '':
    print('default directory set')
    directory = '/Users/lucydolan/Starbase/HD189733/August2007/*/*/*/'

def round_sig(x, sig):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def findfiles(directory, file_type):

    filelist=glob.glob('%s*%s_A.fits'%(directory, file_type))               #finding all A band e2ds spectra
    if len(filelist)==0:
        print('Empty line list please check path: "%s*%s_A.fits"'%(directory, file_type))
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

        ## replacing negative values with a very small number and masking it out ato avoid error in optical depth conversion
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

min_order = int(input('Enter minimum order (Please exclude orders with negative/zero flux):'))
max_order = int(input('Enter maximum order (Please exclude orders with negative/zero flux):'))

if min_order == '':
    min_order = 8
if max_order == '':
    max_order = 71

order_range = np.arange(min_order, max_order)

phase_calc = 'y' #input('Include phase in fits header (y/n):')

if phase_calc == 'y' or '':
    result = input('Enter Period, Mid-Tranist Time and Transit Duration:')
    if result == '':
        P=2.21857567 #Cegla et al, 2006 - days
        T=2454279.436714 #Cegla et al, 2006
        t=0.076125 #Torres et al, 2008
    deltaphi = t/(2*P)


filelist=findfiles(directory, file_type)
poly_ord = int(input('Enter order of polynomial for continuum fit (recommended is 3):'))
if poly_ord == '':
    poly_ord = 3

for order in order_range:

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

    #### setting up empty array for final profiles to go into
    if order == order_range[0]:
        all_frames = np.zeros((len(filelist), 71, 2, len(profile)))

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

    if fig_ans == 'y':
        ## plots the model for 'walks' of the all walkers for the first 5 profile points
        samples = sampler.get_chain(discard = dis_no)
        fig, axes = plt.subplots(len(samples[0, 0, :]), figsize=(10, 7), sharex=True)
        for i in range(len(samples[0, 0, :])):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
        axes[-1].set_xlabel("step number")
        plt.savefig('%sorder%s_walker_paths_%s.png'%(fig_path, order, run_name))

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
        plt.savefig('%sorder%s_mc_mdl_%s.png'%(fig_path, order, run_name))

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

    if fig_ans == 'y':

        ## plotting optical depth profile from mcmc
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
        plt.savefig('%sorder%s_profile_%s.png'%(fig_path, order, run_name))
        

        # plots mcmc continuum fit on top of data
        plt.figure('continuum fit from mcmc')
        plt.plot(x, y, color = 'k', label = 'data')
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
        plt.savefig('%sorder%s_cont_%s.png'%(fig_path, order, run_name))

        mcmc_inputs = np.concatenate((profile, poly_cos))
        mcmc_mdl = model_func(mcmc_inputs, x)
        mcmc_liklihood = log_probability(mcmc_inputs, x, y, yerr)

        residuals_2 = (y+1) - (mcmc_mdl+1)

        fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'MCMC and true model', sharex = True)
        non_masked = tuple([yerr<10])
        ax[1].scatter(x[non_masked], residuals_2[non_masked], marker = '.')
        ax[0].plot(x, y, 'r', alpha = 0.3, label = 'data')
        ax[0].plot(x, mcmc_mdl, 'k', alpha =0.3, label = 'mcmc spec')
        residual_masks = tuple([yerr>=10000000000000000000])
        ax[0].scatter(x[residual_masks], y[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
        ax[0].legend(loc = 'lower right')
        ax[1].plot(x, [0]*len(x), '--')
        plt.savefig('%sorder%s_forward_%s.png'%(fig_path, order, run_name))

        fig, ax0 = plt.subplots()
        ax0.plot(velocities, profile, color = 'r', label = 'mcmc')
        zero_line = [0]*len(velocities)
        ax0.plot(velocities, zero_line)
        ax0.plot(velocities, model_inputs[:k_max], label = 'initial')
        ax0.fill_between(velocities, profile-profile_err, profile+profile_err, alpha = 0.3, color = 'r')
        ax0.set_xlabel('velocities')
        ax0.set_ylabel('optical depth')
        ax0.legend()
        plt.savefig('%sorder%s_final_profile_%s.png'%(fig_path, order, run_name))
        
        plt.figue() ## opening for next loop

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

        profiles.append(profile)
       
        if fig_ans == 'y':
            plt.plot(velocities, profile)
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
            if counter == len(frames)-1:
                plt.savefig('%sorder%s_CCF_profiles_%s.png'%(fig_path, order, run_name))

            plt.figure()
            plt.title('order %s, LSD profiles'%order)
            no=0
            for profile in profiles:
                plt.plot(velocities, np.exp(profile)-1, label = '%s'%no)
                no+=1
            plt.legend()
            plt.ylabel('flux')
            plt.xlabel('velocities km/s')
            if counter == len(frames)-1:
                plt.savefig('%sorder%s_FINALprofiles_%s.png'%(fig_path, order, run_name))

            mcmc_mdl = model_func(np.concatenate((profile, poly_cos)), wavelengths)
            f2 = interp1d(velocities_ccf, ccf_spec/ccf_spec[0]-1, kind='linear', bounds_error=False, fill_value=np.nan)
            ccf_profile = f2(velocities)
            ccf_mdl = model_func(np.concatenate((ccf_profile, poly_cos)), wavelengths)

            residuals_2 = (flux_b) - (mcmc_mdl)

            fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'final LSD forward and true model', sharex = True)
            non_masked = tuple([error_b<1000000000000000000])
            ax[1].scatter(wavelengths[non_masked], residuals_2[non_masked], marker = '.')
            ax[0].plot(wavelengths, flux_b, 'r', alpha = 0.3, label = 'data')
            ax[0].plot(wavelengths, mcmc_mdl, 'k', alpha =0.3, label = 'LSD spec')
            residual_masks = tuple([error_b>=1000000000000000000])
            ax[0].scatter(wavelengths[residual_masks], flux_b[residual_masks], label = 'masked', color = 'b', alpha = 0.3)
            ax[0].legend(loc = 'lower right')
            plt.savefig('%sorder%s_FINALforward_%s_%s.png'%(fig_path, order, run_name, counter))

# adding into fits files for each frame
for frame_no in range(0, len(frames)):
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

        hdr['CRVAL1']=np.min(velocities)
        hdr['CDELT1']=velocities[1]-velocities[0]
        hdr['ORDER'] = order+1

        profile = all_frames[frame_no, order, 0]
        profile_err = all_frames[frame_no, order, 1]

        hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
    hdu.writeto('%s%s_frame%s.fits'%(fig_path, run_name, frame_no), output_verify = 'fix', overwrite = 'True')
