## read in files
## apply basic continuum correction (using polyfit) 
## run LSD function
## plot rvs
import glob
import LSD_func_faster as LSD
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import emcee
from math import log10, floor
from scipy.interpolate import interp1d

def gauss(x, rv, sd, height, cont):
    y = height*np.exp(-(x-rv)**2/(2*sd**2)) + cont
    return y

def round_sig(x1, sig):
    return round(x1, sig-int(floor(log10(abs(x1))))-1)

def findfiles(directory, file_type):
    filelist=glob.glob('%s*%s**A*.fits'%(directory, file_type)) 
    return filelist

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
        idx = tuple([flux_obs<=0])
        flux_obs[idx] = 0.00000000000000001
        new_errors[idx]=1000000000000000
        
        coeffs=coeffs[::-1]
        coeffs = list(coeffs)
        coeffs.append(cont_factor)
        coeffs = np.array(coeffs)
        
        return coeffs, flux_obs, new_errors, fit

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

def task(all_frames, filelist, order, file_no):

    #read in file
    file = filelist[file_no]   
    fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct('e2ds', 'order', order, file, directory, 'unmasked', run_name, 'y')
    fits_file = fits.open(file)
    try:phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
    except:phi = (((fits_file[0].header['TNG DRS BJD'])-T)/P)%1
    phase[file_no]=phi
    plt.figure()
    plt.title('Blaze corrected spectrum')
    plt.plot(wavelengths, fluxes)
    plt.xlabel('wavelengths')
    plt.ylabel('flux')
    plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/bervbeforenew_blaze_corrected_spec_order%s_file%s.png'%(order, file_no))
    plt.close()
    print('File_no: %s, done reading in'%file_no)

    #apply basic continuum correction
    a = 2/(np.max(wavelengths.copy())-np.min(wavelengths.copy()))
    b = 1 - a*np.max(wavelengths.copy())
    poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes, (wavelengths.copy()*a)+b, flux_error_order, poly_ord)
    plt.figure()
    plt.title('Continuum corrected spectrum')
    plt.plot(wavelengths, fluxes)
    plt.xlabel('wavelengths')
    plt.ylabel('flux')
    plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/bervbeforenew_cont_corrected_spec_order%s_file%s.png'%(order, file_no))
    plt.close()
    print('File_no: %s, done reading in'%file_no)
    # poly_inputs, fluxes1, flux_error_order1, fit = continuumfit(fluxes, wavelengths, flux_error_order, poly_ord)
    # fluxes1 = fluxes.copy()/fluxes[0]
    # flux_error_order1=flux_error_order.copy()/fluxes[0]
    print('File_no: %s, done continuum correction'%file_no)

    #### getting the initial profile
    # global alpha
    print('File_no: %s, starting LSD'%file_no)
    vel, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes1, flux_error_order1, linelist, 'False', poly_ord, sn, order, run_name, velocities)
    plt.figure()
    plt.title('LSD Profile')
    plt.plot(velocities, profile)
    plt.xlabel('velocities')
    plt.ylabel('flux')
    plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/bervbeforenew_profile_order%s_file%s.png'%(order, file_no))
    plt.close()

    print('File_no: %s, LSD done'%file_no)

    # ## more complex continuum fit
    # global j_max, k_max
    # j_max = int(len(fluxes))
    # k_max = len(profile)

    # model_inputs = np.concatenate((profile, poly_inputs))

    # ## setting number of walkers and their start values(pos)
    # ndim = len(model_inputs)
    # nwalkers= ndim*3
    # rng = np.random.default_rng()

    # ### starting values of walkers with independent variation
    # sigma = 0.8*0.005
    # pos = []
    # for i in range(0, ndim):
    #     if i <ndim-poly_ord-2:
    #         pos2 = rng.normal(model_inputs[i], sigma, (nwalkers, ))
    #     else:
    #         print(model_inputs[i])
    #         sigma = abs(round_sig(model_inputs[i], 1))/10
    #         print(sigma)
    #         # print(sigma_cont[i-k_max])
    #         pos2 = rng.normal(model_inputs[i], sigma, (nwalkers, ))
    #     pos.append(pos2)

    # pos = np.array(pos)
    # pos = np.transpose(pos)

    # ## the number of steps is how long it runs for - if it doesn't look like it's settling at a value try increasing the number of steps
    # steps_no = 10000
    # #with mp.Pool() as pool:
    # sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(wavelengths, fluxes, flux_error_order))
    # sampler.run_mcmc(pos, steps_no, progress=True)

    # dis_no = int(np.floor(steps_no-1000))

    # flat_samples = sampler.get_chain(discard=dis_no, flat=True)

    # profile = []
    # poly_cos = []
    # profile_err = []
    # poly_cos_err = []

    # for i in range(ndim):
    #     mcmc = np.median(flat_samples[:, i])
    #     error = np.std(flat_samples[:, i])
    #     mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
    #     error = np.diff(mcmc)
    #     if i<k_max:
    #         profile.append(mcmc[1])
    #         profile_err.append(np.max(error))
    #     else:
    #         poly_cos.append(mcmc[1])
    #         poly_cos_err.append(np.max(error))

    # profile = np.array(profile)
    # profile_err = np.array(profile_err)


    # a = 2/(np.max(wavelengths)-np.min(wavelengths))
    # b = 1 - a*np.max(wavelengths)
    
    # mdl1 =0
    # for i in np.arange(0, len(poly_cos)-1):
    #     mdl1 = mdl1+poly_cos[i]*((a*wavelengths)+b)**(i)
    # mdl1 = mdl1*poly_cos[-1]

    # vel, profile, profile_err, alpha, continuum_waves, continuum_flux, no_line= LSD.LSD(wavelengths, fluxes/mdl1, flux_error_order1/mdl1, linelist, 'False', poly_ord, sn, order, run_name, velocities)

    all_frames[file_no, order]=[profile, profile_err]

    return all_frames, phase

file_type = 'e2ds'
linelist = '/home/lsd/Documents/55 Cnc/55Cnc_lines.txt'
directory = '/home/lsd/Documents/55 Cnc/group 2/*/*/*/'
save_path = '/home/lsd/Documents/55 Cnc/results/'
filelist=findfiles(directory, file_type)
run_name = 'test'

P=0.737
T=2455962.0697 
t=1.6/24
deltaphi = t/(2*P)
poly_ord = 3

velocities=np.arange(6, 50, 0.82)
global all_frames
all_frames = np.zeros((len(filelist), 71, 2, len(velocities)))
global phase
phase = np.zeros((len(filelist)))
for order in range(30, 40): 
    task_part = partial(task, all_frames, filelist, order)
    with mp.Pool(mp.cpu_count()) as pool:results=[pool.map(task_part, np.arange(len(filelist)))]
    for i in range(len(filelist)):
        all_frames[i, order]=results[0][i][0][i][order]
        phase[i]=results[0][i][1][i]

# combine LSD profiles for each frame
LSD_profiles = np.zeros((len(filelist), len(velocities)))
berv = []
plt.figure()
plt.title('Average LSD Profiles')
for file_no in range(len(filelist)):
    file = fits.open(filelist[file_no])
    berv.append(file[0].header['ESO DRS BERV'])
    LSD_prof = np.mean(all_frames[file_no, 30:40, 0], axis = 0)
    # f2 = interp1d(velocities+file[0].header['ESO DRS BERV'], LSD_prof, kind='linear', bounds_error=False, fill_value='extrapolate')
    # new_velocities = np.arange(6, 50, 0.82)
    # LSD_profiles[file_no] = f2(new_velocities)
    LSD_profiles[file_no]=LSD_prof
    plt.plot(velocities, LSD_profiles[file_no])
plt.xlabel('velocities')
plt.ylabel('flux')
plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/bervbeforenew_av_profiles.png')
plt.close()

# velocities=new_velocities
# fit RVs of LSD profiles
rvs = []
st = 20
end = -20
for profile in LSD_profiles:
    #profile = np.exp(profile)-1 ## convert from optical depth into flux
    plt.figure()
    plt.plot(velocities[st:end], profile[st:end], color = 'k', label = 'LSD Profile')
    popt, pcov = curve_fit(gauss, velocities[st:end], profile[st:end], p0 = [27, 1, -1, 0])
    perr= np.sqrt(np.diag(pcov))
    plt.plot(velocities[st:end], gauss(velocities[st:end], *popt), color = 'r', label = 'Gaussian Fit')
    plt.legend()
    plt.xlabel('velocities')
    plt.ylabel('flux')
    plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/bervbeforenew_profile_fit_fileno%s..png'%(len(rvs)))
    plt.close()
    rvs.append(popt[0])

plt.figure()
plt.scatter(berv, rvs-np.median(rvs))
plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/berv_tests/bervvsrvs.png')

plt.figure()
plt.scatter(np.array(berv)/0.82, rvs-np.median(rvs))
plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/berv_tests/berv_pixvsrvs.png')

plt.figure()
plt.scatter(phase-np.round(phase), rvs-np.median(rvs))
plt.savefig('/home/lsd/Documents/55 Cnc/TEST_FIGS/bervbeforenew_rvs.png')
plt.show()
