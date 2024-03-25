import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import  fits
import emcee
#import corner
# import LSD_func_faster as LSD
import time
import random
import glob
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import multiprocessing as mp
from functools import partial
import ACID_code.ACID as acid
from matplotlib.colors import ListedColormap
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
from matplotlib import colors as cl
from scipy.optimize import curve_fit

def remove_berv(velocities, spectrum, berv):

    velo = berv/1000
    adjusted_velocities = velocities-velo
    f2 = interp1d(adjusted_velocities, spectrum, kind='linear', bounds_error=False, fill_value='extrapolate')
    velocity_grid = velocities-velo
    adjusted_spectrum = f2(velocity_grid)
    adjusted_spectrum = np.array(adjusted_spectrum)

    # returning clipped profiles to remove noise induced from the edge of orders (??) - CHECK
    return velocity_grid, adjusted_spectrum

plt.rcParams.update({'font.family': 'serif'})
# plt.style.use('seaborn-v0-colorblind')
colors = sns.color_palette("Dark2")
cmap = ListedColormap(sns.color_palette("Dark2"))

def findfiles(directory, file_type):

    filelist1=glob.glob('%s/*/*%s**A_corrected*.fits'%(directory, file_type))    #finding corrected spectra
    filelist=glob.glob('%s/*%s.fits.fz'%(directory, file_type))               #finding all A band spectra

    filelist_final=[]

    for file in filelist:                                                        #filtering out corrected spectra
        count = 0
        for file1 in filelist1:
            if file1 == file:count=1
        if count==0:filelist_final.append(file)

    return filelist

def calc_reflex_sig(phase, K=0.20056, v0=-2.2765):
    e =0
    omega = np.pi/2
    return (v0 + K*(e*np.cos(omega)+np.cos(2*np.pi*phase+omega)))

def read_in_frames(order, filelist):
    frames = []
    errors = []
    frame_wavelengths = []
    sns = []
    global berv
    berv = []
    max_sn = 0
    phases = []

    ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
    # plt.figure('spectra after blaze_correct')
    # finding length:
    length = 1000000000
    for file in filelist:
        fits_file = fits.open(file)
        idx = tuple([fits_file[1].data['order'][1::2]==order])

        fluxes = fits_file[1].data['flux'][1::2][idx]
        wavelengths = fits_file[1].data['wavelength'][1::2][idx]
        idx2 = tuple([wavelengths!=0])
        leng = len(fluxes[idx2][500:-500])
        
        if leng<length:
            length = leng

    for file in filelist:

        ## opening NRES files
        fits_file = fits.open(file)
        idx = tuple([fits_file[1].data['order'][1::2]==order])

        fluxes = fits_file[1].data['flux'][1::2][idx]
        wavelengths = fits_file[1].data['wavelength'][1::2][idx]
        flux_error_order = fits_file[1].data['uncertainty'][1::2][idx]
        normflux = fits_file[1].data['normflux'][1::2][idx]
        blaze = fits_file[1].data['blaze'][1::2][idx]
        fluxes = fluxes/blaze
        sn = fits_file[0].header['SNR']
        T=2457186.91451
        P = 542.08

        if fits_file[0].header['AGU1POSN']=='CALIBRATION':
            plt.figure()
            plt.plot(wavelengths, fluxes, color = colors[0])
            plt.xlabel('Wavelengths Å')
            plt.ylabel('Flux')
            plt.savefig('calibration_spec.png')
            plt.close('all')
            continue
            
        phases.append(((fits_file[0].header['TCORR']-T)/P)%1)
        # phases.append(fits_file[0].header['TCORR'])

        # if sn<20:continue

        # doing berv correction
        idx2 = tuple([wavelengths!=0])
        berv = -fits_file[0].header['BARYCORR']/1000
        final_wavelengths = wavelengths[idx2][500:-500][:length]*(1.+berv/2.99792458e5)

        # correction for systemic velocity
        # K = 
        # v0 = 

        # sys = calc_reflex_sig(fits_file[0].header['UT1-UTC'], K, v0) 
        # final_wavelengths = final_wavelengths *(1.+sys/2.99792458e5)

        # idx2 = tuple([wavelengths!=0])
        frame_wavelengths.append(final_wavelengths)
        frames.append(fluxes[idx2][500:-500][:length])
        errors.append(flux_error_order[idx2][500:-500][:length])
        sns.append(sn)
        # berv.append(fits_file[0].header['BARYCORR']/1000)


        ### finding highest S/N frame, saves this as reference frame
        if sn>max_sn:
            max_sn = sn
            global reference_wave
            global ref_berv
            global reference_frame
            ref_berv = -fits_file[0].header['BARYCORR']/1000
            reference_wave = wavelengths[idx2][500:-500]#*(1.+ref_berv/2.99792458e5)
            reference_frame=fluxes[idx2][500:-500]
            reference_frame[reference_frame == 0]=0.001
            reference_error=flux_error_order[idx2][500:-500]
            reference_error[reference_frame == 0]=1000000000000000000

    frames = np.array(frames)
    errors = np.array(errors)

    global frames_unadjusted
    frames_unadjusted = frames
    global frame_errors_unadjusted
    frame_errors_unadjusted = errors

    # plt.figure()
    # for n in range(len(frames)):
    #     plt.errorbar(frame_wavelengths[n], frames[n], errors[n])
    # plt.show()

    # plt.figure('divided spectra (by reference frame)')
    ## each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
    popts = []
    for n in range(len(frames)):
        f2 = interp1d(frame_wavelengths[n], frames[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
        div_frame = f2(reference_wave)/reference_frame

        # bins = np.linspace(0, 1, floor(len(div_frame)/5))
        # digitized = np.digitize(div_frame, bins)
        # binned = [div_frame[digitized == i].mean() for i in range(1, len(bins))]
        # #digitized = np.digitize(div_frame, bins)
        # binned_waves = [reference_wave[digitized == i].mean() for i in range(1, len(bins))]

        # binned = np.array(binned)
        # binned_waves = np.array(binned_waves)

        ## creating windows to fit polynomial to
        binned = np.zeros(int(len(div_frame)/5))
        binned_waves = np.zeros(int(len(div_frame)/5))
        pos=-1
        for i in range(0, len(div_frame)-5, 5):
            pos +=1
            binned[pos] = np.nanmean(div_frame[i:i+4])
            binned_waves[pos] = np.nanmean(reference_wave[i:i+4])

        # print(len(binned_waves))
        binned_waves = binned_waves[np.logical_and(binned<=2, binned>=-2)]
        binned = binned[np.logical_and(binned<=2, binned>=-2)]

        m = np.median(binned)
        sigma = np.std(binned)
        a = 1

        upper_clip = m+a*sigma
        lower_clip = m-a*sigma

        rcopy = binned.copy()

        idx = np.logical_and(rcopy>=lower_clip, rcopy<=upper_clip)
        #print(idx)
        binned = binned[idx]
        binned_waves = binned_waves[idx]

        ### fitting polynomial to div_frame
        coeffs=np.polyfit(binned_waves, binned, 1)
        poly = np.poly1d(coeffs)
        fit = poly(frame_wavelengths[n])

        # plt.figure()
        # plt.plot(frame_wavelengths[n], frames[n]/poly(frame_wavelengths[n]*(1.+berv[n]/2.99792458e5)), label = 'berv corrected')
        # #plt.plot(frame_wavelengths[n]/(1.+berv[n]/2.99792458e5), frames[n]/fit)
        # plt.plot(frame_wavelengths[n], frames[n]/fit)
        # plt.show()

        m = np.median(frames[n])
        sigma = np.std(frames[n])
        a = 1

        upper_clip = m+a*sigma
        lower_clip = m-a*sigma

        rcopy = frames[n].copy()

        idx = (np.logical_and(rcopy>=0, rcopy<=upper_clip)==False)

        frames[n] = frames[n]/fit
        errors[n] = errors[n]/fit

        frames[n][idx]=np.median(frames[n])
        errors[n][idx]=100000000000.

        # plt.figure()
        # plt.scatter(frame_wavelengths[n], frames[n])
        # plt.scatter(frame_wavelengths[n][idx], frames[n][idx])
        # plt.show()
        # print('done')

        # plt.figure()
        # plt.scatter(binned_waves, binned)
        # plt.show()

        # plt.figure()
        # #plt.plot(reference_wave, div_frame)
        # plt.scatter(binned_waves, binned)
        # plt.plot(frame_wavelengths[n], fit)
        # plt.show()

    return frame_wavelengths, frames, errors, sns, phases

order_range = np.arange(70,108)
# order_range = np.arange(81,82)

P=542
T=2457186.91451
t=13/24 
deltaphi = t/(2*P)

linelist = '/Users/lucydolan/Starbase/NRESfulllinelist0001.txt'
directory_p = '/Users/lucydolan/Documents/HIP41378f/HIP41378f_NRES_data_for_upload/'
directory = '%s'%(directory_p)
print(directory)
filelist=findfiles(directory, '1d')
print(filelist)
phasess=[]
poptss=[]
# temp_file = fits.open(filelist[0])
# offset = (-1)*temp_file[0].header['ESO DRS BERV']
deltav = 1.5
velocities=np.arange(70, 150, deltav)
new_velocities=velocities.copy()

result = np.zeros((len(filelist), len(order_range), 2, len(velocities)))

average_profiles = []
for order in order_range:
    print(order)
    poly_ord = 6

    ### read in spectra for each frame, along with S/N of each frame
    # for file in filelist:
    #     fits_file = fits.open(file)
    #     phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
    #     print(phi)
    #     print(file, order)
    #     fluxes, wavelengths, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory1, 'unmasked', run_name, 'y')
    #     fluxes1, wavelengths1, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory2, 'unmasked', run_name, 'n')
    frame_wavelengths, frames, frame_errors, sns, new_phases = read_in_frames(order, filelist)
    
    new_phases = new_phases - np.round(new_phases)

    # for i in range(len(frame_wavelengths)):
    plt.figure(figsize = (15, 7))
    plt.plot(frame_wavelengths[3], frames[3]/frames[3][0], color = colors[0])
    plt.xlabel('Wavelengths Å')
    plt.ylabel('Flux')
    # plt.ylim(3*10**6, 10*10**6)
    plt.savefig('NRES_spec.png')
    plt.close('all')
    # plt.show()

    if order == min(order_range):
        result = np.zeros((len(frames), len(order_range), 2, len(velocities)))
        phases = new_phases
    if any(phases != new_phases):
        raise ValueError('Phases dont match')

    result = acid.ACID(frame_wavelengths, frames, frame_errors, frame_sns = sns, line=linelist, vgrid = velocities, all_frames=result, order=order-min(order_range), poly_or=poly_ord)

    # result = list(result).copy()
    # filtered_result = [result[r] for r in range(len(result)) if (result[r][order-min(order_range)][0]<1.).all()]
    # filtered_result = np.array(filtered_result)
    result = np.array(result)

    weights = (1/result[:, :, 1, :]**2)
    idx = (result[:, :, 1, :]==0)
    weights[idx]=0
    average_profiles=[np.average(result[:, :, 0, :], weights = weights, axis=1), np.sqrt(1/np.sum(weights, axis = 1))]

    hdu = fits.HDUList()
    hdu.append(fits.PrimaryHDU(data = np.array(average_profiles)))
    hdu.append(fits.PrimaryHDU(data = velocities))
    hdu.append(fits.PrimaryHDU(data = phases))
    hdu.append(fits.PrimaryHDU(data = result))
    # hdu.append(average_profiles)
    hdu.writeto('NRES_ACID_noberv.fits', overwrite = True)

import matplotlib.ticker as tkr

print('done')

f = fits.open('NRES_ACID_noberv.fits')
profiles = f[0].data[0]
# profile_errors = np.std(profiles, axis =1)
profile_errors = f[0].data[1]
velocities = f[1].data
phases = f[2].data

idx = phases.argsort()
profiles = profiles[idx]
profile_errors = profile_errors[idx]
phases = phases[idx]

import seaborn as sns
# cmap = sns.cubehelix_palette(start=.4, rot=-.4, reverse=True, as_cmap = True, n_colors = len(profiles))
colors_cmap = sns.cubehelix_palette(start=.4, rot=-.4, reverse=True, n_colors=len(profiles))
cmap = sns.cubehelix_palette(start=.4, rot=-.4, reverse=True, as_cmap = True)

fig, ax = plt.subplots(figsize = (8, 6))
for i in range(len(profiles)):
    if min(profiles[i])<5:
        # print(phases[i])
        ax.errorbar(velocities, profiles[i]+1, profile_errors[i], color = colors_cmap[i])
    else:
        print(phases[i])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size="7%", pad=0.2,)
norm = cl.BoundaryNorm(phases, cmap.N)
# ticks = np.arange(-0.02, 0.06, 0.02)
cb1 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax, format=tkr.FormatStrFormatter('%.2e'))
ax.set_ylabel('Flux')
ax.set_xlabel('Velocities (km/s)')
ax.set_ylim(1-0.3, 1+0.05)
# ax.set_xlim(45, 110)
plt.savefig('ACID_NRES_test2.png')


from scipy.optimize import curve_fit
from scipy.special import wofz

def voigt_wofz(u, a):
    prof=wofz(u + 1j * a).real
    return prof

def voigt_func(x, sigma, gamma, rv, height, cont):
    a = np.sqrt(gamma**2)/np.sqrt(2)*np.sqrt(sigma**2)
    u = (x-rv)/np.sqrt(2)*np.sqrt(sigma**2)
    mdl=voigt_wofz(u,a)
    return (cont)+height*mdl

new_profiles = []
new_velocities = []
plt.figure('RVs', figsize = (8, 6))
lim = 5
deltaphi = 0.0007
for i in range(len(profiles)):
    f = fits.open(filelist[i])
    # if f[0].header['MJD-OBS']>59900:continue
    # else:print('included')
    berv = 0
    new_v, prof = remove_berv(velocities, profiles[i], berv)
    new_profiles.append(prof)
    new_velocities.append(new_v)
    # new_v, prof = remove_berv(new_v, prof, -berv)
    # if min(profiles[i])<-0.15:
    popt, pcov = curve_fit(voigt_func, new_v[lim:-lim], prof[lim:-lim]+1, sigma = profile_errors[i][lim:-lim], absolute_sigma=True, maxfev = 120000, p0 = [3, 3, 110, -0.4, 1])
    perr = np.sqrt(np.diag(pcov))
    if popt[2]<1000:
        plt.errorbar(phases[i], popt[2], yerr = perr[2]*1000, color = colors[0], marker = 'o')
plt.xlabel('Phase')
plt.ylabel('RV (km/s)')
plt.vlines([-deltaphi, deltaphi], ymin=105, ymax=106.5, colors = colors[2], alpha = 0.5, linestyle = '--')
plt.xlim(-0.001, 0.001)
plt.ylim(105.7, 106.3)
plt.savefig('NRES_ACID_RVs.png')

print('done')

new_profiles = np.array(new_profiles)
new_velocities = np.array(new_velocities)
fig, ax = plt.subplots(figsize = (8, 6))
for i in range(len(new_profiles)):
    if min(profiles[i])<5:
        # print(phases[i])
        ax.errorbar(new_velocities[i], new_profiles[i]+1, profile_errors[i], color = colors_cmap[i])
    else:
        print(phases[i])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size="7%", pad=0.2,)
norm = cl.BoundaryNorm(phases, cmap.N)
# ticks = np.arange(-0.02, 0.06, 0.02)
cb1 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax = cax, format=tkr.FormatStrFormatter('%.2e'))
ax.set_ylabel('Flux')
ax.set_xlabel('Velocities (km/s)')
ax.set_ylim(1-0.3, 1+0.05)
# ax.set_xlim(45, 110)
plt.savefig('ACID_NRES_berv_new.png')
