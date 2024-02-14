import numpy as np
import emcee
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import  fits
import emcee
#import corner
import LSD_func_faster as LSD
import time
import random
import glob
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import multiprocessing as mp
from functools import partial
import ACID_code.ACID as acid


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
        phases.append(fits_file[0].header['UT1-UTC'])

        if fits_file[0].header['AGU1POSN']=='CALIBRATION':continue

        # if sn<20:continue

        berv = fits_file[0].header['BARYCORR']/1000

        idx2 = tuple([wavelengths!=0])
        frame_wavelengths.append(wavelengths[idx2][500:-500][:length]*(1.+berv/2.99792458e5))
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
            ref_berv = fits_file[0].header['BARYCORR']/1000
            reference_wave = wavelengths[idx2][500:-500]*(1.+ref_berv/2.99792458e5)
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
        a = 3.5

        upper_clip = m+a*sigma
        lower_clip = m-a*sigma

        rcopy = frames[n].copy()

        idx = (np.logical_and(rcopy>=lower_clip, rcopy<=upper_clip)==False)

        frames[n] = frames[n]/fit
        errors[n] = errors[n]/fit

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

order_range = np.arange(70,109)
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
velocities=np.arange(-80, 25, deltav)
new_velocities=velocities.copy()

result = np.zeros((len(filelist), len(order_range), 2, len(velocities)))

average_profiles = []
for order in order_range:
    print(order)
    poly_ord = 3

    ### read in spectra for each frame, along with S/N of each frame
    # for file in filelist:
    #     fits_file = fits.open(file)
    #     phi = (((fits_file[0].header['ESO DRS BJD'])-T)/P)%1
    #     print(phi)
    #     print(file, order)
    #     fluxes, wavelengths, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory1, 'unmasked', run_name, 'y')
    #     fluxes1, wavelengths1, flux_error_order, sn, mid_wave_order = LSD.blaze_correct('e2ds', 'order', order, file, directory2, 'unmasked', run_name, 'n')
    frame_wavelengths, frames, frame_errors, sns, new_phases = read_in_frames(order, filelist)
    
    if order == min(order_range):
        result = np.zeros((len(frames), len(order_range), 2, len(velocities)))
        phases = new_phases
    if phases != new_phases:
        raise ValueError('Phases dont match')

    result = acid.ACID(frame_wavelengths, frames, frame_errors, frame_sns = sns, line=linelist, vgrid = velocities, all_frames=result, order=order-min(order_range))

    result = list(result).copy()
    filtered_result = [result[r] for r in range(len(result)) if (result[r][0][0]<1.).all()]
    filtered_result = np.array(filtered_result)
    result = np.array(result)

    weights = (1/filtered_result[:, 0, 1, :]**2)
    average_profiles.append([np.average(filtered_result[:, 0, 0, :], weights = weights, axis =0), np.sqrt(1/np.sum(weights, axis = 0))])

    hdu = fits.HDUList()
    hdu.append(fits.PrimaryHDU(data = np.array(average_profiles)))
    hdu.append(fits.PrimaryHDU(data = velocities))
    hdu.append(fits.PrimaryHDU(data = phases))
    hdu.writeto('NRES_ACID.fits', overwrite = True)

print('done')