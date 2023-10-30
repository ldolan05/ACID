#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:20:32 2021

@author: lucydolan
"""

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import math
import glob
from scipy.interpolate import interp1d
from statistics import stdev
import LSD_func_faster as LSD
from scipy.optimize import curve_fit

# run_name = input('Run name (all_frames or jvc):' )
run_name = 'newdepths'

def gauss(x, rv, sd, height, cont):
    y = cont+(height*np.exp(-(x-rv)**2/(2*sd**2)))
    return y
 
def findfiles(directory):

    filelist= glob.glob('%s*_%s.fits'%(directory, run_name))

    return filelist                       

def remove_reflex(velocities, spectrum, errors, phi, K, e, omega, v0):
    velo = v0 + K*(e*np.cos(omega)+np.cos(2*np.pi*phi+omega))
    adjusted_velocities = velocities-velo
    f2 = interp1d(adjusted_velocities, spectrum, kind='linear', bounds_error=False, fill_value='extrapolate')
    velocity_grid = np.arange(-20,20,0.82)
    adjusted_spectrum = f2(velocity_grid)
    f2 = interp1d(adjusted_velocities, errors, kind='linear', bounds_error=False, fill_value='extrapolate')
    adjusted_errors = f2(velocity_grid)
    
    return velocity_grid, adjusted_spectrum, adjusted_errors

def combineprofiles(spectra, errors, master, velocities):
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

    if master == 'no':

        weights_csv = np.genfromtxt('%sorder_weights.csv'%file_path, delimiter=',')
        orders = np.array(weights_csv[:len(spectra),0], dtype = int)
        weights_temp = np.array(weights_csv[:len(spectra):,1])
        spectra_to_combine = []
        errorss = []
        weights = []
        for i in orders:
            if np.sum(spectra[i-1])!=0:
                spectra_to_combine.append(list(spectra[i-1]))
                errorss.append(list(errors[i-1]))
                weights.append(weights_temp[i-1])
        weights = np.array(weights/sum(weights))

    else:
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

    for n in range(0,width):
        temp_spec = spectra_to_combine[:, n]
        spectrum[0,n]=sum(weights*temp_spec)/sum(weights)
        spec_errors[0,n]=(stdev(temp_spec)**2)*np.sqrt(sum(weights**2))

    spectrum = list(np.reshape(spectrum, (width,)))
    spec_errors = list(np.reshape(spec_errors, (width,)))

    return  spectrum, spec_errors, weights

def continuumfit(fluxes, wavelengths, errors, poly_ord):
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
        fit = poly(wavelengths)
        flux_obs = fluxes/fit
        new_errors = errors/fit

        return flux_obs, new_errors


def classify(ccf, P, t, T):
    #working out if in- or out- of transit
    #print(t, b)
    deltaphi = t/(2*P)
    #print(deltaphi)
    phi = (((ccf[0].header['ESO DRS BJD'])-T)/P)%1
    #z = np.sqrt((a_rstar*(np.sin(2*math.pi*phi))**2)+b**2*(np.cos(2*math.pi*phi))**2)
    #part1=a_rstar*np.sin(2*math.pi*phi)**2
    #part2=b**2*np.cos(2*math.pi*phi)**2
    #z=np.sqrt(part1+part2)
    #print(z, phi)
    '''
    if z < 1+rp_rstar:
        result = 'in'
    else:result = 'out'
    '''
    phase = phi
    if phi < deltaphi:
        result ='in'
    elif phi > 1-deltaphi:
        result = 'in'
    else:result = 'out'
    #print(result)
    if phi>0.5:
        phi = phi-1

    return result, phi, phase
####################################################################################################################################################################

# Parameters for HD189733b

P=2.21857567 #Cegla et al, 2006 - days
i = 85.71*np.pi/180 #got quickly of exoplanet.eu
T=2454279.436714 #cegla et al,2006
t=0.076125 #Torres et al, 2008
e =0
omega=(np.pi/2)
K=0.20056 #km/s Boisse et al, 2009
v0=-2.2765#-0.1875 #km/s Boisse et al, 2009

a_rstar= 8.863 #Agol et al, 2010
rp_rstar= 0.15667 #Agol et al, 2010
u1=0.816
u2=0            #Sing et al, 2011
P=2.21857567 #Cegla et al, 2016 - days
T=2454279.436714 #cegla et al,2016
a_Rs = 8.786 #Cristo et al - 8.786
b=0.687 #Cristo et al, 2022
RpRs = 0.15667

# path = '/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/'
# save_path = '/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/'

path = '/Users/lucydolan/Starbase/newdepths/'
file_path = '/Users/lucydolan/Starbase/'

months = ['August2007',
          'July2007',
          'July2006',
          'Sep2006'
          ]

for month in months:
    filelist = findfiles('%s%s'%(path,month))
    velocities=np.arange(-21, 18, 0.82)

    frame_profiles = np.zeros((len(filelist), 3, len(np.arange(-20,20,0.82))))
    for frame_no in range(len(filelist)):
        file = fits.open(filelist[frame_no])
        order_errors = []
        order_profiles = []

        for order1 in range(0,71):

            profile_errors = file[order1].data[1]
            profile = file[order1].data[0]

            if order1>38 and order1<43:
                print('ignoring order %s'%order1)
                profile_errors = np.zeros(profile_errors.shape)
                profile = np.zeros(profile.shape)
            
            if order1==0:
                phase = file[order1].header['PHASE']
            elif file[order1].header['PHASE']!=file[order1-1].header['PHASE']:
                raise ValueError('Phase does not match for the same frame!')

            order_errors.append(profile_errors)
            order_profiles.append(profile)

        spectrum, errors, weights = combineprofiles(order_profiles, order_errors, 'no', velocities)
        
        final_velocities, spectrum, errors = remove_reflex(velocities, spectrum, errors, phase, K, e, omega, v0)

        frame_profiles[frame_no] = [[phase-np.round(phase)]*len(spectrum), spectrum, errors]

    phi = frame_profiles[:, 0, 0]
    z = np.sqrt( ( (a_Rs)*np.sin(2 * np.pi * phi) )**2 + ( b*np.cos(2. * np.pi * phi))**2)

    out_idx = (z>1+RpRs)
    if len(frame_profiles[out_idx, 0])>1:
        master_out_spec, master_out_errors, master_weights = combineprofiles(frame_profiles[out_idx, 1], frame_profiles[out_idx, 2], 'yes', final_velocities)
    else:
        master_out_spec = frame_profiles[:, 1]
        master_out_errors = frame_profiles[:, 2]

    master_out = np.array([master_out_spec, master_out_errors])

    #write in data
    hdu=fits.HDUList()
    
    #write in header
    for p in range(len(frame_profiles)):
        hdu.append(fits.PrimaryHDU(data=frame_profiles[p, 1:]))
        phase = frame_profiles[p, 0, 0]
        phi = phase
        z = np.sqrt( ( (a_Rs)*np.sin(2 * np.pi * phi) )**2 + ( b*np.cos(2. * np.pi * phi))**2)
        if z>1+RpRs:
            result = 'out'
        else: result = 'in'
        hdr=fits.Header()
        hdr['CRVAL1']=np.min(final_velocities)
        hdr['CDELT1']=final_velocities[1]-final_velocities[0]
        hdr['OBJECT']='HD189733b'
        hdr['NIGHT']='%s'%month
        hdr['K']=K
        hdr['V0']=v0
        hdr['PHASE']=phase
        hdr['RESULT']=result
        hdu[p].header=hdr
    
    hdu.append(fits.PrimaryHDU(data=master_out))
    hdr=fits.Header()
    hdr['CRVAL1']=np.min(final_velocities)
    hdr['CDELT1']=final_velocities[1]-final_velocities[0]
    hdr['OBJECT']='HD189733b'
    hdr['NIGHT']='%s'%month
    hdr['K']=K
    hdr['V0']=v0
    hdr['PHASE']='master out'
    hdr['RESULT']='out'
    hdu[p+1].header=hdr

    hdu.writeto('%s%s_master_out_LSD_profile.fits'%(path, month), output_verify='fix', overwrite = 'True')

    rvs = []
    for y in range(len(frame_profiles[:])):
        ACID_profile = frame_profiles[y, 1]
        ACID_errors = frame_profiles[y, 2]
        popt, pcov = curve_fit(gauss, final_velocities[abs(final_velocities)<5], ACID_profile[abs(final_velocities)<5], sigma = ACID_errors[abs(final_velocities)<5], absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        rvs.append([popt[0], perr[0]])

    rvs = np.array(rvs)
    plt.figure('RVs')
    phi = frame_profiles[:, 0, 0]
    z = np.sqrt( ( (a_Rs)*np.sin(2 * np.pi * phi) )**2 + ( b*np.cos(2. * np.pi * phi))**2)
    out_idx = (z>1+RpRs)
    plt.errorbar(frame_profiles[:, 0, 0], rvs[:, 0], rvs[:, 1], marker ='o', color = 'orange', linestyle='')
    plt.show()
