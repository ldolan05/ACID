#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 12:20:32 2021

@author: lucydolan
"""

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
#import math
import glob
from scipy.interpolate import interp1d
from statistics import stdev
import LSD_func_faster as LSD

def findfiles(directory, file_type):

    filelist_final = glob.glob('/home/lsd/Documents/LSD_Figures/*all_frames*.fits')
    '''
    filelist1=glob.glob('%s/*/*%s**A_corrected*.fits'%(directory, file_type))    #finding corrected spectra
    filelist=glob.glob('%s/*/*%s**A*.fits'%(directory, file_type))               #finding all A band spectra

    filelist_final=[]

    for file in filelist:                                                        #filtering out corrected spectra
        count = 0
        for file1 in filelist1:
            if file1 == file:count=1
        if count==0:filelist_final.append(file)
    '''
    return filelist_final                          #returns list of uncorrected spectra files

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
        phi = phi-1
    else:result = 'out'
    #print(result)
    if phi>0.5:
            phi = phi-1

    return result, phi, phase

def get_errors(wavelengths,fluxes):
        fluxes=fluxes
        idx = wavelengths.argsort()
        wavelength = wavelengths[idx]
        fluxe = fluxes[idx]

        frac = 0.5
        sigma = 1.5*np.median(abs(wavelengths-np.median(wavelengths)))
        sigma_lower = np.min(wavelengths)+frac*sigma
        sigma_upper = np.max(wavelengths)-frac*sigma

        #print(sigma_lower, sigma_upper)

        idx = wavelengths.argsort()

        wavelength = wavelengths[idx]
        flux = fluxe[idx]

        wavelength1 = np.array(wavelength[wavelength<sigma_lower])
        #print(len(wavelength1))
        flux1 = np.array(flux[:len(wavelength1)])
        #print(len(flux1))
        wavelength2 = np.array(wavelength[wavelength>sigma_upper])
        #print(len(wavelength2))
        flux2 = np.array(flux[-len(wavelength2):])
        #print(len(flux2))

        wavelength = np.concatenate((wavelength1, wavelength2))
        fluxe = np.concatenate((flux1, flux2))

        #print(len(wavelength))
        #print(len(fluxe))
        '''
        plt.figure()
        plt.plot(wavelengths, fluxes)
        plt.scatter(wavelength, fluxe)
        plt.show()
        '''
        #print(list(fluxe))
        mean = np.mean(fluxe)
        N = len(fluxe)
        sumof=0
        for x in fluxe:
            su = (x-mean)**2
            sumof += su

        stdev=np.sqrt(sumof/N)
        errors = [stdev]*len(fluxes)

        return errors

def ccfprofiles(ccf):
    '''
    spectra = np.zeros(np.shape(ccf[0].data))
    for order in range(len(ccf[0].data)):
        ccf_spec = ccf[0].data[order]
        #spectrum = ccf_spec/np.max(ccf_spec)-1
        spectrum = ccf_spec
        spectra[order] = spectrum

    '''
    #spectra = np.transpose(ccf_spec)

    ccf_spec = ccf[0].data[72]
    velocities=ccf[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*ccf[0].header['CDELT1'] ### This is the part for the velocities ###
    v_ccf = ccf[0].header['HIERARCH ESO DRS CCF RVC']
    v_ccf0 = ccf[0].header['HIERARCH ESO DRS CCF RV']

    #plt.figure()
    #plt.plot(velocities, ccf_spec)

    errors = get_errors(velocities, ccf_spec)

    #plt.fill_between(velocities, ccf_spec-errors, ccf_spec+errors, alpha = 0.5)
    #plt.show()
    #no_lines = ccf[0].header['HIERARCH ESO DRS CCF LINES']

    #errors = np.sqrt(ccf_spec)
    print('Not sure about errors')
    #print(v_ccf-v_ccf0)
    #print(ccf[0].header['CRVAL1'],ccf[0].header['CDELT1'])

    #K = -2.277 #km/s - Boisse et al, 2009
    #velocities = velocities - K  ### Adjusting doppler reflex ###

    return velocities, ccf_spec, errors

def remove_reflex(velocities, spectrum, errors, phi, K, e, omega, v0):
    velo = v0 + K*(e*np.cos(omega)+np.cos(2*np.pi*phi+omega))
    #print(velo)
    adjusted_velocities = velocities-velo
    f2 = interp1d(adjusted_velocities, spectrum, kind='linear', bounds_error=False, fill_value=np.nan)
    velocity_grid = np.linspace(-19,19,len(spectrum))
    adjusted_spectrum = f2(velocity_grid)
    '''
    for n in range(len(adjusted_spectrum)):
        if adjusted_spectrum[n]==np.min(adjusted_spectrum):
            if round(velocity_grid[n],2)!=0:
                vextra = 0.1875
                adjusted_velocities = velocities-velo-vextra
                f2 = interp1d(adjusted_velocities, spectrum, kind='cubic')
                velocity_grid = np.linspace(-15,15,len(spectrum))
                adjusted_spectrum = f2(velocity_grid)
            else:pass
        else:pass
     '''
    return velocity_grid, adjusted_spectrum, errors

def combineccfs(spectra):
    spectra = np.array(spectra)
    length, width = np.shape(spectra)
    spectrum = np.zeros((1,width))
    for n in range(0,width):
        spectrum[0,n]=sum(spectra[:,n])/length
    spectrum = list(np.reshape(spectrum, (width,)))
    return  spectrum

def combineprofiles(spectra, errors):
    spectra = np.array(spectra)
    errors = np.array(errors)

    plt.figure()
    for i in range(10,len(spectra)):
        plt.plot(velocities, spectra[i])
        plt.fill_between(velocities, spectra[i]-errors[i], spectra[i]+errors[i], alpha = 0.3)
    plt.show()

    length, width = np.shape(spectra)
    spectrum = np.zeros((1,width))
    spec_errors = np.zeros((1,width))
    for n in range(0,width):
        weights = 1/errors[:,n]**2
        #print(weights)
        spectrum[0,n]=sum(weights*spectra[:,n])/sum(weights)
        spec_errors[0,n]=1/sum(weights)

    plt.figure('frame: %s'%frame)
    plt.plot(velocities, spectrum[0, :])
    plt.fill_between(velocities, spectrum[0, :]-spec_errors[0, :], spectrum[0, :]+spec_errors[0, :], alpha = 0.3)
    plt.show()


    spectrum = list(np.reshape(spectrum, (width,)))
    spec_errors = list(np.reshape(spec_errors, (width,)))

    return  spectrum, spec_errors

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
        '''
        fig = plt.figure('fit')
        plt.plot(wavelengths, fluxes)
        plt.plot(wavelengths, fit)
        plt.plot(wavelengths, flux_obs)
        #plt.scatter(clipped_waves, clipped_flux, color = 'k', s=8)
        plt.show()
        '''
        return flux_obs, new_errors
####################################################################################################################################################################


P=2.21857567 #Cegla et al, 2006 - days
#rstar= 0.805*6.96*10**8 #Boyajian et al, 2015)
#a_rstar= 8.863 #Agol et al, 2010
#a = a_rstar*rstar
#rp_rstar= 0.15667 #Agol et al, 2010
#rp = rstar*rp_rstar
i = 85.71*np.pi/180 #got quickly of exoplanet.eu
T=2454279.436714 #cegla et al,2006
#b=a_rstar*np.cos(i)
#t=(P/math.pi)*np.arcsin(np.sqrt(((rp+rstar)**2-(b*rstar)**2))/a)
t=0.076125 #Torres et al, 2008
e =0
omega=(np.pi/2)
#omega = 0
K=0.20056 #km/s Boisse et al, 2009
#K=0.230
v0=-2.2765#-0.1875 #km/s Boisse et al, 2009
#v0=-2.23
#v0=-2.317 #Gaia

#rstar= 0.805*6.96*10**8 #Boyajian et al, 2015)
a_rstar= 8.863 #Agol et al, 2010
rp_rstar= 0.15667 #Agol et al, 2010
#a = a_rstar*rstar
u1=0.816
u2=0            #Sing et al, 2011
#b=a_rstar*np.cos(i)


path = '/home/lsd/Documents/LSD_Figures/'
save_path = '/home/lsd/Documents/LSD_Figures/'
month = 'August2007' #August, July, Sep

months = ['August2007',
          #'July2007',
          #'July2006',
          #'Sep2006'
          ]
#linelist = '/Users/lucydolan/Documents/Least_Squares_Deconvolution/LSD/Archive_stuff/archive/fulllinelist018.txt'
# s1d or e2ds
file_type = 'e2ds'

# order or full(can't fit properly yet as continuum fit goes to zero)
spec_type = 'order'
order = 26
masking = 'masked'
#months = ['August2007']
#months = ['July2007']
#months = ['July2006']
#months = ['Sep2006']

total= []

positions=[]
ins = 0
phases1 = []
v_obs = []
velos=[]
outliers = []
lengths = []


phases = [0.9829554832972711,
 0.014062783831011672,
 0.9933784904576477,
 0.019279504696676497,
 0.0333385608285397,
 0.003712814469444936,
 0.04198266527329153,
 0.01060410073691287]

results = ['in', 'in', 'in', 'out', 'out', 'in', 'out', 'in']

for month in months:
    plt.figure(month)
    directory =  '%s%s/'%(path,month)

    filelist = findfiles(directory, file_type)
    out_ccfs = []
    out_errors = []
    all_ccfs = []
    #phases = []

    velos1=[]
    #results = []
    #befores = []
    #afters = []
    matched=[]
    lengths.append(len(filelist))
    for frame in range(0, len(filelist)):
        file = fits.open(filelist[frame])
        order_errors = []
        order_profiles = []
        for order in range (10,70):
            profile = file[order].data[0]
            profile_errors = file[order].data[1]
            #velocities = file[order].data[2]
            velocities=np.linspace(-21,18,len(profile))
            #fluxes, wavelengths, flux_error = LSD.blaze_correct(file_type, spec_type, order, file, directory, masking)
            #velocities, profile, profile_errors = LSD.LSD(wavelengths, fluxes, flux_error, linelist)
            '''
            if frame==4:
                plt.figure()
                plt.plot(profile)
                plt.fill_between(velocities, profile-profile_errors, profile+profile_errors, alpha = 0.3)
                plt.show()
                order_errors.append(profile_errors*10)
                order_profiles.append(profile*10)
            '''
            #else:
            order_errors.append(profile_errors)
            order_profiles.append(profile)

        #print(np.shape(ccf[0].data)) #(73,161)

        #opened_file = fits.open(file)
        #result, phi, phase = classify(opened_file, P, t, T) #phi is adjusted, phase is original
        phase = phases[frame]
        if phase>0.5:
            phi = phase-1
        else:phi = phase
        result = results[frame]

        #print(result, phi, phase)
        #plt.figure(file)
        spectrum, errors = combineprofiles(order_profiles, order_errors)
        #plt.plot(velocities, spectrum)
        velocities, spectrum, errors = remove_reflex(velocities, spectrum, errors, phi, K, e, omega, v0)
        phases1.append(phi)
        #plt.plot(velocities, spectrum)
        #plt.show()
        #spectrum, errors = continuumfit(spectrum, velocities, errors, 1) # <--------------- this is continuum fit
        #break
        '''
        z = ma.MandelandAgol(phi, a_rstar, i)
        transit_curve = ma.occultquad(z, rp_rstar, [u1, u2])
        print(transit_curve)
        spectrum = spectrum*transit_curve
        errors = errors*transit_curve
        '''
        '''
        plt.figure()
        plt.plot(velocities, spectrum)
        plt.plot(velocities, spectrum+errors)
        '''
        all_ccfs.append([spectrum, errors])
        total.append(spectrum)
        #phases.append(phi)
        results.append(result)
        if result == 'out':
            #plt.plot(spectrum)
            out_ccfs.append(spectrum)
            out_errors.append(errors)
    #velos.append(velos1)
    #break
    master_out_spec, master_out_errors = combineprofiles(out_ccfs, out_errors)
    master_out = np.array([master_out_spec, master_out_errors])

    phases = np.array(phases1)
    all_ccfs = np.array(all_ccfs)
    results = np.array(results)

    idx = phases.argsort()
    phases = phases[idx]
    all_ccfs = all_ccfs[idx]
    results = results[idx]
    phases = list(phases)

    #write in data
    hdu=fits.HDUList()
    for data in all_ccfs:
        hdu.append(fits.PrimaryHDU(data=data))

    hdu.append(fits.PrimaryHDU(data=master_out))
    phases.append('out')

    #write in header
    for p in range(len(phases)):
        phase = phases[p]
        hdr=fits.Header()
        hdr['CRVAL1']=np.min(velocities)
        hdr['CDELT1']=velocities[1]-velocities[0]
        hdr['OBJECT']='HD189733b'
        hdr['NIGHT']='%s'%month
        hdr['K']=K
        hdr['V0']=v0
        hdr['PHASE']=phase
        hdu[p].header=hdr

    hdu.writeto('%s%s_master_out_LSD_profile.fits'%(save_path, month), output_verify='fix', overwrite = 'True')

    #opened_file.close()

#plt.close('all')
'''
    ins = ins+(len(all_ccfs)-len(out_ccfs))

    min_points = []
    #alll_ccfs=[]
    for c in range(len(all_ccfs)):
        ccf = all_ccfs[c]
        plt.plot(velocities, ccf)
        for vel in range(len(velocities)):
            if ccf[vel]==np.min(ccf):
                min_points.append(velocities[vel])#, results[c]])
                if round(velocities[vel], 2)!=0:outliers.append([phases1[c], velos[c]])

    plt.show()

    print(min_points)
'''
'''
#plots phases against velocity shift
plt.figure('phases')

colours = ['b', 'g', 'k', 'r', 'm']
#colours =  ['k', 'r', 'm']
n2=0

for n1 in range(0,len(lengths)):
    n3=n2
    #print(n3)
    for n2 in range(0+n3, n3+lengths[n1]):
        #print(n1, n2)
        #plt.scatter(phases1[n2], velos[n2], marker ='.')
        plt.scatter(phases1[n2], v_obs[n2], marker='x', color = colours[n1])

plt.scatter(phases1, velos, marker ='.')
#plt.scatter(phases1, v_obs, marker='x')

velos_m=[]
phases_m = np.arange(-0.1,1.1,0.001)
for phi in phases_m:
    velo = v0 + K*(e*np.cos(omega)+np.cos(2*np.pi*phi-omega))
    velos_m.append(velo)

plt.plot(phases_m, velos_m)


for out in outliers:
    phase, velo = out
    plt.scatter(phase-np.round(phase), velo)

plt.xlim(-0.05, 0.05)
plt.ylim(-2.35, -2.1)
plt.show()

plt.figure('difference')

n2=0
for n1 in range(0,len(lengths)):
    n3=n2
    #print(n3)
    for n2 in range(0+n3, n3+lengths[n1]):
        #print(n1, n2)
        #plt.scatter(phases1[n2], velos[n2], marker ='.')
        plt.plot(phases1[n2], v_obs[n2]-velos[n2], '.', color = colours[n1])

plt.ylim(-0.1,0.1)
plt.xlim(-0.05,0.05)
'''
'''
n1 = len(total)

col0=((np.round(np.linspace(0,170,num=n1))).astype(int))
#col1=((np.round(np.linspace(0,170,num=n2))).astype(int))
#col2=((np.round(np.linspace(50,220,num=n3))).astype(int))

#plt.cm.Reds(col1[i-n1])
#plt.cm.Greens(col2[i])
#plt.cm.Reds(col1[i])
#plt.cm.Blues(col0[i])

fig, ax1 = plt.subplots(1,1)

for i in range(len(total)):
    ax1.plot(velocities,total[i],color=plt.cm.gist_rainbow(col0[i]),linewidth=0.7)
plt.show()

#plt.pcolormesh()
'''

print('Completed')
