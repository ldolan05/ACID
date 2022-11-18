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
from scipy.special import erf
from matplotlib.pyplot import cm

run_name = input('Run name (all_frames or jvc):' )

def gauss(x, rv, sd, height, cont):
    y = height*np.exp(-(x-rv)**2/(2*sd**2)) + cont
    return y

def skewnormal(x, scaleheight, omega, gamma, alpha, zeroorder):
    result = zeroorder+(scaleheight*(2/omega)*np.exp(-(((x-gamma)/omega)**2)/2)/(np.sqrt(2*np.pi))*0.5*(erf((((alpha*(x-gamma))/omega))/(np.sqrt(2)))+1))
    return result
 
def findfiles(directory, file_type):

    filelist_final = glob.glob('%s*_%s.fits'%(directory, run_name))

    if len(filelist_final)>40:
        filelist=glob.glob('%s*_%s.fits'%(directory, run_name))    #finding corrected spectra
        filelist1=glob.glob('%s*_syn_%s.fits'%(directory, run_name))               #finding all A band spectra

        filelist_final=[]

        for file in filelist:                                                        #filtering out corrected spectra
            count = 0
            for file1 in filelist1:
                if file1 == file:count=1
            if count==0:filelist_final.append(file)
    
    # directory = '/home/lsd/Documents/HD189733/August2007/'

    
    ccf_directory = directory.replace('/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/', '/home/lsd/Documents/Starbase/novaprime/Documents/HD189733/')
    print('%s*/*/*/*ccf**K5_A*.fits'%ccf_directory)

    filelist=glob.glob('%s/*/*/*/*ccf**K5_A*.fits'%ccf_directory)
    # filelist=glob.glob('%s/*/*ccf**A*.fits'%ccf_directory)
    print(filelist)
    print(filelist_final)

    inp = input('Enter to continue...')
    return filelist_final, filelist                          #returns list of uncorrected spectra files

def classify(phase):
    #working out if in- or out- of transit
    #print(t, b)
    # deltaphi = t/(2*P)
    #print(deltaphi)
    # phase = (((ccf[0].header['ESO DRS BJD'])-T)/P)%1
    #z = np.sqrt((a_rstar*(np.sin(2*math.pi*phi))**2)+b**2*(np.cos(2*math.pi*phi))**2)
    #part1=a_rstar*np.sin(2*math.pi*phi)**2
    #part2=b**2*np.cos(2*math.pi*phi)**2
    #z=np.sqrt(part1+part2)
    #print(z, phi)
    phi = phase-np.round(phase)

    P=2.21857567 #Cegla et al, 2016 - days
    T=2454279.436714 #cegla et al,2016
    a_Rs = 8.786 #Cristo et al - 8.786
    b=0.687 #Cristo et al, 2022
    RpRs = 0.15667
    RpRs_max = RpRs

    z = np.sqrt( ( (a_Rs)*np.sin(2 * np.pi * phi) )**2 + ( b*np.cos(2. * np.pi * phi))**2)

    if z<=(1+RpRs_max):
        result = 'in'
    else: result = 'out'

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
    f2 = interp1d(adjusted_velocities, spectrum, kind='linear', bounds_error=False, fill_value='extrapolate')
    velocity_grid = np.linspace(-10,10,len(spectrum))
    adjusted_spectrum = f2(velocity_grid)
    
    return velocity_grid, adjusted_spectrum, errors


def remove_berv(velocities, spectrum, berv):
    adjusted_velocities = velocities+berv
    f2 = interp1d(adjusted_velocities, spectrum, kind='linear', bounds_error=False, fill_value='extrapolate')
    adjusted_spectrum = f2(velocities)
    return adjusted_spectrum

def combineccfs(spectra):
    spectra = np.array(spectra)
    length, width = np.shape(spectra)
    spectrum = np.zeros((1,width))
    for n in range(0,width):
        spectrum[0,n]=sum(spectra[:,n])/length
    spectrum = list(np.reshape(spectrum, (width,)))
    return  spectrum

def combineprofiles(spectra, errors, ccf, master, velocities):
    spectra = np.array(spectra)
    idx = np.isnan(spectra)
    print(len(spectra[idx]))
    shape_og = spectra.shape
    if len(spectra[idx])>0:
        print(spectra)
        inp = input('Nans were found')
        spectra = spectra.reshape((len(spectra)*len(spectra[0]), ))
        print(len(spectra))
        print(spectra.shape)
        for n in range(len(spectra)):
            if spectra[n] == np.nan:
                spectra[n] = (spectra[n+1]+spectra[n-1])/2
                if spectra[n] == np.nan:
                    spectra[n] = 0.
    spectra = spectra.reshape(shape_og)
    errors = np.array(errors)

    if master == 'no':

        weights_csv = np.genfromtxt('/home/lsd/Documents/Starbase/novaprime/Documents/order_weights.csv', delimiter=',')
        orders = np.array(weights_csv[7:,0], dtype = int)
        # print(orders)
        weights = np.array(weights_csv[7:,1])
        # print(weights)

        '''
        #calc_errors=np.zeros(np.shape(errors))
        plt.figure('order_vs_cont')
        plt.title('Order vs Continuum Level (CCF profiles), frame: %s'%frame)
        plt.xlabel('Order')
        plt.ylabel('Continuum Level')

        plt.figure('order_vs_line_depth')
        plt.title('Order vs Line Depth, frame: %s'%frame)
        plt.xlabel('Order')
        plt.ylabel('Line Depth')

        plt.figure('cont_vs_line_depth')
        plt.title('Continuum Level vs Line Depth (CCF profiles), frame: %s'%frame)
        plt.ylabel('Line Depth')
        plt.xlabel('Continuum Level')

        for i in range(9,len(spectra)+1):
            #if np.max(abs(spectra[i]))<0.25:
            #continue
            counter = i-1
            points = spectra[counter]
            ccf_points = ccf[0].data[counter]
            continuum_points_ccf = np.concatenate((ccf_points[:5], ccf_points[-5:]))
            continuum_level_ccf = np.mean(continuum_points_ccf)
            ccf_points=ccf_points/continuum_level_ccf -1

            ## work out conitnuum level
            continuum_points = np.concatenate((points[:5], points[-5:]))
            x_ticks = np.arange(0, len(points))

            continuum_level = np.mean(continuum_points)

            continuum_points_ccf = np.concatenate((ccf_points[:5], ccf_points[-5:]))
            x_ticks_ccf = np.arange(0, len(ccf_points))

            continuum_level_ccf = np.mean(continuum_points_ccf)
            plt.figure('order_vs_cont')
            plt.scatter(orders[i-9], continuum_level)
            #plt.scatter(orders[i-9], continuum_level_ccf, color = 'k')

            ## work out line depth
            bottom_points = points.copy()
            idx = bottom_points.argsort()
            bottom_points = bottom_points[idx]
            bottom_points = bottom_points[:3]
            line_depth = abs(continuum_level-np.mean(bottom_points))

            bottom_points_ccf = ccf_points.copy()
            idx = bottom_points_ccf.argsort()
            bottom_points_ccf = bottom_points_ccf[idx]
            bottom_points_ccf = bottom_points_ccf[:3]
            line_depth_ccf = abs(continuum_level_ccf-np.mean(bottom_points_ccf))

            plt.figure('order_vs_line_depth')
            plt.scatter(orders[i-9], line_depth, color = 'r', label = 'LSD')
            plt.scatter(orders[i-9], line_depth_ccf, color = 'k', label = 'CCF')

            plt.legend(['LSD', 'CCF'])

            plt.figure('cont_vs_line_depth')
            plt.scatter(continuum_level, line_depth, color = 'r')
            #plt.scatter(continuum_level_ccf, line_depth_ccf, color = 'k')

        plt.show()
        '''
        spectra_to_combine = []
        errorss = []

        # plt.figure()
        # plt.title('All orders, frame: %s'%frame)
        for i in orders:
            plt.plot(velocities, [0]*len(velocities))
            # print(i)
            # print(np.shape(spectra))
            spectra_to_combine.append(list(spectra[i-1]))
            errorss.append(list(errors[i-1]))
            # plt.plot(velocities, spectra[i-1])
            # plt.fill_between(velocities, spectra[i-1]-errors[i-1], spectra[i-1]+errors[i-1], alpha = 0.3, label = 'order: %s'%i)
            # plt.legend(ncol = 3)
            # #plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/orderserr%s_profile_%s'%(i, run_name))
            # plt.ylim(-0.8, 0.2)
        #plt.show()
        # plt.close()

    else:
        spectra_to_combine = []
        weights=[]
        #all_weights = np.zeros(np.shape(errors))
        for n in range(0, len(spectra)):
            spectra_to_combine.append(list(spectra[n]))
            temp_err = np.array(errors[n, :])
            #temp_err = temp_err/np.max(abs(spectra[n]))
            #errors[n, :]=temp_err
            print(temp_err)
            weight = (1/temp_err**2)
            print(weight)
            weights.append(np.mean(weight))
        weights = np.array(weights/sum(weights))

        # plt.figure()
        # plt.title('master out - all profiles')
        # for i in range(len(spectra)):
        #     #spectra_to_combine.append(list(spectra[i-1]))
        #     #errorss.append(list(errors[i-1]))
        #     plt.plot(velocities, spectra[i])
        #     plt.fill_between(velocities, spectra[i]-errors[i], spectra[i]+errors[i], alpha = 0.3, label = 'frame: %s'%i)
        #     plt.legend(ncol = 3)
        #     #plt.savefig('/home/lsd/Documents/LSD_Figures/profiles/orderserr%s_profile_%s'%(i, run_name))
        #     #plt.show()
        #     plt.close()

    spectra_to_combine = np.array(spectra_to_combine)
    #errorss = np.array(errorss)

    length, width = np.shape(spectra_to_combine)
    spectrum = np.zeros((1,width))
    spec_errors = np.zeros((1,width))

    all_weights = np.zeros(np.shape(errors))
    for n in range(0,width):
        temp_spec = spectra_to_combine[:, n]
        #temp_err = np.array(errorss[:, n])
        #weights = (1/temp_err**2)
        #weights = np.array(weights/sum(weights))
        #all_weights[:,n]=weights
        # print(spectra_to_combine)
        # print(temp_spec)
        # print(weights)
        spectrum[0,n]=sum(weights*temp_spec)/sum(weights)
        #spectrum[0,n]=sum(temp_spec)/len(temp_spec)
        spec_errors[0,n]=(stdev(temp_spec)**2)*np.sqrt(sum(weights**2))
        #spec_errors[0, n] = stdev(temp_spec)
        #print(temp_spec)
        #print(temp_err)
    '''
    plt.figure()
    plt.scatter(orders, weights)
    plt.xlabel('order')
    plt.ylabel('weight')
    '''
    '''
    plt.figure()
    plt.scatter(orders, temp_err)
    plt.xlabel('order')
    plt.ylabel('error')
    '''
    '''
    plt.figure('frame: %s'%frame)
    plt.title('frame: %s'%frame)
    plt.plot(velocities, spectrum[0, :])
    plt.plot(velocities, [0]*len(velocities))
    plt.fill_between(velocities, spectrum[0, :]-spec_errors[0, :], spectrum[0, :]+spec_errors[0, :], alpha = 0.3)
    print(spectrum[0,:])
    plt.show()
    '''

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


path = '/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/'
save_path = '/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/'

#path = '/Users/lucydolan/Starbase/LSD_Figures/'
#save_path = '/Users/lucydolan/Starbase/LSD_Figures/'

month = 'August2007' #August, July, Sep

months = [#'August2007',
          'July2007',
          #'July2006',
          #'Sep2006'
          ]
#linelist = '/Users/lucydolan/Documents/Least_Squares_Deconvolution/LSD/Archive_stuff/archive/fulllinelist018.txt'
# s1d or e2ds
file_type = 'e2ds'

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

v_obs = []
velos=[]
outliers = []
lengths = []

all_weights_total = []

all_profiles = []
all_profile_errors = []
all_ccf_profiles = []
ccf_phases = []
all_results = []
ccf_results = []

month_profiles = []
month_ccfs = []
all_phases = []
header_rvs = []
for month in months:
    plt.figure(month)
    directory =  '%s%s'%(path,month)

    filelist, ccf_list = findfiles(directory, file_type)

    out_ccfs = []
    out_errors = []
    all_ccfs = []
    #phases = []
    results = []
    phases = []
    bjds = []
    mjds = []
    berv = []
    phases1 = []
    velos1=[]
    #results = []
    #befores = []
    #afters = []
    matched=[]
    rv_drift = []
    lengths.append(len(filelist))
    framelist = np.arange(0, len(filelist))
    # framelist = framelist[:2]
    #framelist = framelist[framelist!=4]
    print(framelist)
    #plt.figure('all_frames')
    all_order_rvs = []
    all_order_rvs_ccf = []
    all_rvs = np.zeros((71, len(framelist)))
    counter = -1
    for frame in framelist:
        counter +=1
        file = fits.open(filelist[frame])
        print(file)
        ccf_file = ccf_list[frame]
        print(ccf_file)
        ccf = fits.open(ccf_file)
        order_errors = []
        order_profiles = []
        ccf_profiles = []

        order_rvs = []
        order_fwhm = []
        order_rvs_ccf = []
        order_fwhm_ccf = []
        # plt.figure('all orders %s'%frame)
        # plt.xlabel('Velocity (km/s)')
        # plt.ylabel('Normalised Flux')
        # plt.title('ACID Profiles (All Orders)')
        for order1 in range(1,71):

            profile_errors = file[order1].data[1]
            profile = file[order1].data[0]
            if len(profile) == 48:
                velocities = np.linspace(-21, 18, 48)
            else:velocities=np.arange(-16, 11, 0.82)
            ccf_profile = ccf[0].data[order1]
            if order1 ==1:
                header_rvs = list(header_rvs)
                header_rvs.append(ccf[0].header['HIERARCH ESO DRS CCF RV'])#+ccf[0].header['ESO DRS BERV'])
            velocities_ccf=ccf[0].header['CRVAL1']+(np.arange(ccf_profile.shape[0]))*ccf[0].header['CDELT1']
            # if np.sum(abs(profile))>0:
            #     plt.plot(velocities, profile)
            ccf_phi = (((ccf[0].header['ESO DRS BJD'])-T)/P)%1
            print(berv)
            mjd = ccf[0].header["MJD-OBS"]
            if ccf_phi>0.5: ccf_phi = ccf_phi-1
            order = file[order1].header['ORDER']
            phase = file[order1].header['PHASE']
            # print(phase)
            result = file[order1].header['result']

            ## investigation section - delete/comment out when done
            # velocities_ccf_temp, spectrum_ccf_temp, ccf_errors_temp = remove_reflex(velocities_ccf, ccf_profile/np.mean(ccf_profile[:5])-1, ccf_profile/100, ccf_phi, K, e, omega, v0)
            # velocities_temp, spectrum_temp, errors_temp = remove_reflex(velocities, profile, profile_errors, phase, K, e, omega, v0)
            # st = 15 
            # end =-15
            # try: 
            #     popt, pcov = curve_fit(gauss, velocities[st:end], profile[st:end])
            #     perr= np.sqrt(np.diag(pcov))
            #     all_rvs[order1, counter] = popt[0]
            #     order_fwhm.append(popt[1])
            # except:
            #     all_rvs[order1, counter] = popt[0]
            #     order_fwhm.append(1.)

            # st = 30
            # end =-30
            # try:
            #     popt, pcov = curve_fit(gauss, velocities_ccf[st:end], spectrum_ccf[st:end])
            #     perr= np.sqrt(np.diag(pcov))
            #     order_rvs_ccf.append(popt[0])
            #     order_fwhm_ccf.append(popt[1])
            # except: 
            #     order_rvs_ccf.append(1.)
            #     order_fwhm_ccf.append(1.)

            ## end of section
            '''
            plt.figure('%s'%order)
            plt.plot(velocities, profile)
            plt.plot(velocities_ccf, ccf_profile/ccf_profile[0]-1)
            plt.show()
            '''
            '''
            profile = np.exp(profile)
            profile_errors = np.sqrt(profile_errors**2/profile**2)
            profile = profile-1

            profile[np.isnan(profile)]=0
            profile_errors[np.isnan(profile)] = 1
            profile_errors[np.isnan(profile_errors)]=1
            '''
            #profile_errors = abs(profile_errors/profile)
            #profile_errors[np.isnan(profile_errors)]=0.5
            #print(profile_errors)
            #print('profile errors^^')
            #profile = np.exp(profile)-1
            print(profile)
            #velocities = file[order].data[2]
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
            print(np.mean(ccf_profile[:5]))
            print(ccf_profile/np.mean(ccf_profile[:5])-1)
            ccf_profile = ccf_profile/np.mean(ccf_profile[:5])-1
            for n in range(len(ccf_profile)):
                if ccf_profile[n] == 'nan':
                    if n==0 or n==len(ccf_profile): ccf_profile[n]=0.
                    else:
                        ccf_profile[n] = np.mean([ccf_profile[n-1], ccf_profile[n+1]])
                        print('nan found')
                        print(ccf_profile[n])
                        inp = input('Enter to continue...')

            ccf_profiles.append(ccf_profile)
            
            plt.ylim(-0.75, 0.15)
            # plt.savefig('all_orders/ACIDprof_orders%s'%frame)
            
            # print(order_profiles)
            # print(ccf_profiles)

        # print(len(ccf_profiles))
        # print(len(order_profiles))
        berv.append(ccf[0].header['ESO DRS BERV'])
        mjds.append(mjd)
        all_order_rvs.append(order_rvs)
        all_order_rvs_ccf.append(order_rvs_ccf)

        if frame == framelist[0]:
            plt.figure('LSD')
            plt.imshow(np.array(order_profiles), extent = [velocities[0], velocities[-1], 0, len(order_profiles)-1])
            plt.vlines(-2.276, 0, len(order_profiles)-1)
            plt.colorbar()

            plt.figure()
            for prof in order_profiles:
                plt.plot(velocities, prof)

            plt.figure('CCFs')
            plt.imshow(np.array(ccf_profiles), extent = [velocities_ccf[0], velocities_ccf[-1], 0, len(order_profiles)-1])
            plt.vlines(-2.276, 0, len(order_profiles)-1)
            plt.colorbar()
            plt.show()

        print(phase, result)
        result, phi, phase = classify(phase) #phi is adjusted, phase is original
        print(phase, result)
        # inp = input('Enter to continue...')
        if phase>0.5:
            phi = phase-1
        else:phi = phase

        #print(result, phi, phase)
        #plt.figure(file)
        #spectrum, errors = combineprofiles(order_profiles, order_errors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 35, 38, 39, 41, 42, 43, 44, 45, 45, 47, 50, 51, 55, 56, 63, 64, 65, 69])
        ccf_profiles = np.array(ccf_profiles)

        idx = np.isnan(order_profiles)
        order_profiles = np.array(order_profiles)
        order_profiles[idx] = 0.

        spectrum, errors, weights = combineprofiles(order_profiles, order_errors, ccf, 'no', velocities)
        print(abs(np.max(spectrum)-np.min(spectrum)))
        if abs(np.max(spectrum)-np.min(spectrum))<0.1:
            spectrum = order_profiles[27]
            errors = order_errors[27]
        # plt.figure()
        # plt.plot(spectrum)
        # plt.show()
        
        spectrum_ccf = ccf[0].data[72]
        velocities_ccf=ccf[0].header['CRVAL1']+(np.arange(ccf_profile.shape[0]))*ccf[0].header['CDELT1']
        # spectrum_ccf, errors_ccf, weights_ccf = combineprofiles(ccf_profiles, np.ones(ccf_profiles.shape)*0.0001, ccf, 'no', velocities_ccf)

        all_weights_total.append(weights)
        #plt.plot(velocities, spectrum)
        # velocities_ccf, spectrum_ccf, ccf_errors = remove_reflex(velocities_ccf, spectrum_ccf, spectrum_ccf/100, ccf_phi, K, e, omega, v0)
        velocities, spectrum, errors = remove_reflex(velocities, spectrum, errors, phi,K, e, omega, v0)

        all_profiles = list(all_profiles)
        all_profile_errors = list(all_profile_errors)
        all_ccf_profiles = list(all_ccf_profiles)
        ccf_phases = list(ccf_phases)
        all_phases = list(all_phases)
        all_profiles.append(spectrum)
        all_profile_errors.append(errors)
        all_ccf_profiles.append(spectrum_ccf/np.mean(spectrum_ccf[:5])-1)
        all_phases.append(phi)
        ccf_phases.append(ccf_phi)
        all_results.append(result)

        fig = plt.figure('all frames')
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Normalised Flux')
        plt.title('ACID Profiles')
        plt.plot(velocities, spectrum)
        plt.figure('not all frames')
        # plt.plot(velocities, [0]*len(spectrum))
        # print(file[0].header['ESO DRS BJD'])
        # print(ccf[0].header['ESO DRS BJD'])
        bjds.append(ccf[0].header['ESO DRS BJD'])
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
        all_ccfs.append([list(spectrum), list(errors)])
        total.append(spectrum)
        #phases.append(phi)
        results.append(result)
        if result == 'out':
            #plt.plot(spectrum)
            out_ccfs.append(spectrum)
            out_errors.append(errors)


    # plt.show()
    #velos.append(velos1)
    #break

   
    # month_profiles.append([velocities, spectrum])
    # month_ccfs.append([velocities_ccf, spectrum_ccf])
    phases = np.array(phases1)
    all_ccfs = np.array(all_ccfs)
    results = np.array(results)
    berv = np.array(berv)

    idx = phases.argsort()
    phases = phases[idx]
    results = results[idx]
    all_ccfs = all_ccfs[idx]
   
    phases = list(phases)
    results = list(results)

    all_phases = np.array(all_phases)
    all_profiles = np.array(all_profiles)
    all_results = np.array(all_results)
    idx = all_phases.argsort()
    all_phases = all_phases[idx]
    all_profiles = all_profiles[idx]
    all_phases = list(all_phases)
    all_results = all_results[idx]

    ccf_phases = np.array(ccf_phases)
    header_rvs = np.array(header_rvs)
    idc = ccf_phases.argsort()
    ccf_phases = ccf_phases[idc]
    all_ccf_profiles = np.array(all_ccf_profiles)
    all_ccf_profiles = all_ccf_profiles[idc]
    header_rvs = header_rvs[idc]
    berv = berv[idc]
    berv = list(berv)

    print(ccf_phases)
    print(phases)
    inp = input('Above should be the same')

    # for i in range(len(all_ccfs)):
    #     spectrum_ccf = all_ccf_profiles[i]
    #     spectrum = all_ccfs[i, 0]
    #     ## adjusting rv of ACID profile to match that of CCF profile
    #     popt_ccf, pcov_ccf = curve_fit(gauss, velocities_ccf[15:-15], spectrum_ccf[15:-15])
    #     popt_lsd, pcov_lsd = curve_fit(gauss, velocities[15:-15], spectrum[15:-15]+1)
    #     rv_diff = popt_lsd[0]-popt_ccf[0]
    #     velocities_ad = velocities - rv_diff
    #     f2 = interp1d(velocities_ad, spectrum, kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
    #     spectrum = f2(velocities)

    #     all_ccfs[i, 0] = spectrum
    #     all_profiles[i] = spectrum


    ### REMOVING THE BERV 
    # for i in range(len(all_profiles)):
    #     print(ccf_phases[i], phases[i], berv[i])
    #     profile3 = remove_berv(velocities, all_profiles[i], berv[i])
    #     all_profiles[i] = profile3
    #     all_ccfs[i][0] = profile3

    #making master out

    idx_out = tuple([all_results=='out'])
    out_ccfs= all_ccfs[idx_out]
    out_errors = out_ccfs[:, 1]
    out_ccfs = out_ccfs[:, 0]
    frame = 'master_out'
    count = 0
    if len(out_ccfs)>1:
        plt.figure('all ccfs')
        for ccf in all_ccf_profiles:
            plt.plot(velocities_ccf, ccf, label = 'Frame %s'%count)
            count+=1
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Normalised Flux')
        plt.title('CCF Profiles')
        plt.savefig('CCFprofs.png')
        plt.show()
        master_out_spec, master_out_errors, master_weights = combineprofiles(out_ccfs, out_errors ,ccf, 'yes', velocities)
    else:
        master_out_spec = out_ccfs[0]
        master_out_errors = out_errors[0]

    master_out = np.array([master_out_spec, master_out_errors])


    #write in data
    hdu=fits.HDUList()
    for data in all_ccfs:
        hdu.append(fits.PrimaryHDU(data=data))

    hdu.append(fits.PrimaryHDU(data=master_out))
    phases.append('out')
    results.append('master_out')
    bjds.append('out')
    mjds.append('out')
    berv.append('out')

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
        hdr['bjd']=bjds[p]
        hdr['mjd']=mjds[p]
        hdr['berv']=berv[p]
        hdr['RESULT']=results[p]
        hdu[p].header=hdr
    
    print(velocities)

    hdu.writeto('%s%s_master_out_LSD_profile.fits'%(save_path, month), output_verify='fix', overwrite = 'True')

    # opened_file.close()

# plt.figure()
# all_profs = []
# Aug_LSD = month_profiles[0]
# Aug_ccf = month_ccfs[0]
# Jul_LSD = month_profiles[1]
# Jul_ccf = month_ccfs[1]
# plt.plot(Aug_LSD[0], Aug_LSD[1], label = 'August 2007 - LSD')
# plt.plot(Jul_LSD[0], Jul_LSD[1], label = 'July 2007 - LSD')
# aug = Aug_ccf[1]
# jul = Jul_ccf[1]
# plt.plot(Aug_ccf[0], Aug_ccf[1]/aug[0]-1, label = 'August 2007 - ccf')
# plt.plot(Jul_ccf[0], Jul_ccf[1]/jul[0]-1, label = 'July 2007 - ccf')
# plt.legend()
# plt.show()

# count = 0
# rv_phases = []
# rvs = []
# plt.figure()
# for y in month_profiles:
#    # if (-t/(2*P)+0.001)<phases[count]<0.0155:
#     y = y[count]
#     popt, pcov = curve_fit(gauss, y[0], y[1])
#     perr= np.sqrt(np.diag(pcov))
#     rvs.append([popt[0], perr[0]])
#     plt.plot('LSD %s'%months[count], popt[0])
#     #rv_phases.append(phase[count])
#     count += 1
# count = 0
# for y in month_ccfs:
#    # if (-t/(2*P)+0.001)<phases[count]<0.0155:
#     popt, pcov = curve_fit(gauss, y[0], y[1])
#     perr= np.sqrt(np.diag(pcov))
#     rvs.append([popt[0], perr[0]])
#     plt.plot('CCF %s'%months[count], popt[0])
#     #rv_phases.append(phase[count])
#     count += 1

# rvs = []
# plt.figure()

# popt, pcov = curve_fit(gauss, Aug_LSD[0], Aug_LSD[1])
# perr= np.sqrt(np.diag(pcov))
# plt.scatter('LSD - Aug 07', popt[0])
# rvs.append([popt[0], perr[0]])

# popt, pcov = curve_fit(gauss, Aug_ccf[0], Aug_ccf[1]/aug[0]-1)
# perr= np.sqrt(np.diag(pcov))
# plt.scatter('CCF - Aug 07', popt[0])
# rvs.append([popt[0], perr[0]])

# popt, pcov = curve_fit(gauss, Jul_LSD[0], Jul_LSD[1])
# perr= np.sqrt(np.diag(pcov))
# plt.scatter('LSD - Jul 07', popt[0])
# rvs.append([popt[0], perr[0]])

# popt, pcov = curve_fit(gauss, Jul_ccf[0], Jul_ccf[1]/jul[0]-1)
# perr= np.sqrt(np.diag(pcov))
# plt.scatter('CCF - Jul 07', popt[0])
# rvs.append([popt[0], perr[0]])

# plt.show()
count = 0
rv_phases = []
rv_results = []
ccf_rv_results = []
rvs = []
fwhm = []
plt.figure()
# st = 15
# end = -15
st = 0
end = len(velocities)
# cmap = plt.colormaps('Blues')'
colour = cm.Blues(np.linspace(0, 1, len(all_profiles)))
plt.figure()
for y in all_profiles:

    popt, pcov = curve_fit(gauss, velocities[st:end], y[st:end])
    perr= np.sqrt(np.diag(pcov))
    plt.plot(velocities[st:end], y[st:end], 'k')
    plt.plot(velocities[st:end], gauss(velocities[st:end], popt[0], popt[1], popt[2], popt[3]), 'r')
    rvs.append([popt[0], perr[0]])
    fwhm.append(2.355*popt[1])
    rv_phases.append(all_phases[count])
    rv_results.append(all_results[count])
    count += 1
#plt.legend()
plt.show()

rv_results = np.array(rv_results)

count = 0
rv_phases_ccf = []
ccf_rvs = []
ccf_fwhm = []
plt.figure()
st = 0
end = len(velocities_ccf)
st = 15
end = -15
for y in all_ccf_profiles:
    # if (-t/(2*P)+0.001)<phases[count]<0.0155:
    # popt1, pcov1 = curve_fit(skewnormal, velocities_ccf, y)
    # perr1= np.sqrt(np.diag(pcov1))
    # beta1=np.sqrt(2/np.pi)*(popt1[3]/(np.sqrt(1+popt1[3]**2)))
    # profilemean1=popt1[2]+popt1[1]*beta1
    # beta_error1=np.sqrt(2/np.pi)*(perr1[3]/(np.sqrt(1+perr1[3]**2)))
    # profilemean_error1=perr1[2]+perr1[1]*beta_error1
    # ccf_rvs.append([profilemean1, profilemean_error1])
    # rv_phases_ccf.append(all_phases[count])

    popt, pcov = curve_fit(gauss, velocities_ccf[st:end], y[st:end]+1)
    plt.plot(velocities_ccf[st:end], y[st:end]+1, 'k')
    plt.plot(velocities_ccf[st:end], gauss(velocities_ccf[st:end], popt[0], popt[1], popt[2], popt[3]), 'r')
    perr= np.sqrt(np.diag(pcov))
    ccf_rvs.append([popt[0], perr[0]])
    ccf_fwhm.append(2.355*popt[1])
    rv_phases_ccf.append(ccf_phases[count])
    count += 1
plt.show()

rvs = np.array(rvs)
rvs = rvs[:, 0]
ccf_rvs = np.array(ccf_rvs)
ccf_rvs = ccf_rvs[:, 0]
fwhm = np.array(fwhm)
ccf_fwhm = np.array(ccf_fwhm)
rv_phases = np.array(rv_phases)
rv_phases_ccf = np.array(rv_phases_ccf)

idx_in = tuple([rv_results == 'in'])
idx_out = tuple([rv_results == 'out'])

plt.figure('ACID and CCF Delta plot', figsize = [9, 7])
#plt.title('ACID and CCF FWHM')
plt.ylabel('ACID - CCF FWHM')
plt.xlabel('ACID - CCF RVs')
plt.scatter(rvs[idx_out]-ccf_rvs[idx_out], fwhm[idx_out]-ccf_fwhm[idx_out], label = 'out', color = 'g', alpha = 0.75)
plt.scatter(rvs[idx_in]-ccf_rvs[idx_in], fwhm[idx_in]-ccf_fwhm[idx_in], label = 'in', color = 'b', alpha = 0.75)
plt.legend()
plt.savefig('Sep_rr_DELTAplot.png')

plt.figure('ACID and CCF (-median)', figsize = [9, 7])
#plt.title('ACID and CCF FWHM')
plt.ylabel('FWHM - median(FWHM)')
plt.xlabel('RV - median(RV)')
plt.scatter(ccf_rvs[idx_out]-np.median(ccf_rvs), ccf_fwhm[idx_out]-np.median(ccf_fwhm), label = 'CCF out', color = 'c', alpha = 1)
plt.scatter(ccf_rvs[idx_in]-np.median(ccf_rvs), ccf_fwhm[idx_in]-np.median(ccf_fwhm), label = 'CCF in', color = 'c', alpha = 0.25)
plt.scatter(rvs[idx_out]-np.median(rvs), fwhm[idx_out]-np.median(fwhm), label = 'ACID out', color = 'm')
plt.scatter(rvs[idx_in]-np.median(rvs), fwhm[idx_in]-np.median(fwhm), label = 'ACID in', color = 'm', alpha = 0.25)
plt.legend()
plt.savefig('Sep_rr_Medianplot.png')

plt.figure('ACID and CCF FWHM (-median)', figsize = [9, 7])
#plt.title('ACID and CCF FWHM')
plt.ylabel('FWHM - median(FWHM)')
plt.xlabel('Phase')
plt.scatter(rv_phases[idx_out], ccf_fwhm[idx_out]-np.median(ccf_fwhm), label = 'CCF out', color = 'c', alpha = 1)
plt.scatter(rv_phases[idx_in], ccf_fwhm[idx_in]-np.median(ccf_fwhm), label = 'CCF in', color = 'c', alpha = 0.25)
plt.scatter(rv_phases[idx_out], fwhm[idx_out]-np.median(fwhm), label = 'ACID out', color = 'm')
plt.scatter(rv_phases[idx_in], fwhm[idx_in]-np.median(fwhm), label = 'ACID in', color = 'm', alpha = 0.25)
plt.legend()
plt.savefig('Sep_rr_FWHMcurve.png')

plt.figure('ACID and CCF RV (-median)', figsize = [9, 7])
plt.ylabel('RV - median(RV)')
plt.xlabel('Phase')
plt.scatter(rv_phases[idx_out], ccf_rvs[idx_out]-np.median(ccf_rvs), label = 'CCF out', color = 'c', alpha = 1)
plt.scatter(rv_phases[idx_in], ccf_rvs[idx_in]-np.median(ccf_rvs), label = 'CCF in', color = 'c', alpha = 0.25)
plt.scatter(rv_phases[idx_out], rvs[idx_out]-np.median(rvs), label = 'ACID out', color = 'm')
plt.scatter(rv_phases[idx_in], rvs[idx_in]-np.median(rvs), label = 'ACID in', color = 'm', alpha = 0.25)
plt.legend()
plt.savefig('Jul_rr_RVcurve.png')

plt.figure('ACID and CCF RV (-median)', figsize = [9, 7])
plt.ylabel('RV - median(RV)')
plt.xlabel('Phase')
plt.scatter(rv_phases, ccf_rvs-np.median(ccf_rvs), label = 'CCF', color = 'c', alpha = 1)
plt.scatter(rv_phases, rvs-np.median(rvs), label = 'ACID NEW', color = 'm')
# plt.scatter(rv_phases, rvs2-np.median(rvs2), label = 'ACID NEW_moremask', color = 'k')
plt.legend()
plt.savefig('Jul_rr_RVcurve.png')

# plt.scatter(rv_phases[idx_in], rvs[idx_in]-ccf_rvs[idx_in], label = 'in', color = 'b', alpha = 0.75)
# plt.scatter(rv_phases[idx_out], rvs[idx_out]-ccf_rvs[idx_out], label = 'out', color = 'g', alpha = 0.75)

# plt.scatter(ccf_rvs[idx_in]-np.median(ccf_rvs), ccf_fwhm[idx_in]-np.median(ccf_fwhm), label = 'CCF in', color = 'b', alpha = 0.5)
# plt.scatter(rvs[idx_in]-np.median(rvs), fwhm[idx_in]-np.median(fwhm), label = 'ACID in', color = 'r')
# plt.scatter(ccf_rvs[idx_out]-np.median(ccf_rvs), ccf_fwhm[idx_out]-np.median(ccf_fwhm), label = 'CCF out', color = 'c', alpha = 0.5)
# plt.scatter(rvs[idx_out]-np.median(rvs), fwhm[idx_out]-np.median(fwhm), label = 'ACID out', color = 'm')

# plt.scatter(rv_phases[idx_in], ccf_fwhm[idx_in]-np.median(ccf_fwhm), label = 'CCF in', color = 'b', alpha = 0.5)
# plt.scatter(rv_phases[idx_in], fwhm[idx_in]-np.median(fwhm), label = 'ACID in', color = 'r')
# plt.scatter(rv_phases[idx_out], ccf_fwhm[idx_out]-np.median(ccf_fwhm), label = 'CCF out', color = 'c', alpha = 0.5)
# plt.scatter(rv_phases[idx_out], fwhm[idx_out]-np.median(fwhm), label = 'ACID out', color = 'm')

# #plt.close('all')
# '''
#     ins = ins+(len(all_ccfs)-len(out_ccfs))

#     min_points = []
#     #alll_ccfs=[]
#     for c in range(len(all_ccfs)):
#         ccf = all_ccfs[c]
#         plt.plot(velocities, ccf)
#         for vel in range(len(velocities)):
#             if ccf[vel]==np.min(ccf):
#                 min_points.append(velocities[vel])#, results[c]])
#                 if round(velocities[vel], 2)!=0:outliers.append([phases1[c], velos[c]])

#     plt.show()

#     print(min_points)
# '''
# '''
# #plots phases against velocity shift
# plt.figure('phases')

# colours = ['b', 'g', 'k', 'r', 'm']
# #colours =  ['k', 'r', 'm']
# n2=0

# for n1 in range(0,len(lengths)):
#     n3=n2
#     #print(n3)
#     for n2 in range(0+n3, n3+lengths[n1]):
#         #print(n1, n2)
#         #plt.scatter(phases1[n2], velos[n2], marker ='.')
#         plt.scatter(phases1[n2], v_obs[n2], marker='x', color = colours[n1])

# plt.scatter(phases1, velos, marker ='.')
# #plt.scatter(phases1, v_obs, marker='x')

# velos_m=[]
# phases_m = np.arange(-0.1,1.1,0.001)
# for phi in phases_m:
#     velo = v0 + K*(e*np.cos(omega)+np.cos(2*np.pi*phi-omega))
#     velos_m.append(velo)

# plt.plot(phases_m, velos_m)


# for out in outliers:
#     phase, velo = out
#     plt.scatter(phase-np.round(phase), velo)

# plt.xlim(-0.05, 0.05)
# plt.ylim(-2.35, -2.1)
# plt.show()

# plt.figure('difference')

# n2=0
# for n1 in range(0,len(lengths)):
#     n3=n2
#     #print(n3)
#     for n2 in range(0+n3, n3+lengths[n1]):
#         #print(n1, n2)
#         #plt.scatter(phases1[n2], velos[n2], marker ='.')
#         plt.plot(phases1[n2], v_obs[n2]-velos[n2], '.', color = colours[n1])

# plt.ylim(-0.1,0.1)
# plt.xlim(-0.05,0.05)
# '''
# '''
# n1 = len(total)

# col0=((np.round(np.linspace(0,170,num=n1))).astype(int))
# #col1=((np.round(np.linspace(0,170,num=n2))).astype(int))
# #col2=((np.round(np.linspace(50,220,num=n3))).astype(int))

# #plt.cm.Reds(col1[i-n1])
# #plt.cm.Greens(col2[i])
# #plt.cm.Reds(col1[i])
# #plt.cm.Blues(col0[i])

# fig, ax1 = plt.subplots(1,1)

# for i in range(len(total)):
#     ax1.plot(velocities,total[i],color=plt.cm.gist_rainbow(col0[i]),linewidth=0.7)
# plt.show()

# #plt.pcolormesh()
# '''

# print('Completed')
