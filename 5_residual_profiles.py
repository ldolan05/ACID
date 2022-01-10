#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:14:17 2021

@author: lucydolan
"""

import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import math
import glob

#def combineccfs(spectra):
   # spectrum[:] = np.sum(spectra[:])/len(spectra[:])
    #return  spectrum


def residualccfs(in_ccfs, in_ccfs_errors, master_out, master_out_errors, velocities):
    residuals=[]
    residual_errors = []
    #plt.figure('residuals')
    for i in range(len(in_ccfs)):
        ccf = in_ccfs[i]
        ccf_err = in_ccfs_errors[i]
        residual = (master_out+1)-(ccf+1)
        error = (np.sqrt(master_out_errors**2 + ccf_err**2))/np.sqrt(len(ccf_err))
        #residual = (ccf+1)/(master_out+1)-1
        #plt.scatter(velocities,residual)
        residuals.append(residual)
        residual_errors.append(error)
        '''
        plt.figure(1)
        #plt.plot(ccf)
        #plt.plot(master_out)
        plt.plot(residual)
        plt.show()
        '''
        #plt.show()

    return residuals, residual_errors
####################################################################################################################################################################


##path = '/Users/lucydolan/Documents/CCF_method/HD189733_HARPS_CCFS/'
path = '/home/lsd/Documents/LSD_Figures/'
#path = '/Users/lucydolan/Starbase/LSD_Figures/'
month = 'August2007' #August, July, Sep
#path = '%s%s_master_out_LSD_profile.fits'%(save_path, month)

months = ['August2007', #'July2007',
          #'July2006',
          #'Sep2006'
          ]

all_resi=[]
all_phase=[]
results = []

for month in months:

    directory =  '%s'%(path)
    #file ='%s%s_master_out.fits'%(directory, month)
    file ='%s%s_master_out_LSD_profile.fits'%(directory, month)
    ccf_file='%s%s_master_out_ccfs.fits'%(directory, month)
    #file ='%s%s_master_out_LSD_v3.fits'%(directory, month)
    #file ='%s%s_master_out_cegla.fits'%(directory, month)
    all_ccfs = fits.open(ccf_file)
    all_profiles = fits.open(file)

    profile_spec = all_profiles[0].data[0]
    velocities=all_profiles[0].header['CRVAL1']+(np.arange(profile_spec.shape[0]))*all_profiles[0].header['CDELT1']

    ccf_spec = all_ccfs[0].data[0]
    ccf_velocities=all_ccfs[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*all_ccfs[0].header['CDELT1']

    master_position_ccf = len(all_ccfs)-1
    master_out_ccf, master_out_errors_ccf= all_ccfs[master_position_ccf].data

    master_position = len(all_profiles)-1
    master_out, master_out_errors= all_profiles[master_position].data

    plt.figure('master out')
    plt.plot(velocities, master_out, label = 'LSD')
    plt.plot(ccf_velocities, master_out_ccf/master_out_ccf[0]-1, label = 'ccf')
    plt.legend()
    plt.show()

    in_profiles = []
    in_profiles_errors = []
    phases = []

    in_ccfs = []
    in_ccfs_errors = []
    phases_ccfs = []

    plt.figure('all_ccfs')
    for line in range(0,master_position):
        ccf = all_ccfs[line].data[0]
        ccf_errors = all_ccfs[line].data[1]
        ccf_phase = all_ccfs[line].header['PHASE']
        #ccf_result = all_ccfs[line].header['RESULT']
        in_ccfs.append(ccf/ccf[0]-1)
        in_ccfs_errors.append(ccf_errors/ccf[0])
        #plt.plot(ccf, label = '%s_%s'%(result, line))
        phases_ccfs.append(ccf_phase)
        #all_phase.append(phase)
        #results.append(result)

        profile = all_profiles[line].data[0]
        profile_errors = all_profiles[line].data[1]
        phase = all_profiles[line].header['PHASE']
        result = all_profiles[line].header['RESULT']
        in_profiles.append(profile)
        in_profiles_errors.append(profile_errors)
        plt.plot(velocities, profile, label = '%s_%s'%(result, line))
        phases.append(phase)
        all_phase.append(phase)
        results.append(result)
    plt.legend()
    plt.show()

    #print(phases)
    profile_spec = all_profiles[0].data[0]
    #velocities=all_profiles[0].header['CRVAL1']+(np.arange(profile_spec.shape[0]))*all_profiles[0].header['CDELT1']
    velocities = np.linspace(-21, 18, 48)

    ccf_spec = all_ccfs[0].data[0]
    ccf_velocities=all_ccfs[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*all_ccfs[0].header['CDELT1']


   # K = -2.277 #km/s - Boisse et al, 2009
    #velocities = velocities - K  ### Adjusting doppler reflex ###

    residual_profiles, residual_profile_errors = residualccfs(in_profiles, in_profiles_errors, master_out, master_out_errors, velocities)
    residual_ccfs, residual_errors = residualccfs(in_ccfs, in_ccfs_errors, master_out_ccf, master_out_errors_ccf, ccf_velocities)

    '''
    plt.figure(month)
    outs=[]
    ins = []
    for ccf in residual_ccfs:
        if max(ccf)<0.4:
            plt.plot(ccf)
            ins.append(ccf)
        else:outs.append(ccf)
    print(len(outs))
    print(len(ins))
    '''

    print(month)
    plt.figure()
    i=0
    for ccf1 in residual_ccfs:
        plt.plot(ccf_velocities, ccf1, label = '%s'%(i))
        #plt.fill_between(ccf_velocities, ccf1-residual_errors[i], ccf1+residual_errors[i], alpha = 0.3)
        #all_resi.append(ccf1)
        i+=1
    plt.legend()
    plt.show()

    print(month)
    plt.figure(month)
    i=0
    for ccf1 in residual_profiles:
        plt.plot(velocities, ccf1, label = '%s_%s'%(results[i], i))
        plt.fill_between(velocities, ccf1-residual_profile_errors[i], ccf1+residual_profile_errors[i], alpha = 0.3)
        all_resi.append(ccf1)
        i+=1
    plt.legend()
    plt.show()


    #write in data
    hdu=fits.HDUList()
    for data in residual_profiles:
        hdu.append(fits.PrimaryHDU(data=data))

    #hdu.append(fits.PrimaryHDU(data=master_out))

    #write in header
    for p in range(len(phases)):
        phase = phases[p]
        hdr=fits.Header()
        hdr['CRVAL1']=all_profiles[0].header['CRVAL1']
        hdr['CDELT1']=all_profiles[0].header['CDELT1']
        hdr['OBJECT']='HD189733b'
        hdr['NIGHT']='%s'%month
        hdr['K']=-2.277
        hdr['PHASE']=phase
        hdu[p].header=hdr

    #hdu.writeto('%s%s_residual_ccfs.fits'%(directory, month), output_verify='fix', overwrite = 'True')
    #hdu.writeto('%s%s_residual_ccfs_LSD_v3.fits'%(directory, month), output_verify='fix', overwrite = 'True')

n1 = len(all_resi)

col0=((np.round(np.linspace(0,170,num=n1))).astype(int))
#col1=((np.round(np.linspace(0,170,num=n2))).astype(int))
#col2=((np.round(np.linspace(50,220,num=n3))).astype(int))

#plt.cm.Reds(col1[i-n1])
#plt.cm.Greens(col2[i])
#plt.cm.Reds(col1[i])
#plt.cm.Blues(col0[i])
'''
fig, ax1 = plt.subplots(1,1)

for i in range(len(all_resi)):
    ax1.plot(velocities, all_resi[i],color=plt.cm.gist_rainbow(col0[i]),linewidth=0.7)
plt.show()
'''
plt.figure('phase map')
fig, ax0 = plt.subplots(nrows=1)
cmap = plt.get_cmap('jet')
im = ax0.pcolormesh(velocities, all_phase, all_resi, cmap = cmap)
plt.ylim(-0.02, 0.02)
fig.colorbar(im, ax=ax0,label='Residual Flux')
plt.xlabel('Velocity (km/s)')
plt.ylabel('Phase')
#plt.savefig('/Users/lucydolan/Documents/CCF_method/Figures/residual_phasemap_mine')
plt.show()


plt.figure('finalgraph')
for ccf in all_resi:
    plt.plot(velocities, ccf)
plt.xlabel('Velocity (km/s)')
plt.ylabel('Residual Flux')
#plt.savefig('/Users/lucydolan/Documents/CCF_method/Figures/residual_ccfs_mine')
plt.show()
