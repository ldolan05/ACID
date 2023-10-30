#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 14:14:17 2021

@author: lucydolan
"""
import numpy as np
import math
import glob
import Mandel_Agol as ma
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.pyplot import cm

#def combineccfs(spectra):
   # spectrum[:] = np.sum(spectra[:])/len(spectra[:])
    #return  spectrum


def gauss(x, rv, sd, height, cont):
    y = cont*(1-height*np.exp(-(x-rv)**2/(2*sd**2)))
    return y

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
        plt.figure()
        plt.plot(ccf)
        plt.plot(master_out)
        #plt.plot(residual)
        plt.show()
        '''
        #plt.show()

    return residuals, residual_errors
####################################################################################################################################################################


##path = '/Users/lucydolan/Documents/CCF_method/HD189733_HARPS_CCFS/'
path = '/home/lsd/Documents/Starbase/novaprime/Documents/LSD_Figures/'
#path = '/Users/lucydolan/Starbase/LSD_Figures/'
month = 'August2007' #August, July, Sep
#path = '%s%s_master_out_LSD_profile.fits'%(save_path, month)

months = [#'August2007', 
          'July2007'#,
          #'July2006',
          #'Sep2006'
          ]

all_resi=[]
all_phase=[]
all_prof=[]
results_all = []

masters = []
for month in months:
    directory =  '%s'%(path)
    #file ='%s%s_master_out.fits'%(directory, month)
    file ='%s%s_master_out_LSD_profile.fits'%(directory, month)
    #ccf_file='%s%s_master_out_ccfs.fits'%(directory, month)
    #file ='%s%s_master_out_LSD_v3.fits'%(directory, month)
    #file ='%s%s_master_out_cegla.fits'%(directory, month)
    # all_ccfs = fits.open(ccf_file)
    all_profiles = fits.open(file)

    profile_spec = all_profiles[0].data[0]
    velocities = np.linspace(-10,10,len(profile_spec))


    count = 0
    rv_phases = []
    rv_results = []
    ccf_rv_results = []
    rvs = []
    fwhm = []
    plt.figure()
    st = 15
    end = -15
    st = 0
    end = len(velocities)
    # cmap = plt.colormaps('Blues')'

    # colour = cm.Blues(np.linspace(0, 1, len(all_profiles)))
    # plt.figure()
    # for i in range(len(all_profiles)-1):
    #     y = all_profiles[i].data[0]
    #     popt, pcov = curve_fit(gauss, velocities[st:end], y[st:end])
    #     perr= np.sqrt(np.diag(pcov))
    #     plt.plot(velocities[st:end], y[st:end], 'k')
    #     plt.plot(velocities[st:end], gauss(velocities[st:end], popt[0], popt[1], popt[2], popt[3]), 'r')
    #     rvs.append(popt[0])
    #     fwhm.append(2.355*popt[1])
    #     rv_phases.append(all_profiles[count].header['PHASE'])
        
    #     rv_results.append(all_profiles[count].header['RESULT'])
    #     count += 1
    # #plt.legend()

    # plt.figure('ACID and CCF RV (-median)', figsize = [9, 7])
    # plt.ylabel('RV - median(RV)')
    # plt.xlabel('Phase')
    # plt.scatter(rv_phases, rvs-np.median(rvs), label = 'ACID NEW', color = 'm')
    # # plt.scatter(rv_phases, rvs2-np.median(rvs2), label = 'ACID NEW_moremask', color = 'k')
    # plt.legend()
    # plt.show()

    # ccf_spec = all_ccfs[0].data[0]
    # ccf_velocities=all_ccfs[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*all_ccfs[0].header['CDELT1']
    # ccf_velocities = np.linspace(-19,19,len(ccf_spec))

    # master_position_ccf = len(all_ccfs)-1
    # master_out_ccf, master_out_errors_ccf= all_ccfs[master_position_ccf].data

    master_position = len(all_profiles)-1
    master_out, master_out_errors= all_profiles[master_position].data
    masters.append(master_out)

    plt.figure('master out')
    plt.plot(velocities, master_out, label = 'LSD')
    # plt.plot(ccf_velocities, master_out_ccf/master_out_ccf[0]-1, label = 'ccf')
    plt.legend()
    plt.show()

    # master_out_ccf = master_out_ccf/master_out_ccf[0]-1

    in_profiles = []
    in_profiles_errors = []
    phases = []
    results = []
    in_ccfs = []
    in_ccfs_errors = []
    phases_ccfs = []
    ccf_results = []
    out_profiles = []
    out_profile_error = []

    plt.figure('all_ccfs')
    # for line in range(0,master_position_ccf):
    #     ccf = all_ccfs[line].data[0]
    #     ccf_errors = all_ccfs[line].data[1]
    #     ccf_phase = all_ccfs[line].header['PHASE']
    #     in_ccfs.append(ccf/ccf[0]-1)
    #     in_ccfs_errors.append(ccf_errors/ccf[0])
    #     #plt.plot(ccf, label = '%s_%s'%(result, line))
    #     phases_ccfs.append(ccf_phase)
    #     #all_phase.append(phase)
    #     #ccf_results.append(ccf_result)
    for line in range(0, master_position):
        profile = all_profiles[line].data[0]
        profile_errors = all_profiles[line].data[1]
        phase = all_profiles[line].header['PHASE']
        result = all_profiles[line].header['RESULT']
        ##adding in M and A curve
        P=2.21857567 #Cegla et al, 2016 - days
        T=2454279.436714 #cegla et al,2016
        a_Rs = 8.786 #Cristo et al - 8.786
        b=0.687 #Cristo et al, 2022
        RpRs = 0.15667
        u1=0.816
        u2=0  #Sing et al, 2011
        i = 85.5*np.pi/180 #Cristo et al, 2022

        z = ma.MandelandAgol(phase, a_Rs, i)
        transit_curve = ma.occultquad(z, RpRs, [u1, u2])
        print(line)
        print(phase)
        print(transit_curve)
        print(result)

        # profile = (profile[5:-5]+1)*transit_curve
        # profile_errors = profile_errors[5:-5]*transit_curve
        # if line ==0:
        #     velocities = velocities[5:-5]
        # print(velocities.shape, profile.shape)

        profile = (profile+1)*transit_curve
        profile_errors = profile_errors*transit_curve
        
        in_profiles.append(profile)
        in_profiles_errors.append(profile_errors)
        plt.plot(velocities, profile, label = '%s_%s'%(result, line))
        phases.append(phase)
        all_phase.append(phase)
        if result == 'out':
            out_profiles.append(profile)
            out_profile_error.append(profile)
        results.append(result)
        results_all.append(result)
    plt.legend()
    plt.show()

    # plt.figure('ccfs - LSDs')
    # f2 = interp1d(ccf_velocities, in_ccfs, kind='linear', bounds_error=False, fill_value=np.nan)
    # plt.imshow(f2(velocities)-in_profiles)
    # plt.colorbar()
    # plt.show()

    #print(phases)
    # profile_spec = all_profiles[0].data[0]
    # #velocities=all_profiles[0].header['CRVAL1']+(np.arange(profile_spec.shape[0]))*all_profiles[0].header['CDELT1']
    # velocities = np.linspace(-15,15,len(profile_spec))

    # ccf_spec = all_ccfs[0].data[0]
    # ccf_velocities=all_ccfs[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*all_ccfs[0].header['CDELT1']


   # K = -2.277 #km/s - Boisse et al, 2009
    #velocities = velocities - K  ### Adjusting doppler reflex ###
    residual_profiles, residual_profile_errors = residualccfs(in_profiles, in_profiles_errors, master_out, master_out_errors, velocities)
    # plt.figure()
    # for out_prof in out_profiles:
    #     residual_profiles, residual_profile_errors = residualccfs(out_profiles, out_profile_error, out_prof, master_out_errors, velocities)
    #     # for resi_prof in residual_profiles:
    #     plt.plot(np.mean(residual_profiles, axis = 1))
    # plt.show()
    # residual_ccfs, residual_errors = residualccfs(in_ccfs, in_ccfs_errors, master_out_ccf, master_out_errors_ccf, ccf_velocities)

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
    # phases = np.array(phases)
    # results = np.array(results)
    # print(month)
    # plt.figure()
    # i=0
    # for ccf1 in residual_ccfs:
    #     for phase in phases:
    #         if phases_ccfs[i] == phase:
    #             print(results[tuple([phases==phase])])
    #             if results[tuple([phases==phase])]=='out':colour = '--'
    #             else:colour = '-'
    #             plt.plot(ccf_velocities, ccf1, label = '%s'%(i), linestyle = colour)
    #
    #     #plt.fill_between(ccf_velocities, ccf1-residual_errors[i], ccf1+residual_errors[i], alpha = 0.3)
    #     #all_resi.append(ccf1)
    #     i+=1
    # #plt.legend()
    # plt.show()

    print(month)
    plt.figure(month)
    i=0
    for ccf1 in residual_profiles:
        if results[i]=='out':colour = '--'
        else:colour = '-'
        plt.plot(velocities, ccf1, label = '%s_%s'%(results[i], i), linestyle = colour)
        plt.fill_between(velocities, ccf1-residual_profile_errors[i], ccf1+residual_profile_errors[i], alpha = 0.3)
        all_resi.append(ccf1+1)
        all_prof.append(in_profiles[i])
        i+=1
    plt.legend()
    plt.show()


    #write in data
    hdu=fits.HDUList()
    for data in residual_profiles:
        hdu.append(fits.PrimaryHDU(data=data))

    #hdu.append(fits.PrimaryHDU(data=master_out))

    plt.figure()
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
        # hdu[p].header=hdr
        print(phase)
        print(results[p])
        
        if results[p] == 'out':
            plt.plot(residual_profiles[p], label = '%s, %s'%(phases[p], max(residual_profiles[p])-min(residual_profiles[p])))
    plt.legend()
    plt.show()
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

fig, ax = plt.subplots(2)
#ax[0], ax[1] = fig.add_subplot()
ax[0].set_title('Residual LSD Profiles')
for ccf in all_resi:
    ax[0].plot(velocities, ccf)
ax[0].set_ylabel('Residual Flux')
#plt.savefig('/Users/lucydolan/Documents/CCF_method/Figures/residual_ccfs_mine')

cmap = plt.get_cmap('jet')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size="7%", pad=0.2,)
im = ax[1].pcolormesh(velocities, all_phase, all_resi, cmap = cmap)
ax[1].set_ylim(-0.02, 0.02)
fig.colorbar(im,label='Residual Flux', cax = cax)
ax[1].set_xlabel('Velocity (km/s)')
ax[1].set_ylabel('Phase')
plt.show()

## Cegla style fitting of rvs
x = velocities#[1:]
all_resi = np.array(all_resi)
# all_resi = all_resi[:, 1:]
rvs = []
count = 0

P=2.21857567 #Cegla et al, 2006 - days
t=0.076125 #Torres et al, 2008 - days

rv_phases = []
for y in all_prof:
    #if (-t/(2*P)+0.001)<all_phase[count]<0.0155:
    
        popt, pcov = curve_fit(gauss, x, y)
        perr= np.sqrt(np.diag(pcov))
        rvs.append([popt[0], perr[0]])
        rv_phases.append(all_phase[count])
        count += 1

rvs = np.array(rvs)

plt.figure('RV measured from full profile')
plt.xlabel('Phase')
plt.ylabel('Local RV (km/s)')
plt.errorbar(rv_phases, rvs[:,0], yerr = rvs[:,1], fmt='o', label = 'LSD')

x = velocities#[1:]
all_resi = np.array(all_resi)
#all_resi = all_resi[:, 1:]
rvs = []
count = 0

P=2.21857567 #Cegla et al, 2006 - days
t=0.076125 #Torres et al, 2008 - days

rv_phases = []

plt.figure()
for y in all_resi:
    #if (-t/(2*P)+0.001)<all_phase[count]<0.0155:
    if np.max(y) - np.min(y)>0.006:
        # plt.figure()
        # plt.plot(x, y)
        # plt.show()
        try:
            popt, pcov = curve_fit(gauss, x, y)
            perr= np.sqrt(np.diag(pcov))
            rvs.append([popt[0], perr[0]])
            rv_phases.append(all_phase[count])
        except:
            print('could not fit phase: %s'%all_phase[count])
    else: 
        plt.plot(x, y)
        print(results_all[count])
    count += 1

rvs = np.array(rvs)

# plt.figure('RV measured from residual profile')
# plt.xlabel('Phase')
# plt.ylabel('Local RV (km/s)')
# plt.errorbar(rv_phases, rvs[:,0], yerr = rvs[:,1], fmt='o', label = 'LSD')
# plt.show()

plt.figure('master profiles')
for i in range(len(months)):
    plt.plot(velocities, masters[i], label = '%s'%months[i])
plt.legend()
plt.show()
