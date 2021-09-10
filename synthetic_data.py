import numpy as np
import emcee
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import minimize
import emcee
#import corner
import LSD_func_faster as LSD
import time
import pandas as pd

#linelist = '/Users/lucydolan/Documents/Least_Squares_Deconvolution/LSD/Archive_stuff/archive/fulllinelist018.txt'

def make_profile(x, pars):
    #print(pars)

    mdl = pars[3]+(1+pars[1]*np.exp(-(x-pars[2])**2/2.*pars[0]**2))
    '''
    plt.figure('profile')
    plt.plot(x, mdl-1)
    plt.savefig('/home/lsd/Documents/original_profile_syn.png')
    plt.show()
    '''
    return mdl-1

#p0=[0.1,0.1,0.,0.,0., 0.,0.] #sigma, gamma, offset, height, polynomial coefficents

def convolve(p_vel, p_fluxes, linelist, wavelengths):

    deltav = 0.8
    print(deltav)

    linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
    wavelengths_expected1 =np.array(linelist_expected[:,0])
    depths_expected1 = np.array(linelist_expected[:,1])

    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    wavelengths_expected=[]
    depths_expected=[]
    for some in range(0, len(wavelengths_expected1)):
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max:
            wavelengths_expected.append(wavelengths_expected1[some])
            depths_expected.append(depths_expected1[some])
        else:
            pass

    blankwaves=wavelengths

    alpha=np.zeros((len(blankwaves), len(p_vel)))

    limit=np.max(p_vel)*np.max(wavelengths_expected)/2.99792458e5
    #print(limit)

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):

            diff=blankwaves[j]-wavelengths_expected[i]
            #limit=np.max(velocities)*wavelengths_expected[i]/2.99792458e5
            if abs(diff)<=(limit):
                vel=2.99792458e5*diff/wavelengths_expected[i]
                for k in range(0, len(p_vel)):
                    x=(p_vel[k]-vel)/deltav
                    if -1.<x and x<0.:
                        delta_x=(1+x)
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
                    elif 0.<=x and x<1.:
                        delta_x=(1-x)
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
            else:
                pass

    spectrum=np.dot(alpha, p_fluxes)

    return wavelengths, spectrum
'''
vgrid = np.arange(-25, 25, 0.8)
p0=[0.36, -0.6, 0, 0, 11.38 , -0.000422450, -0.000000408]
wavelengths = np.arange(4575, 4626, 0.1)
'''
def make_spectrum(vgrid, p0, wavelengths, linelist):
    profile = make_profile(vgrid, p0)
    wavelengths, spectrum = convolve(vgrid, profile, linelist, wavelengths)
    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)

    p1 = p0[4:]
    #print(p1)
    #print(wavelengths)
    #print(np.max(wavelengths))

    mdl =0
    for i in np.arange(0,len(p1)):
        #print(i)
        #print(mdl)
        #mdl = mdl+p1[i]*(wavelengths/np.max(wavelengths))**(i-0)

        mdl = mdl+p1[i]*((a*wavelengths)+b)**(i-0)
        #print(mdl)
        #print(p1[i])


    print(mdl)
    plt.figure('synthetic spectrum')
    plt.plot(wavelengths, spectrum+1)
    plt.plot(wavelengths, mdl, 'k')
    plt.show()



    spectrum = ((spectrum+1)*mdl)
    errors = np.ones(np.shape(spectrum))
    errors = errors*0.01
    #errors = np.sqrt(spectrum)
    #where_are_NaNs = np.isnan(errors)
    #errors[where_are_NaNs] = 10000000000

    #print(where_are_nans)

    #print(spectrum, errors)

    return spectrum-1, errors, profile
'''
spectrum, errors = make_spectrum(vgrid, p0, wavelengths, linelist)

plt.figure()
plt.plot(wavelengths, spectrum)
plt.show()
'''
