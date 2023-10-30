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


## makes a guassian profile
def make_profile(x, pars):
    mdl = pars[3]+(1+pars[1]*np.exp(-(x-pars[2])**2/2.*pars[0]**2))
    return mdl-1

## combines linelist and profile
def convolve(p_vel, p_fluxes, linelist, wavelengths):

    ## must match LSD if comparing forard models
    deltav = 0.8

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

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):

            diff=blankwaves[j]-wavelengths_expected[i]
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

# makes synthetic spectrum with given profile parameters and continuum coefficents
def make_spectrum(vgrid, p0, wavelengths, linelist):
    profile = make_profile(vgrid, p0)
    wavelengths, spectrum = convolve(vgrid, profile, linelist, wavelengths)

    a = 2/(np.max(wavelengths)-np.min(wavelengths))
    b = 1 - a*np.max(wavelengths)
    p1 = p0[4:]

    mdl =0
    for i in np.arange(0,len(p1)):
        mdl = mdl+p1[i]*((a*wavelengths)+b)**(i-0)

    plt.figure('synthetic spectrum')
    plt.plot(wavelengths, spectrum+1)
    plt.plot(wavelengths, mdl, 'k')
    plt.show()

    spectrum = ((spectrum+1)*mdl)
    errors = np.ones(np.shape(spectrum))
    errors = errors*0.01

    return spectrum-1, errors, profile
