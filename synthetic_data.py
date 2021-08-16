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

    deltav = (np.max(p_vel) - np.min(p_vel))/len(p_vel)

    lp=pd.read_csv(linelist, delimiter = ',', usecols=['Spec Ion','WL_air(A)','depth'])
    #print(lp.columns)
    l_wavelengths1 = list(lp.loc[:,'WL_air(A)'])
    l_depths1 = list(lp.loc[:,'depth'])
    l_elements1 = list(lp.loc[:,'Spec Ion' ])

    l_wavelengths1 = np.array(list(l_wavelengths1))
    l_depths1 = np.array(list(l_depths1))
    l_elements1 = np.array(list(l_elements1))


    l_depths = []
    l_wavelengths = []
    l_elements = []

    wavelength_min  = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    for some in range(0, len(l_wavelengths1)):
        if l_wavelengths1[some]>=wavelength_min and l_wavelengths1[some]<=wavelength_max:
            l_wavelengths.append(l_wavelengths1[some])
            l_depths.append(l_depths1[some])
            l_elements.append(l_elements1[some])
        else:
            pass

    l_blankwaves=wavelengths
    velocities=p_vel

    delta_x=np.zeros([len(l_wavelengths), (len(l_blankwaves)*len(velocities))])

    limit=np.max(velocities)*np.max(l_wavelengths)/2.99792458e5

    for j in range(0, len(l_blankwaves)):
        for i in (range(0,len(l_wavelengths))):

            diff=l_blankwaves[j]-l_wavelengths[i]
            if abs(diff)<=(limit):
                vel=2.99792458e5*diff/l_wavelengths[i]
                for k in range(0, len(velocities)):
                    x=(velocities[k]-vel)/deltav
                    if -1.<x and x<0.:
                        delta_x[i,(k+j*len(velocities))]=(1+x)
                    elif 0.<=x and x<1.:
                        delta_x[i,(k+j*len(velocities))]=(1-x)
            else:
                pass

    alpha=[]
    alpha=np.dot(l_depths,delta_x)
    alpha=np.reshape(alpha, (len(l_blankwaves), len(p_vel)))

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

    p1 = p0[4:]
    print(p1)
    print(wavelengths)
    print(np.max(wavelengths))

    mdl =0
    for i in np.arange(0,len(p1)):
        #print(i)
        #print(mdl)
        mdl = mdl+p1[i]*(wavelengths/np.max(wavelengths))**(i-0)
        #mdl = mdl+p1[i]*(wavelengths)**(i-0)
        #print(mdl)
        #print(p1[i])


    print(mdl)
    plt.figure('this is the correct figure')
    plt.plot(wavelengths, spectrum+1)
    plt.plot(wavelengths, mdl, 'k')
    plt.show()


    spectrum = ((spectrum+1)*mdl)
    errors = np.ones(np.shape(spectrum))
    errors = errors*0.001
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
