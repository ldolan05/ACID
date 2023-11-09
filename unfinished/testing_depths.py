import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz

def voigt_wofz(u, a):
    prof=wofz(u + 1j * a).real
    return prof/np.max(prof)

def voigt_func(x, pars):
    a = pars[1]/np.sqrt(2)*pars[0]
    u = x-pars[2]/np.sqrt(2)*pars[0]
    mdl=voigt_wofz(u,a)
    return mdl

def gauss(x, A, linewidth):
    profile=1.-A*np.exp( -(x**2)/2./linewidth**2)
    return profile

plt.figure()
velocities = np.linspace(-15, 15, 48)
for A in np.arange(0.00001, 0.2, 0.0001):
    ## create voigt profile with depth A
    linewidth = 3
    p0=[0., 1., linewidth] #height, offset, width, polynomial coefficents
    profile = A*voigt_func(velocities, p0)

    profile=1.-profile

    ## fit gaussian to the profile and save depth
    popt, pcov = curve_fit(gauss, velocities, profile)
    plt.scatter(A, popt[0], marker = '.')

plt.plot(np.arange(0.00001, 0.2, 0.0001), np.arange(0.00001, 0.2, 0.0001), linestyle = '--')
plt.xlabel('Voigt Depth')
plt.ylabel('Gaussian Depth')
plt.show()