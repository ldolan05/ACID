import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import emcee
from scipy.special import wofz

# gaussian model
def gauss(x, rv, sd, height, cont):
    y = cont+(height*np.exp(-(x-rv)**2/(2*sd**2)))
    return y

def create_alpha(velocities, linelist, wavelengths):
    deltav = velocities[1]-velocities[0]
    
    #### This is the EXPECTED linelist (for a slow rotator of the same spectral type) ####
    linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
    wavelengths_expected1 =np.array(linelist_expected[:,0])
    depths_expected1 = np.array(linelist_expected[:,1])
    # print(len(depths_expected1))

    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    wavelengths_expected=[]
    depths_expected=[]
    no_line =[]
    for some in range(0, len(wavelengths_expected1)):
        line_min = 1/300
        # line_min = 0.01
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max and depths_expected1[some]>=line_min:
            wavelengths_expected.append(wavelengths_expected1[some])
            #depths_expected.append(depths_expected1[some]+random.uniform(-0.1, 0.1))
            depths_expected.append(depths_expected1[some])
        else:
            pass
        
    depths_expected1 = np.array(depths_expected)
    depths_expected = -np.log(1-depths_expected1)

    blankwaves=wavelengths

    # alpha=np.zeros((len(blankwaves), len(velocities)))

    # for j in range(0, len(blankwaves)):
    #     for i in (range(0,len(wavelengths_expected))):
    #         vdiff = ((blankwaves[j] - wavelengths_expected[i])*2.99792458e5)/wavelengths_expected[i]
    #         if vdiff<=(np.max(velocities)+deltav) and vdiff>=(np.min(velocities)-deltav):
    #             diff=blankwaves[j]-wavelengths_expected[i]
    #             vel=2.99792458e5*(diff/wavelengths_expected[i])
    #             for k in range(0, len(velocities)):
    #                 x=(velocities[k]-vel)/deltav
    #                 if -1.<x and x<0.:
    #                     delta_x=(1+x)
    #                     alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
    #                 elif 0.<=x and x<1.:
    #                     delta_x=(1-x)
    #                     alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
    #         else:
    #             pass
    
    # Find differences and velocities
    diff = blankwaves[:, np.newaxis] - wavelengths_expected
    vel = 2.99792458e5 * (diff / wavelengths_expected)

    # Calculate x and delta_x for valid velocities
    x = (vel[:, :, np.newaxis] - velocities) / deltav
    alpha_mask_1 = np.logical_and(-1. < x, x < 0.)
    alpha_mask_2 = np.logical_and(0. <= x, x < 1.)
    delta_x_1 = 1 + x
    delta_x_2 = 1 - x
    delta_x_1[alpha_mask_1==False]=0
    delta_x_2[alpha_mask_2==False]=0

    # Update alpha array using calculated delta_x values
    alpha = np.zeros((len(blankwaves), len(velocities)))
    alpha += (depths_expected[:, np.newaxis] * delta_x_1).sum(axis=1)
    alpha += (depths_expected[:, np.newaxis] * delta_x_2).sum(axis=1)

    return alpha

# making alpha
linelist = '/Users/lucydolan/Starbase/novaprime/Documents/fulllinelist0001.txt'
wavelengths = np.arange(4000, 4100, 0.015)
velocities = np.arange(-25, 25, 0.82)
alpha = create_alpha(velocities, linelist, wavelengths)

for spec_no in range(1, 4):
    print('Running for spectrum %s/4'%spec_no)
    ## create synthetic profile
    number = np.random.normal(0, 4)
    profile = gauss(velocities, 0.+number, 4, -0.4, 1)
    # profile_errors = (1-profile)/10
    input_profile = profile.copy()
    number = np.random.normal(0, 0.001, size=profile.shape)
    profile_errors = abs(number)
    input_profile_err = profile_errors.copy()
    # profile = profile + number

    ## create synthetic spectrum
    # creates a fake data set to fit a polynomial to - is just an easy way to get continuum coefficents that make sense for the wavelength range
    flux = [0.98, 1.048, 0.85, 1.03, 0.9, 1, 0.82, 1.037]
    waves = np.linspace(np.min(wavelengths), np.max(wavelengths), len(flux))
    ## used for adjusting wavelegnths to between -1 and 1
    a = 2/(np.max(waves)-np.min(waves))
    b = 1 - a*np.max(waves)
    p1=np.polyfit(((waves*a)+b), flux, 3)
    p1 = p1[::-1]
    mdl =0
    for i in np.arange(0,len(p1)):
        mdl = mdl+p1[i]*((a*wavelengths)+b)**(i-0)
    spectrum = np.exp((np.dot(alpha, np.log(profile))))

    errors = np.zeros((len(alpha), ))
    for j in range(len(alpha)):
        errors[j] = np.sqrt(sum((alpha[j, :]*(profile_errors[:]/profile))**2))
    errors = spectrum*(np.dot(alpha, profile_errors/profile))
    errors[errors==0]=0.00001

    data = [wavelengths, spectrum, errors, [100]]
    hdu = fits.HDUList()
    for dat in data:
        hdu.append(fits.PrimaryHDU(data = dat))
    hdu.writeto('sample_spec_%s.fits'%spec_no, output_verify = 'fix', overwrite = True)

