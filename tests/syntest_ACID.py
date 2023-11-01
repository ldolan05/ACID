import ACID_code as ACID_code
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import emcee
import LSD_func_faster as LSD
from scipy.special import wofz

# gaussian model
def gauss(x, rv, sd, height, cont):
    y = cont+(height*np.exp(-(x-rv)**2/(2*sd**2)))
    return y

# lorentian model
def lorentz(x, rv, gamma, height, cont):
    y = cont + (1-height*((1/np.pi)*(gamma/((x-rv)**2+gamma**2))))
    return y 

# #voigt model - currently not working - makes RV incorrect when not 0.
# def voigt_func(x, rv, height, cont, gamma, sd):
#     gausss = np.exp(-(x-rv)**2/(2*sd**2))
#     lor = (gamma)/((x-rv)**2 + gamma**2)
#     lor = lor/np.max(lor)
#     plt.figure()
#     plt.plot(x, gausss, label = 'gauss')
#     plt.plot(x, lor, label = 'lorentz')
#     profile = (np.convolve(lor, gausss, 'same'))
#     profile = profile/np.max(profile)
#     plt.plot(x, profile, label = 'convolution')
#     plt.legend()
#     plt.show()
#     return cont+(height*profile)

def voigt_wofz(u, a):
    prof=wofz(u + 1j * a).real
    return prof/np.max(prof)

def voigt_func(x, pars):
    #pars[0, 1, 2, 3, 4] = sigma, gamma, rv, height, cont
    a = pars[1]/np.sqrt(2)*pars[0]
    u = x-pars[2]/np.sqrt(2)*pars[0]
    mdl=voigt_wofz(u,a)
    return pars[4]+pars[3]*mdl

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
        # line_min = 1/300
        line_min = 0.01
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max and depths_expected1[some]>=line_min:
            wavelengths_expected.append(wavelengths_expected1[some])
            #depths_expected.append(depths_expected1[some]+random.uniform(-0.1, 0.1))
            depths_expected.append(depths_expected1[some])
        else:
            pass
    
    depths_expected1 = np.array(depths_expected)
    depths_expected = -np.log(1-depths_expected1)

    blankwaves=wavelengths

    alpha=np.zeros((len(blankwaves), len(velocities)))

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):
            vdiff = ((blankwaves[j] - wavelengths_expected[i])*2.99792458e5)/wavelengths_expected[i]
            if vdiff<=(np.max(velocities)+deltav) and vdiff>=(np.min(velocities)-deltav):
                diff=blankwaves[j]-wavelengths_expected[i]
                vel=2.99792458e5*(diff/wavelengths_expected[i])
                for k in range(0, len(velocities)):
                    x=(velocities[k]-vel)/deltav
                    if -1.<x and x<0.:
                        delta_x=(1+x)
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
                    elif 0.<=x and x<1.:
                        delta_x=(1-x)
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
            else:
                pass

    return alpha

def get_wave(data,header):

  wave=np.array(data*0., dtype = 'float128')
  no=data.shape[0]
  npix=data.shape[1]
  d=header['ESO DRS CAL TH DEG LL']
  xx0=np.arange(npix)
  xx=[]
  for i in range(d+1):
      xx.append(xx0**i)
  xx=np.asarray(xx, dtype = 'float128')

  for o in range(no):
      for i in range(d+1):
          idx=i+o*(d+1)
          par=np.float128(header['ESO DRS CAL TH COEFF LL%d' % idx])
          wave[o,:]=wave[o,:]+par*xx[i,:]
       #for x in range(npix):
       #  wave[o,x]=wave[o,x]+par*xx[i,x]#float(x)**float(i)

  return np.float64(wave)

## create synthetic profile
# velocities = np.concatenate((-np.arange(0.83, 25, 0.83), np.arange(0, 25, 0.83)))
velocities = np.concatenate((-np.arange(0.82+0.5, 25+0.5, 0.82), np.arange(0-0.5, 25-0.5, 0.82)))
# velocities.sort()
velocities = np.arange(-25, 25, 0.82)
rv = 0.25
# profile = gauss(velocities, rv, 4, -0.4, 1)
profile = voigt_func(velocities, [2, 2, rv, -0.4, 1])
# profile_errors = (1-profile)/10
input_profile = profile.copy()
number = np.random.normal(0, 0.01, size=profile.shape)
profile_errors = abs(number)
input_profile_err = profile_errors.copy()
profile = profile + number

plt.figure()
plt.plot(velocities, profile)
plt.show()

## create synthetic spectrum
linelist = '/Users/lucydolan/Starbase/novaprime/Documents/fulllinelist0001.txt'
filelist=ACID_code.findfiles('/Users/lucydolan/Starbase/novaprime/Documents/HD189733 old/HD189733/August2007', 'e2ds')
file = fits.open(filelist[0])
wavelengths =get_wave(file[0].data, file[0].header)[30]
alpha = create_alpha(velocities, linelist, wavelengths)
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

# number = np.random.normal(0, 0.01, size=spectrum.shape)
# spectrum = spectrum + number
# errors = abs(number)

errors = np.zeros((len(alpha), ))
for j in range(len(alpha)):
    errors[j] = np.sqrt(sum((alpha[j, :]*(profile_errors[:]/profile))**2))
errors = spectrum*(np.dot(alpha, profile_errors/profile))
errors[errors==0]=0.00001

plt.figure()
plt.errorbar(wavelengths, spectrum, errors)
plt.show()

velocities, profile, profile_errors, alpha, wavelengths_expected, depths_expected1, no_line = LSD.LSD(wavelengths, spectrum.copy(), errors, linelist, 'n', 3, 1/(3*0.01), 10, 'test', velocities)

plt.figure('Profile with rv = %s (errors)'%rv)
plt.plot(velocities, np.sqrt(profile_errors**2/np.exp(profile)**2), label = 'output profile errors - original velocity grid (np.arange(-25, 25, 0.82))')

# plt.plot(velocities, 1-np.sqrt(profile+1), label = 'np.sqrt(profile), sqrt errors')

plt.figure('Profile with rv = %s (profiles)'%rv)
plt.plot(velocities, np.exp(profile), label = 'output profile - original velocity grid (np.arange(-25, 25, 0.82))')

velocities = np.concatenate((-np.arange(0.82-rv, 25-rv, 0.82), np.arange(0+rv, 25+rv, 0.82)))
velocities.sort()
velocities, profile, profile_errors, alpha, wavelengths_expected, depths_expected1, no_line = LSD.LSD(wavelengths, spectrum.copy(), errors, linelist, 'n', 3, 1/(3*0.01), 10, 'test', velocities)

plt.figure('Profile with rv = %s (errors)'%rv)
plt.title('Profile with rv = %s (errors)'%rv)
plt.plot(velocities, np.sqrt(profile_errors**2/np.exp(profile)**2), label = 'output profile errors - velocity grid centered at %s'%rv)
# plt.plot(np.arange(-25, 25, 0.82), input_profile_err, label = 'input profile errors')
# plt.legend()

plt.figure('Profile with rv = %s (profiles)'%rv)
plt.title('Profile with rv = %s (profiles)'%rv)
plt.plot(velocities, np.exp(profile), label = 'output profile - velocity grid centered at %s'%rv)
# plt.plot(np.arange(-25, 25, 0.82), input_profile, label = 'input profile', linestyle='--')
# plt.legend()

velocities = np.concatenate((-np.arange(0.82, 25, 0.82), np.arange(0, 25, 0.82)))
velocities.sort()
velocities, profile, profile_errors, alpha, wavelengths_expected, depths_expected1, no_line = LSD.LSD(wavelengths, spectrum.copy(), errors, linelist, 'n', 3, 1/(3*0.01), 10, 'test', velocities)

plt.figure('Profile with rv = %s (errors)'%rv)
plt.title('Profile with rv = %s (errors)'%rv)
plt.plot(velocities, np.sqrt(profile_errors**2/np.exp(profile)**2), label = 'output profile errors - velocity grid centered at 0')
plt.plot(np.arange(-25, 25, 0.82), input_profile_err/np.sqrt(no_line), label = 'input profile errors / sqrt(number of lines)')
# plt.vlines([0.25], 0, 0.14)
plt.legend()
plt.savefig('profile_errors_lsd.png')

plt.figure('Profile with rv = %s (profiles)'%rv)
plt.title('Profile with rv = %s (profiles)'%rv)
plt.plot(velocities, np.exp(profile), label = 'output profile - velocity grid centered at 0')
plt.plot(np.arange(-25, 25, 0.82), input_profile, label = 'input profile', linestyle='--')
# plt.vlines([0.25], 0, 1)
plt.legend()
plt.savefig('profile_lsd.png')

plt.show()

spectrum = spectrum*mdl

plt.figure()
plt.plot(wavelengths, spectrum)
plt.show()

all_frames = ACID_code.ACID([wavelengths], [spectrum], [errors], linelist, sns1=[1/(3*0.01)])

output_profile = all_frames[0, 0, 0]
output_profile_err = all_frames[0, 0, 1]

plt.figure()
plt.plot(velocities, output_profile)

plt.figure()
plt.title('Error output from ACID')
# plt.plot(velocities, output_profile_err*np.sqrt(no_line), label = 'ACID Output Profile Error * sqrt(number of lines)')
plt.plot(velocities, output_profile_err , label = 'ACID Output Profile Error')
plt.plot(velocities, input_profile_err, label = 'Input Profile Error')
plt.legend()
plt.savefig('output_ACID_error.png')
# plt.plot(velocities, number, label = 'noise')
plt.show()


