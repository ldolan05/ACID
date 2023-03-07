import numpy as np
from scipy import linalg
from astropy.io import  fits
import glob
import matplotlib.pyplot as plt
import random
from astropy import units as u
from specutils import Spectrum1D
from scipy.signal import find_peaks
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv, spsolve
from scipy.interpolate import interp1d,LSQUnivariateSpline

def LSD(wavelengths, flux_obs, rms, linelist, adjust_continuum, poly_ord, sn, berv, offset, velocities):

    # convert spectrum into optical depth
    rms = rms/flux_obs
    flux_obs = np.log(flux_obs)
    deltav = velocities[1]-velocities[0]

    # Read in the expected line list from VALD
    linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
    wavelengths_expected1 =np.array(linelist_expected[:,0])
    depths_expected1 = np.array(linelist_expected[:,1])

    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    # Filtering out small lines and those outside of the spectrum's wavelength range
    wavelengths_expected=[]
    depths_expected=[]
    no_line =[]
    for some in range(0, len(wavelengths_expected1)):
        line_min = 1/(3*sn)
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max and depths_expected1[some]>=line_min:
            wavelengths_expected.append(wavelengths_expected1[some])
            depths_expected.append(depths_expected1[some])
        else:
            pass

    # Convert depths from linelist in optical depth
    depths_expected1 = np.array(depths_expected)
    depths_expected = np.log(1+depths_expected1)

    # Constructing Alpha Matrix
    alpha=np.zeros((len(wavelengths), len(velocities)))
    delta_x_array = alpha.copy()

    for j in range(0, len(wavelengths)):
        for i in (range(0,len(wavelengths_expected))):
            vdiff = ((wavelengths[j] - wavelengths_expected[i])*2.99792458e5)/wavelengths_expected[i]
            if vdiff<=(np.max(velocities)+deltav) and vdiff>=(np.min(velocities)-deltav):
                diff=wavelengths[j]-wavelengths_expected[i]
                vel=2.99792458e5*(diff/wavelengths_expected[i])
                for k in range(0, len(velocities)):
                    x=(velocities[k]-vel)/deltav
                    if -1.<x and x<0.:
                        delta_x=(1+x)
                        if i == int(len(depths_expected)/2)+1:
                            delta_x_array[j, k] = delta_x
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
                    elif 0.<=x and x<1.:
                        delta_x=(1-x)
                        if i == int(len(depths_expected)/2)+1:
                            delta_x_array[j, k] = delta_x
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
            else:
                pass
    
    # Constructing S_matrix (containg errors)
    id_matrix=np.identity(len(flux_obs))
    S_matrix=(1/rms)*id_matrix

    S_squared=np.dot(S_matrix, S_matrix)
    alpha_transpose=(np.transpose(alpha))

    # Minimising Chi-Squared to find final profile
    RHS_1=np.dot(alpha_transpose, S_squared)
    RHS_final=np.dot(RHS_1, flux_obs)

    LHS_preinvert=np.dot(RHS_1, alpha)
    LHS_prep=np.matrix(LHS_preinvert, dtype = 'float64')

    P,L,U=linalg.lu(LHS_prep)

    n=len(LHS_prep)
    B=np.identity(n)
    Z = linalg.solve_triangular(L, B, lower=True)
    X = linalg.solve_triangular(U, Z, lower=False)
    LHS_final = np.matmul(X,np.transpose(P))

    profile=np.dot(LHS_final, RHS_final)
    profile_errors_squared=np.diagonal(LHS_final)
    profile_errors=np.sqrt(profile_errors_squared)

    return velocities, profile, profile_errors, alpha

# Calculates wavelength grid for HARPS e2ds spectra
def get_wave(data,header): 

    wave=np.float128(data*0.) 
    no=data.shape[0] 
    npix=data.shape[1] 
    try:d=header['ESO DRS CAL TH DEG LL'] 
    except:d=header['TNG DRS CAL TH DEG LL'] 
    xx0=np.arange(npix) 
    xx=[] 
    for i in range(d+1): 
        xx.append(xx0**i) 
    xx=np.asarray(xx) 

    for o in range(no): 
        for i in range(d+1): 
            idx=i+o*(d+1) 
            try:par=np.float128(header['ESO DRS CAL TH COEFF LL%d' % idx]) 
            except:par=np.float128(header['TNG DRS CAL TH COEFF LL%d' % idx]) 
            wave[o,:]=wave[o,:]+par*xx[i,:]  
    return wave 

def blaze_correct(file_type, order, file, directory):
    #### Inputing spectrum depending on file_type and spec_type #####

    if file_type == 's1d':
        #### Finding min and max wavelength from e2ds for the specified order ######
        file_e2ds = file.replace('s1d', 'e2ds')
        print(file_e2ds)
        hdu=fits.open('%s'%file_e2ds)
        sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        spec=hdu[0].data
        header=hdu[0].header
        brv=header['ESO DRS BERV']

        flux_error = np.sqrt(spec)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spec == 0)[0]
        flux_error[where_are_zeros] = 1000000000000

        wave=get_wave(spec, header)*(1.+brv/2.99792458e5)
        wavelengths_order = wave[order]
        wavelength_min = np.min(wavelengths_order)
        wavelength_max = np.max(wavelengths_order)

        hdu.close()

        #### Now reading in s1d file ########
        hdu=fits.open('%s'%file)
        spec=hdu[0].data
        header=hdu[0].header
        spec_check = spec[spec<=0]
        wave=hdu[0].header['CRVAL1']+(np.arange(spec.shape[0]))*hdu[0].header['CDELT1']
        wavelengths=[]
        fluxes=[]
        for value in range(0, len(wave)):
            l_wave=wave[value]
            l_flux=spec[value]
            if l_wave>=wavelength_min and l_wave<=wavelength_max:
                wavelengths.append(l_wave)
                fluxes.append(l_flux)
        wavelengths = np.array(wavelengths)
        fluxes = np.array(fluxes)

        if len(wavelengths)>5144:
            wavelengths = wavelengths[:5144]
            fluxes = fluxes[:5144]
    
        overlap = []
        spec_check = fluxes[fluxes<=0]
        if len(spec_check)>0:
            print('WARNING NEGATIVE/ZERO FLUX - corrected')

        flux_error = np.sqrt(fluxes)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(fluxes == 0)[0]
        flux_error[where_are_zeros] = 1000000000000

        flux_error_order = flux_error
        masked_waves = []
        masked_waves = np.array(masked_waves)

    elif file_type == 'e2ds':
        hdu=fits.open('%s'%file)
        spec=hdu[0].data
        header=hdu[0].header
        sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        flux_error = np.sqrt(spec)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spec == 0)[0]
        flux_error[where_are_zeros] = 1000000000000

        brv=np.float128(header['ESO DRS BERV'])
        wave_nonad=get_wave(spec, header)
        wave = wave_nonad*(1.+brv/2.99792458e5)
        
        blaze_file = glob.glob('%sblaze_folder/**blaze_A*.fits'%(directory))
        blaze_file = blaze_file[0]
        blaze =fits.open('%s'%blaze_file)
        blaze_func = blaze[0].data
        spec = spec/blaze_func
        flux_error = flux_error/blaze_func
    
        fluxes = spec[order]
        flux_error_order = flux_error[order]
        wavelengths = wave[order]

        hdu.close()

    return fluxes, wavelengths, flux_error_order, sn

