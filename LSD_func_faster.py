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
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv, spsolve
from scipy.interpolate import interp1d,LSQUnivariateSpline

def LSD(wavelengths, flux_obs, rms, linelist, adjust_continuum, poly_ord, sn, order, run_name, velocities):

    rms = rms/flux_obs
    flux_obs = np.log(flux_obs)
    deltav = velocities[1]-velocities[0]
    
    df = pd.read_csv(linelist, header=None, skiprows=lambda x: x % 4 != 3)
    df = df.iloc[:-(df.apply(lambda x: len(x.dropna()), axis=1) <= 10).idxmax()]
    linelist_expected = df.values.astype(str)

    wavelengths_expected1 =np.array(linelist_expected[:,1], dtype = 'float')
    depths_expected1 = np.array(linelist_expected[:,9], dtype = 'float')
    if np.max(depths_expected1)>1:
        depths_expected1 = np.array(linelist_expected[:,13], dtype = 'float')

    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

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

    depths_expected1 = np.array(depths_expected)
    depths_expected = np.log(1+depths_expected1)
    blankwaves=wavelengths
    R_matrix=flux_obs

    alpha=np.zeros((len(blankwaves), len(velocities)))

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):
            vdiff = ((blankwaves[j] - wavelengths_expected[i])*2.99792458e5)/wavelengths_expected[i]
            if vdiff<=(np.max(velocities)+deltav) and vdiff>=(np.min(velocities)-deltav):
                diff=blankwaves[j]-wavelengths_expected[i]
                if rms[j]<1:no_line.append(i)
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
            
    no_line = list(dict.fromkeys(no_line))
    
    id_matrix=np.identity(len(flux_obs))
    S_matrix=(1/rms)*id_matrix

    S_squared=np.dot(S_matrix, S_matrix)
    alpha_transpose=(np.transpose(alpha))

    RHS_1=np.dot(alpha_transpose, S_squared)
    RHS_final=np.dot(RHS_1, R_matrix )

    LHS_preinvert=np.dot(RHS_1, alpha)
    LHS_prep=np.matrix(LHS_preinvert)

    P,L,U=linalg.lu(LHS_prep)

    n=len(LHS_prep)
    B=np.identity(n)
    Z = linalg.solve_triangular(L, B, lower=True)
    X = linalg.solve_triangular(U, Z, lower=False)
    LHS_final = np.matmul(X,np.transpose(P))

    profile=np.dot(LHS_final, RHS_final)
    profile_errors_squared=np.diagonal(LHS_final)
    profile_errors=np.sqrt(profile_errors_squared)

    return velocities, profile, profile_errors, alpha, wavelengths_expected, depths_expected1, len(no_line)

def continuumfit(wavelengths1, fluxes1, poly_ord):

        fluxes = fluxes1
        wavelengths = wavelengths1

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
        fit = poly(wavelengths1)
        flux_obs = fluxes1/fit
        
        return flux_obs

## functions below are specific for HARPS spectra
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

  return wave

def blaze_correct(file_type, spec_type, order, file, directory, masking, run_name, berv_opt):
    
    if file_type == 's1d':
        #### Finding min and max wavelength from e2ds for the specified order ######
        file_e2ds = file.replace('s1d', 'e2ds')
        print(file_e2ds)
        hdu=fits.open('%s'%file_e2ds)
        sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        spec=hdu[0].data
        header=hdu[0].header
        brv=header['ESO DRS BERV']
        # print('hi')
        spec_check = spec[spec<=0]
        if len(spec_check)>0:
            print('WARNING NEGATIVE/ZERO FLUX - corrected')

        flux_error = np.sqrt(spec)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spec == 0)[0]
        flux_error[where_are_zeros] = 1000000000000

        wave=get_wave(spec, header)*(1.+brv/2.99792458e5)
        wavelengths_order = wave[order]
        wavelength_min = np.min(wavelengths_order)
        wavelength_max = np.max(wavelengths_order)

        ## remove overlapping region (always remove the overlap at the start of the order, i.e the min_overlap)
        last_wavelengths = wave[order-1]
        next_wavelengths = wave[order+1]
        min_overlap = np.max(last_wavelengths)
        max_overlap = np.min(next_wavelengths)

        hdu.close()

        #### Now reading in s1d file ########
        print(file)
        hdu=fits.open('%s'%file)
        spec=hdu[0].data
        header=hdu[0].header
        spec_check = spec[spec<=0]
        wave=hdu[0].header['CRVAL1']+(np.arange(spec.shape[0]))*hdu[0].header['CDELT1']

        if spec_type == 'order':
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

        elif spec_type == 'full':
            ## not set up properly.
            wavelengths = wave
            fluxes = spec

    elif file_type == 'e2ds':
        hdu=fits.open('%s'%file)
        spec=hdu[0].data
        header=hdu[0].header
        sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        print('S/N: %s'%sn)
        spec_check = spec[spec<=0]
        if len(spec_check)>0:
            print('WARNING NEGATIVE/ZERO FLUX - corrected')

        flux_error = np.sqrt(spec)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spec == 0)[0]
        flux_error[where_are_zeros] = 1000000000000
        brv=np.float128(header['ESO DRS BERV'])
        wave_nonad=get_wave(spec, header)
        wave = wave_nonad*(1.+brv/2.99792458e5)
        wave = np.array(wave, dtype = 'float64')
    
        blaze_file = glob.glob('%sblaze_folder/**blaze_A*.fits'%(directory))
        blaze_file = blaze_file[0]
        blaze =fits.open('%s'%blaze_file)
        blaze_func = blaze[0].data
        spec = spec/blaze_func
        flux_error = flux_error/blaze_func

        fluxes = spec[order]
        flux_error_order = flux_error[order]
        wavelengths = wave[order]

        # find overlapping regions 
        last_wavelengths = wave[order-1]
        next_wavelengths = wave[order+1]
        last_spec = spec[order-1]
        next_spec = spec[order+1]
        last_error = flux_error[order-1]
        next_error = flux_error[order+1]
        min_overlap = np.min(wavelengths)
        max_overlap = np.max(wavelengths)
        last_idx = np.logical_and(last_wavelengths>min_overlap, last_wavelengths<max_overlap)
        next_idx = np.logical_and(next_wavelengths>min_overlap, next_wavelengths<max_overlap)
        overlap = np.array(([list(last_wavelengths[last_idx]), list(last_spec[last_idx]), list(last_error[last_idx])], [list(next_wavelengths[next_idx]), list(next_spec[next_idx]), list(next_error[next_idx])]))

        hdu.close()

    return np.array(fluxes), np.array(wavelengths), np.array(flux_error_order), sn, np.median(wavelengths), np.zeros(wavelengths.shape), overlap
