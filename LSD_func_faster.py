import numpy as np
from scipy import linalg
from astropy.io import  fits
import glob
import matplotlib.pyplot as plt
import random
import itertools

def LSD(wavelengths, flux_obs, rms, linelist, sn):

    ## converts spectrum into opacity space
    rms = rms/flux_obs
    flux_obs = np.log(flux_obs)

    '''
    resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    deltav = resol1/(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
    print(deltav)

    resol1 = deltav*(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
    shift = int(centre/deltav)
    centre1 = shift*deltav

    vmin = int(centre1-(width/2))
    vmax = int(centre1+(width/2))
    no_pix = int(width/deltav)
    '''
    
    ## setting up velocity grid 
    vmin = -21
    vmax = 18
    no_pix = 48

    velocities=np.linspace(vmin,vmax,int(no_pix))
    deltav = velocities[1]-velocities[0]

    ## reading in line depths from linelist (reads in for all wavelengths)
    
    ## from MM-LSD
    counter = 0
    for line in reversed(open(linelist).readlines()):
        if len(line.split(",")) > 10:
            break
        counter += 1
    num_lines = sum(1 for line in open(linelist))
    last_line = num_lines - counter + 3

    with open(linelist) as f_in:
        x = np.genfromtxt(itertools.islice(f_in, 3, last_line, 4), dtype=str,delimiter=',')

    wavelengths_expected1 = x[:,1].astype(float)
    depths_expected1 = x[:,9].astype(float)

    # linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
    # # linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, skip_footer=100, delimiter='\t', usecols=(1,9))
    # wavelengths_expected1 =np.array(linelist_expected[:,0])
    # depths_expected1 = np.array(linelist_expected[:,1])

    ## filters out unneccessary line depths
    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    line_min = 1/(3*sn)

    idx = np.logical_and(wavelengths_expected1>wavelength_min, wavelengths_expected1<wavelength_max)
    wavelengths_expected = wavelengths_expected1[idx]
    depths_expected = depths_expected1[idx]
    
    idx = tuple([depths_expected>line_min])
    wavelengths_expected = wavelengths_expected[idx]
    depths_expected = depths_expected[idx]

    no_line = []

    ## conversion for depths into optical depth
    depths_expected1 = np.array(depths_expected)
    depths_expected = np.log(1+depths_expected1)

    ## constructing the alpha matrix
    blankwaves=wavelengths
    R_matrix=flux_obs

    alpha=np.zeros((len(blankwaves), len(velocities)))

    limit=max(abs(velocities))*max(wavelengths_expected)/2.99792458e5

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):
            diff=blankwaves[j]-wavelengths_expected[i]
            if abs(diff)<=(limit):
                if rms[j]<1:no_line.append(i)
                vel=2.99792458e5*diff/wavelengths_expected[i]
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

    ## performing LSD
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

def get_wave(data,header):

  wave=data*0.
  no=data.shape[0]
  npix=data.shape[1]
  d=header['ESO DRS CAL TH DEG LL']
  xx0=np.arange(npix)
  xx=[]
  for i in range(d+1):
      xx.append(xx0**i)
  xx=np.asarray(xx)

  for o in range(no):
      for i in range(d+1):
          idx=i+o*(d+1)
          par=header['ESO DRS CAL TH COEFF LL%d' % idx]
          wave[o,:]=wave[o,:]+par*xx[i,:]

  return wave

def blaze_correct(file_type, spec_type, order, file, directory, masking, run_name):
    print(file_type)
    if file_type == 's1d':
        #### Finding min and max wavelength from e2ds for the specified order ######
        file_e2ds = file.replace('s1d', 'e2ds')
        #print(file_e2ds)
        hdu=fits.open('%s'%file_e2ds)
        sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        spec=hdu[0].data
        header=hdu[0].header
        spec_check = spec[spec<=0]
        if len(spec_check)>0:
            print('WARNING NEGATIVE/ZERO FLUX - corrected')

        flux_error = np.sqrt(spec)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spec == 0)[0]
        flux_error[where_are_zeros] = 1000000000000

        wave=get_wave(spec, header)
        wavelengths_order = wave[order]
        wavelength_min = np.min(wavelengths_order)
        wavelength_max = np.max(wavelengths_order)
        # wavelength_min = 5900
        # wavelength_max = wavelength_min+200   ## if you want to do a set wavelength range just input min and max here
        hdu.close()

        #### Now reading in s1d file ########
        #print(file)
        hdu=fits.open('%s'%file)
        spec=hdu[0].data
        header=hdu[0].header
        spec_check = spec[spec<=0]

        wave=hdu[0].header['CRVAL1']+(hdu[0].header['CRPIX1']+np.arange(spec.shape[0]))*hdu[0].header['CDELT1']
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

        elif spec_type == 'full':
            ## not set up properly.
            wavelengths = wave
            fluxes = spec

    elif file_type == 'e2ds':
        hdu=fits.open('%s'%file)
        spec=hdu[0].data
        header=hdu[0].header
        # print(header)
        # print(len(hdu))
        # for i in range(len(hdu)):
        #     print(hdu[i].data)
        #     try:
        #         plt.figure()
        #         plt.plot(hdu[i].data[0])
        #         plt.show()
        #     except: print('None')
        # print(spec)
        #print('S/N: %s'%sn)
        spec_check = spec[spec<=0]
        if len(spec_check)>0:
            print('WARNING NEGATIVE/ZERO FLUX - corrected')

        flux_error = np.sqrt(spec)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spec == 0)[0]
        flux_error[where_are_zeros] = 1000000000000
        
        try: 
            sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
            brv=header['ESO DRS BERV']
            wave=get_wave(spec, header)*(1.+brv/2.99792458e5)
        except: 
            sn = hdu[0].header['HIERARCH TNG QC ORDER%s SNR'%order]
            brv = hdu[0].header['HIERARCH TNG QC BERV']
            wave = hdu[5].data*(1.+brv/2.99792458e5)

        try: 
            blaze_file = glob.glob('%s**blaze_A*.fits'%(directory))
            blaze_file = blaze_file[0]
            blaze =fits.open('%s'%blaze_file)
            blaze_func = blaze[0].data
            spec = spec/blaze_func
            flux_error = flux_error/blaze_func
            blaze.close()
        except: print('ERROR: No blaze file found. Continuing without correction')
        
        wavelengths = wave[order]
        fluxes = spec[order]

        last_wavelengths = wave[order-1]
        next_wavelengths = wave[order+1]

        min_overlap = np.max(last_wavelengths)
        max_overlap = np.min(next_wavelengths)

        flux_error_order = flux_error[order]

        hdu.close()

    return fluxes, wavelengths, flux_error_order, sn, np.median(wavelengths) 
