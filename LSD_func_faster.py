import numpy as np
from scipy import linalg
from astropy.io import  fits
import glob
import matplotlib.pyplot as plt
import random

def LSD(wavelengths, flux_obs, rms, linelist, adjust_continuum, poly_ord, sn, order, run_name):

    #converting to optical depth
    idx = tuple([flux_obs!=0])
    flux_obs = np.log(flux_obs[idx])
    rms = rms[idx]/flux_obs

    wavelengths = wavelengths[idx]

    plt.figure()
    plt.plot(wavelengths, flux_obs)
    plt.show()

    width = 40
    centre = -2.1
    #deltav=0.8
    #vmin=-vmax

    resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    deltav = resol1/(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5

    shift = int(centre/deltav)
    centre1 = shift*deltav

    vmin = int(centre1-(width/2))
    vmax = int(centre1+(width/2))
    no_pix = int(width/deltav)

    velocities=np.linspace(vmin,vmax,no_pix)

    #print(vgrid[1]-vgrid[0])
    #print('Matrix S has been set up')

    #### This is the EXPECTED linelist (for a slow rotator of the same spectral type) ####
    linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
    wavelengths_expected1 =np.array(linelist_expected[:,0])
    depths_expected1 = np.array(linelist_expected[:,1])

    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    wavelengths_expected=[]
    depths_expected=[]
    for some in range(0, len(wavelengths_expected1)):
        line_min = 1/(3*sn)
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max and depths_expected1[some]>=line_min:
            wavelengths_expected.append(wavelengths_expected1[some])
            #depths_expected.append(depths_expected1[some]+random.uniform(-0.1, 0.1))
            depths_expected.append(depths_expected1[some])
        else:
            pass

    ## depths from linelist in optical depth space
    depths_expected = np.array(depths_expected)
    depths_expected = np.log(1+depths_expected)

    plt.figure()
    plotdepths = -np.array(depths_expected)
    plt.vlines(wavelengths_expected, plotdepths, 0, label = 'line list', color = 'c', alpha = 0.5)
    plt.show()

    blankwaves=wavelengths
    R_matrix=flux_obs

    alpha=np.zeros((len(blankwaves), len(velocities)))

    limit=max(abs(velocities))*max(wavelengths_expected)/2.99792458e5
    #print(limit)

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):

            diff=blankwaves[j]-wavelengths_expected[i]
            #limit=np.max(velocities)*wavelengths_expected[i]/2.99792458e5
            if abs(diff)<=(limit):
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

    ### FITTING CONTINUUM OF SPECTRUM ###
    if adjust_continuum == 'True':
        ## Identifies the continuum points as those corresponding to a row of zeros in the alpha matrix (a row of zeros implies zero contribution from all lines in the line list)
        continuum_matrix = []
        continuum_waves = []
        for j in range(0, len(blankwaves)):
            row = alpha[j, :]
            row = np.array(row)
            edge_size = int(np.floor(len(row)/8))
            #print(edge_size)
            row = row[edge_size:len(row)-edge_size]
            weight = sum(row)
            #print(row)
            if weight == 0:
                continuum_waves.append(blankwaves[j])
                continuum_matrix.append(R_matrix[j]+1)

        if len(continuum_waves)<3:R_matrix = R_matrix
        else:
            ## Plotting the continuum points on top of original spectrum - highlights any lines missing from linelist ##

            plt.figure()
            plt.plot(blankwaves, R_matrix+1, linewidth = 0.25)
            plt.scatter(continuum_waves, continuum_matrix, color = 'k', s=8)
            #plotdepths = [0.5]*len(wavelengths_expected)
            #plt.vlines(wavelengths_expected, plotdepths, np.max(continuum_matrix), label = 'line list', alpha = 0.5, linewidth = 0.5)
            plt.show()


            ## Fits a second order(although usually defaults to first order) polynomial to continuum points and divides original spectrum by the fit.
            coeffs=np.polyfit(continuum_waves, continuum_matrix, poly_ord)
            poly = np.poly1d(coeffs)
            fit = poly(blankwaves)
            R_matrix_1 = R_matrix+1
            R_matrix = (R_matrix_1/fit)-1
            rms = rms/fit

            ## Plotting the original spectrum with the fit.

            plt.figure()
            plt.plot(blankwaves, R_matrix_1)
            plt.plot(blankwaves, fit)
            plt.show()

    else:
        continuum_waves = []
        continuum_matrix = []

            ### Continuum fitting done - feeds corrected R_matrix and errors (denoted rms, but not actually the rms) back into LSD.

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

    return velocities, profile, profile_errors, alpha, wavelengths_expected, depths_expected

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
       #for x in range(npix):
       #  wave[o,x]=wave[o,x]+par*xx[i,x]#float(x)**float(i)

  return wave

def blaze_correct(file_type, spec_type, order, file, directory, masking, run_name):
    #### Inputing spectrum depending on file_type and spec_type #####

    if file_type == 's1d':
        #### Finding min and max wavelength from e2ds for the specified order ######
        file_e2ds = file.replace('s1d', 'e2ds')
        print(file_e2ds)
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
        #wavelength_min = np.min(wavelengths_order)
        #wavelength_max = np.max(wavelengths_order)
        wavelength_min = 5900
        wavelength_max = wavelength_min+200    ###### if you want to do a WAVELENGTH RANGE just input min and max here ######
        #print(wavelength_max)
        hdu.close()
        #### Now reading in s1d file ########
        print(file)
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

            if masking == 'masked':
                ## I've just been updating as I go through so it's not complete
                #if you want to add to it the just add an element of form: [min wavelength of masked region, max wavelength of masked region]
                masks_csv = np.genfromtxt('/home/lsd/Documents/HD189733b_masks.csv', delimiter=',')
                min_waves_mask = np.array(masks_csv[:,0])
                max_waves_mask = np.array(masks_csv[:,1])
                masks = []
                for mask_no in range(len(min_waves_mask)):
                    masks.append([min_waves_mask[mask_no], max_waves_mask[mask_no]])

                masked_waves=[]

                for mask in masks:
                    #print(np.max(mask), np.min(mask))
                    idx = np.logical_and(wavelengths>=np.min(mask), wavelengths<=np.max(mask))
                    #print(flux_error_order[idx])
                    flux_error_order[idx] = 10000000000000000000
                    #print(idx)
                    if len(wavelengths[idx])>0:
                        masked_waves.append(wavelengths[idx])

                #masks = []
                ### allows extra masking to be added ##

                plt.figure('masking')
                plt.plot(wavelengths, fluxes)
                if len(masked_waves)>0:
                    for masked_wave in masked_waves:
                        plt.axvspan(np.min(masked_wave), np.max(masked_wave), alpha=0.5, color='red')
                #plt.show()

                #response = input('Are there any regions to be masked? y or n: ')
                response = 'y'
                if response == 'y':
                    '''
                    print('Take note of regions.')
                    plt.figure('masking')
                    plt.plot(wavelengths, fluxes)
                    plt.show()
                    '''
                    #response1 = int(input('How many regions to mask?: '))
                    response1 = 0
                    for i in range(response1):
                        min_wave = float(input('Minimum wavelength of region %s: '%i))
                        max_wave = float(input('Maximum wavelength of region %s: '%i))
                        masks.append([min_wave, max_wave])
                    masked_waves=[]
                    #print(masks)
                    for mask in masks:
                        print(np.max(mask), np.min(mask))
                        idx = np.logical_and(wavelengths>=np.min(mask), wavelengths<=np.max(mask))
                        #print(flux_error_order[idx])
                        flux_error_order[idx] = 10000000000000000000

                        if len(wavelengths[idx])>0:
                            masked_waves.append(wavelengths[idx])

                    plt.figure('masking')
                    plt.plot(wavelengths, fluxes)
                    print(masked_waves)
                    for masked_wave in masked_waves:
                        plt.axvspan(np.min(masked_wave), np.max(masked_wave), alpha=0.5, color='red')
                    #print('new version')
                    plt.savefig('/home/lsd/Documents/LSD_Figures/masking_plots/order%s_masks_%s'%(order, run_name))

                    plt.figure('errors')
                    plt.plot(wavelengths, flux_error_order)
                    plt.close('all')
                    #plt.show()

                if response == 'n':
                    print('yay!')


            elif masking == 'unmasked':
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
        sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        spec_check = spec[spec<=0]
        if len(spec_check)>0:
            print('WARNING NEGATIVE/ZERO FLUX - corrected')

        flux_error = np.sqrt(spec)
        where_are_NaNs = np.isnan(flux_error)
        flux_error[where_are_NaNs] = 1000000000000
        where_are_zeros = np.where(spec == 0)[0]
        flux_error[where_are_zeros] = 1000000000000
        '''
        flux_error1 = header['HIERARCH ESO DRS SPE EXT SN%s'%order]

        flux_error = header['HIERARCH ESO DRS CAL TH RMS ORDER%s'%order]
        print(flux_error, flux_error1)

        flux_error = flux_error*np.ones(np.shape(spec))
        '''
        brv=header['ESO DRS BERV']
        wave=get_wave(spec, header)*(1.+brv/2.99792458e5)


        blaze_file = glob.glob('%sblaze_folder/**blaze_A*.fits'%(directory))
        print('%sblaze_folder/**blaze_A*.fits'%(directory))
        print(blaze_file)
        blaze_file = blaze_file[0]
        blaze =fits.open('%s'%blaze_file)
        blaze_func = blaze[0].data
        spec = spec/blaze_func
        flux_error = flux_error/blaze_func
        blaze.close()

        wavelengths = wave[order]
        fluxes = spec[order]

        last_wavelengths = wave[order-1]
        next_wavelengths = wave[order+1]

        min_overlap = np.max(last_wavelengths)
        max_overlap = np.min(next_wavelengths)

        flux_error_order = flux_error[order]

        if masking == 'masked':
            ## I've just been updating as I go through so it's not complete
            #if you want to add to it the just add an element of form: [min wavelength of masked region, max wavelength of masked region]
            masks_csv = np.genfromtxt('/home/lsd/Documents/HD189733b_masks.csv', delimiter=',')
            min_waves_mask = np.array(masks_csv[:,0])
            max_waves_mask = np.array(masks_csv[:,1])
            masks = []
            for mask_no in range(len(min_waves_mask)):
                masks.append([min_waves_mask[mask_no], max_waves_mask[mask_no]])

            masked_waves=[]

            for mask in masks:
                #print(np.max(mask), np.min(mask))
                idx = np.logical_and(wavelengths>=np.min(mask), wavelengths<=np.max(mask))
                #print(flux_error_order[idx])
                flux_error_order[idx] = 10000000000000000000
                #print(idx)
                if len(wavelengths[idx])>0:
                    masked_waves.append(wavelengths[idx])

            #masks = []
            ### allows extra masking to be added ##

            plt.figure('masking')
            plt.plot(wavelengths, fluxes)
            if len(masked_waves)>0:
                for masked_wave in masked_waves:
                    plt.axvspan(np.min(masked_wave), np.max(masked_wave), alpha=0.5, color='red')
            #plt.show()

            #response = input('Are there any regions to be masked? y or n: ')
            response = 'y'
            if response == 'y':
                '''
                print('Take note of regions.')
                plt.figure('masking')
                plt.plot(wavelengths, fluxes)
                plt.show()
                '''
                #response1 = int(input('How many regions to mask?: '))
                response1 = 0
                for i in range(response1):
                    min_wave = float(input('Minimum wavelength of region %s: '%i))
                    max_wave = float(input('Maximum wavelength of region %s: '%i))
                    masks.append([min_wave, max_wave])
                masked_waves=[]
                #print(masks)
                for mask in masks:
                    print(np.max(mask), np.min(mask))
                    idx = np.logical_and(wavelengths>=np.min(mask), wavelengths<=np.max(mask))
                    #print(flux_error_order[idx])
                    flux_error_order[idx] = 10000000000000000000

                    if len(wavelengths[idx])>0:
                        masked_waves.append(wavelengths[idx])

                plt.figure('masking')
                plt.plot(wavelengths, fluxes)
                print(masked_waves)
                for masked_wave in masked_waves:
                    plt.axvspan(np.min(masked_wave), np.max(masked_wave), alpha=0.5, color='red')
                #print('new version')
                plt.savefig('/home/lsd/Documents/LSD_Figures/masking_plots/order%s_masks_%s'%(order, run_name))

                plt.figure('errors')
                plt.plot(wavelengths, flux_error_order)
                plt.close('all')
                #plt.show()

            if response == 'n':
                print('yay!')


        elif masking == 'unmasked':
            masked_waves = []
            masked_waves = np.array(masked_waves)

        else: print('WARNING - masking not catching - must be either "masked" or "unmasked"')

        hdu.close()


    flux_error_order = (flux_error_order)/(np.max(fluxes)-np.min(fluxes))
    print('flux error: %s'%flux_error_order)
    fluxes = (fluxes - np.min(fluxes))/(np.max(fluxes)-np.min(fluxes))

    idx = tuple([fluxes!=0])

    return fluxes[idx], wavelengths[idx], flux_error_order[idx], sn, np.median(wavelengths) ## for just LSD
############################################################################################################
