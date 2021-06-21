import numpy as np
from scipy import linalg
from astropy.io import  fits
import glob
import matplotlib.pyplot as plt

def LSD(wavelengths, flux_obs, rms, linelist):

    vmax=25
    #deltav=1.1
    vmin=-vmax

    resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    deltav = resol1/(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
    #print(resol1)

    velocities=np.arange(vmin,vmax,deltav)

    id_matrix=np.identity(len(flux_obs))
    S_matrix=(1/rms)*id_matrix
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
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max:
            wavelengths_expected.append(wavelengths_expected1[some])
            depths_expected.append(depths_expected1[some])
        else:
            pass

    #print('number of lines: %s'%len(depths_expected))
    #print('Expected linelist has been read in')

    blankwaves=wavelengths
    R_matrix=flux_obs
    #print('Matrix R has been set up')

    #delta_x=np.zeros([len(wavelengths_expected), (len(blankwaves)*len(velocities))])
    #print('Delta x')
    #print(np.shape(delta_x))

    alpha=np.zeros((len(blankwaves), len(velocities)))

    limit=np.max(velocities)*np.max(wavelengths_expected)/2.99792458e5

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):

            diff=blankwaves[j]-wavelengths_expected[i]
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
    '''
    #print('Delta_x has been calculated')
    continuum_matrix = []
    continuum_waves = []
    for j in range(0, len(blankwaves)):
        row = alpha[j, :]
        num_non_zeros = np.count_nonzero(row)
        if num_non_zeros == 0 :
            continuum_waves.append(blankwaves[j])
            continuum_matrix.append(R_matrix[j])

    plt.figure()
    plt.plot(blankwaves, R_matrix)
    plt.scatter(continuum_waves, continuum_matrix, 'k')
    plt.show()
    '''
    #print('Calculating Alpha...')

    #alpha=np.dot(depths_expected,delta_x)
    #alpha=np.reshape(alpha, (len(blankwaves), len(velocities)))

    #print('Alpha Calculated')

    S_squared=np.dot(S_matrix, S_matrix)
    alpha_transpose=(np.transpose(alpha))

    #print('Beginning deconvolution')
    RHS_1=np.dot(alpha_transpose, S_squared)
    RHS_final=np.dot(RHS_1, R_matrix )

    #print('RHS ready')

    LHS_preinvert=np.dot(RHS_1, alpha)
    LHS_prep=np.matrix(LHS_preinvert)

    #print('Beginning inversion')
    P,L,U=linalg.lu(LHS_prep)

    n=len(LHS_prep)
    B=np.identity(n)
    Z = linalg.solve_triangular(L, B, lower=True)
    X = linalg.solve_triangular(U, Z, lower=False)
    LHS_final = np.matmul(X,np.transpose(P))

    #print('Inversion complete')

    profile=np.dot(LHS_final, RHS_final)
    profile_errors_squared=np.diagonal(LHS_final)
    profile_errors=np.sqrt(profile_errors_squared)
    '''
    upper_errors = profile+profile_errors
    lower_errors = profile-profile_errors


    fig3 = plt.figure(3)
    plt.plot(velocities, profile, color = 'b')
    plt.fill_between(velocities, lower_errors, upper_errors, alpha=0.4)
    plt.xlabel('Velocity(km/s)')
    plt.ylabel('Flux(Arbitrary Units)')
    #fig3.savefig('%s.png'%spectrum)

    #stop = timeit.default_timer() #end point of the timer
    #print('Time: ', stop - start)

    plt.show()
    '''
    return velocities, profile, profile_errors

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

def blaze_correct(file_type, spec_type, order, file, directory, masking):
    #### Inputing spectrum depending on file_type and spec_type #####

    if file_type == 's1d':
        #### Finding min and max wavelength from e2ds for the specified order ######
        file_e2ds = file.replace('s1d', 'e2ds')
        print(file_e2ds)
        hdu=fits.open('%s'%file_e2ds)
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
        #wavelength_min = ...
        #wavelength_max = wavelength_min+200    ###### if you want to do a WAVELENGTH RANGE just input min and max here ######
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

            if masking =='masked':print('WARNING - masking not set up for s1d')

        elif spec_type == 'full':
            ## not set up properly.
            wavelengths = wave
            fluxes = spec

    elif file_type == 'e2ds':
        hdu=fits.open('%s'%file)
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
        '''
        flux_error1 = header['HIERARCH ESO DRS SPE EXT SN%s'%order]

        flux_error = header['HIERARCH ESO DRS CAL TH RMS ORDER%s'%order]
        print(flux_error, flux_error1)

        flux_error = flux_error*np.ones(np.shape(spec))
        '''
        brv=header['ESO DRS BERV']
        wave=get_wave(spec, header)*(1.+brv/2.99792458e5)


        blaze_file = glob.glob('%sblaze_folder/**blaze_A*.fits'%(directory))
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
            masks = [[4955, 4960], [4933.5, 4934.5], [6496, 6497.5], [3981, 3982.5], [3988.5, 3990.5]
                     ,[4287.5, 4290.5], [4293.5, 4294.5], [4299.5,4303], [4336, 4339], [4341, 4344.5], [4381, 4386], [4402, 4407]
                     ,[4417.3,4418], [4443.4,4444.2], [4501,4501.5], [4533.5, 4534.4], [4553.6, 4554.4], [4563.4, 4564.2], [4571.6, 4572.4]
                     ,[4860.5, 4862], [4855.1, 4855.6], [4828.9, 4829.2], [4890.3,4892.49], [4903.9, 4904.9], [4913.7, 4914.4], [5013.7, 5014.7]
                     ,[5039.5,5040.4], [5080.15, 5081.41], [5126.64, 5126.98], [5136.84, 5137.2], [5146.27, 5146.48], [5166.19, 5187.44], [5141, 5143]
                     ,[5231.81, 5233.96], [5352.89, 5354.04], [5369.6, 5372.9], [5444.6, 5445.6], [4003, 4008], [4043, 4048], [4069, 4073], [4091, 4093]
                     ,[4141, 4145], [4224, 4229], [5475, 5478], [5885, 5900], [6140, 6143], [6255, 6259], [6313, 6316], [6558, 6565], [6121,6124.5], [6159, 6172], [6100, 6104] ]
            masked_waves=[]

            for mask in masks:
                #print(np.max(mask), np.min(mask))
                idx = np.logical_and(wavelengths>=np.min(mask), wavelengths<=np.max(mask))
                #print(flux_error_order[idx])
                flux_error_order[idx] = 10000000000000000000

                masked_waves.append(wavelengths[idx])

        elif masking == 'unmasked':
            masked_waves = []
            masked_waves = np.array(masked_waves)

        else: print('WARNING - masking not catching - must be either "masked" or "unmasked"')

        hdu.close()
        div = np.max(fluxes)
        fluxes = fluxes/div
        flux_error_order = flux_error_order/div

        #print(flux_error_order[idx])
        #plt.figure()
        #plt.plot(wavelengths, fluxes)

    #fluxes = continuumfit(fluxes, wavelengths, 2)

    return fluxes, wavelengths, flux_error_order

def continuumfit(fluxes, wavelengths, errors, poly_ord):
        fluxes=fluxes
        idx = wavelengths.argsort()
        wavelength = wavelengths[idx]
        fluxe = fluxes[idx]

        frac = 0.75
        sigma = 1.5*np.median(abs(wavelengths-np.median(wavelengths)))
        sigma_lower = np.min(wavelengths)+frac*sigma
        sigma_upper = np.max(wavelengths)-frac*sigma

        #print(sigma_lower, sigma_upper)

        idx = wavelengths.argsort()

        wavelength = wavelengths[idx]
        flux = fluxe[idx]


        wavelength1 = np.array(wavelength[wavelength<sigma_lower])
        #print(len(wavelength1))
        flux1 = np.array(flux[:len(wavelength1)])
        #print(len(flux1))
        wavelength2 = np.array(wavelength[wavelength>sigma_upper])
        #print(len(wavelength2))
        flux2 = np.array(flux[-len(wavelength2):])
        #print(len(flux2))

        wavelength = np.concatenate((wavelength1, wavelength2))
        fluxe = np.concatenate((flux1, flux2))

        coeffs=np.polyfit(wavelength, fluxe, poly_ord)
        poly = np.poly1d(coeffs)
        fit = poly(wavelengths)
        flux_obs = fluxes/fit
        flux_error = errors/fit

        fig = plt.figure('fit')
        plt.plot(wavelengths, fluxes)
        plt.plot(wavelengths, fit)
        #plt.plot(wavelengths, flux_obs)
        plt.scatter(wavelength, fluxe, color = 'k', s=8)
        plt.show()

        return flux_obs, flux_error, poly
############################################################################################################
file = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_e2ds_A.fits'
directory = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/'
linelist = '/Users/lucydolan/Documents/Least_Squares_Deconvolution/LSD/Archive_stuff/archive/fulllinelist018.txt'
# s1d or e2ds
file_type = 'e2ds'

# order or full(can't fit properly yet as continuum fit goes to zero)
spec_type = 'order'
order = 28
masking = 'masked'

fluxes, wavelengths, flux_error = blaze_correct(file_type, spec_type, order, file, directory, masking)
velocities, profile, profile_errors = LSD(wavelengths, fluxes, flux_error, linelist)

p, v, poly = continuumfit(profile, velocities, profile_errors, 1)

fit = poly(wavelengths)
fit_plot = poly(velocities)

plt.figure('Before Correction')
plt.plot(wavelengths, fluxes)
plt.plot(wavelengths, fit)
plt.xlabel('wavelength')
plt.ylabel('flux')

plt.figure('LSD Profile(without continuum correction): Order %s'%order)
plt.plot(velocities, profile)
plt.plot(velocities, fit_plot)
plt.xlabel('velocities')
plt.ylabel('flux')
plt.show()

fluxes = (fit*fluxes)
flux_error = fit*flux_error

plt.figure('After Correction')
plt.plot(wavelengths, fluxes)
plt.xlabel('wavelength')
plt.ylabel('flux')

velocities, profile, profile_errors = LSD(wavelengths, fluxes, flux_error, linelist)

plt.figure('Final LSD Profile: Order %s'%order)
plt.plot(velocities, profile)
plt.xlabel('velocities')
plt.ylabel('flux')
plt.show()
