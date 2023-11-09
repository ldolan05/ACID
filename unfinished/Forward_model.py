import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from matplotlib import rc
import pandas as pd
from astropy.io import fits
import glob
from scipy import linalg
import csv

def convolve(p_vel, p_fluxes, linelist, wavelengths):


    '''
    plt.figure(1)
    plt.plot(p_vel,p_fluxes)
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Flux (arbitrary units)')
    plt.savefig('%slineprofile.png'%path)
    '''
    deltav = (np.max(p_vel) - np.min(p_vel))/len(p_vel)

    #######################################################################################################################################################
    # Setting up line list, reads in and filters out wavelengths with depth<0.5, fills other values in with zeros - thought this would make it look nicer #
    #######################################################################################################################################################

    print('Reading in line list...')

    #linelist = np.genfromtxt(linelist, skip_header=6, delimiter=',', usecols=(0,1,9), skip_footer=114)


    lp=pd.read_csv(linelist, delimiter = ',', usecols=['Spec Ion','WL_air(A)','depth'])
    #print(lp.columns)
    l_wavelengths1 = list(lp.loc[:,'WL_air(A)'])
    l_depths1 = list(lp.loc[:,'depth'])
    l_elements1 = list(lp.loc[:,'Spec Ion' ])

    ### secondary linelist
    '''
    linelist_ex= np.genfromtxt('%s_addon.txt'%linelist.replace('.txt',''), skip_header=1, delimiter=',')
    wavelengths_ex =np.array(linelist_ex[:,0])
    depths_ex = np.array(linelist_ex[:,1])
    print(wavelengths_ex, depths_ex)

    print(len(l_wavelengths1))
    for n in range(len(wavelengths_ex)):
        l_wavelengths1.append(wavelengths_ex[n])
        l_depths1.append(depths_ex[n])
        l_elements1.append('nan')
    '''
    ### comment out to here if only 1 linelist is needed.

    print(len(l_wavelengths1))
    #print(l_elements1)

    l_wavelengths1 = np.array(list(l_wavelengths1))
    l_depths1 = np.array(list(l_depths1))
    l_elements1 = np.array(list(l_elements1))
    '''

    l_wavelengths1 =np.array(linelist[:,1])
    l_depths1 = np.array(linelist[:,2])
    l_elements1 = np.array(linelist[:,0])
    '''
    #print(l_elements1)
    print(l_wavelengths1)
    print(len(l_wavelengths1))

    l_depths = []
    l_wavelengths = []
    l_elements = []
    #count = 0


    ##################################################
    # Calculate delta x, reshape and calculate alpha #
    ##################################################
    wavelength_min  = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    for some in range(0, len(l_wavelengths1)):
        if l_wavelengths1[some]>=wavelength_min and l_wavelengths1[some]<=wavelength_max:
            l_wavelengths.append(l_wavelengths1[some])
            l_depths.append(l_depths1[some])
            l_elements.append(l_elements1[some])
        else:
            pass


    '''
    for n in range(0,len(l_depths1)):
        depth = l_depths1[n]
        wavelength = l_wavelengths1[n]
        if depth>0.5:
            l_depths.append(depth)
            #plotdepths.append(depth*(-1))
            l_wavelengths.append(wavelength)
            count = count+1
        else:
            l_depths.append(0)
            #plotdepths.append(0)
            l_wavelengths.append(wavelength)

    d = {'Wavelengths':l_wavelengths, 'Depths': l_depths}
    list_data = pd.DataFrame(data = d)
    '''
    l_blankwaves=wavelengths
    print(len(l_blankwaves))
    #l_blankdepths=np.zeros(len(l_blankwaves))
    '''
    d1 = {'Wavelengths':l_blankwaves, 'Depths':l_blankdepths}
    blank_data = pd.DataFrame(data = d1)


    line_list = pd.concat([list_data,blank_data]).drop_duplicates().reset_index(drop=True)
    line_list = line_list.set_index('Wavelengths')
    line_list = line_list.sort_index(ascending = True)


    #line_list.to_csv('%sline_list.csv'%path)

    #reader=pd.read_csv('%sline_list.csv'%path)
    matrix_waves=linelist[:,0]
    matrix_depths=linelist[:,1]

    l_wavelengths_full=np.array(matrix_waves)
    #print(l_wavelengths)
    l_depths_full=np.array(matrix_depths)
    #print(l_depths)
    #plotdepths=l_depths_full*(-1)


    #line_list.plot()
    #plt.show()

    print('Line list read in and filtered. %s lines remain.'%len(l_depths))

    plt.figure(2)
    plt.vlines(l_wavelengths_full, plotdepths, 0)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Line depths')
    plt.savefig('%slinelist.png'%path)
    '''
    p_vel=velocities

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

    print(np.shape(delta_x))

    #delta_x=np.reshape(delta_x, (len(l_wavelengths), len(l_wavelengths)*len(p_vel)))
    print('Delta_x has been calculated')

    # alpha is calculated by matrix multiplication between delta_x and the depths of each line from the line list.

    print('Calculating Alpha...')

    alpha=[]
    alpha=np.dot(l_depths,delta_x)
    alpha=np.reshape(alpha, (len(l_blankwaves), len(p_vel)))
    print('Alpha Calculated')

    ##################################
    # Calculate delta power spectrum #
    ##################################

    print('Calculating the  Power Spectrum...')
    spectrum=np.dot(alpha, p_fluxes)

    print('Power Spectrum has been calculated')

    '''
    #save the data in another file so it can be easily accessed

    savedata=open("%s5000-5080filteredsynthetic.txt"%path,"w")

    for point in range(len(spectrum)):
        savedata.write("%f\t%f\n" % (l_wavelengths_full[point], spectrum[point]))

    savedata.close()

    plt.figure(3)
    plt.plot(l_wavelengths_full, spectrum)
    plt.xlabel('Wavelength (Angstrom)')
    plt.ylabel('Flux (arbitrary units)')
    plt.savefig('%sspectrum.png'%path)

    # Save plots into folder (designated by path)

    print('Spectrum data and plot saved in %s'%path)
    plt.show()
    '''
    #spectrum = (spectrum/np.max(spectrum))-1
    return l_blankwaves, spectrum, l_wavelengths, l_depths, l_elements

def LSD(wavelengths, flux_obs, rms, linelist):

    vmax=25
    deltav=0.8
    vmin=-vmax

    #resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    #deltav = resol1/(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
    #print(resol1)

    velocities=np.arange(vmin,vmax,deltav)

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

    #print('Delta_x has been calculated')
    continuum_matrix = []
    continuum_waves = []
    for j in range(0, len(blankwaves)):
        row = alpha[j, :]
        row = np.array(row)
        #print(row)
        #print(blankwaves[j])
        non_zeros = row[row>0]
        num_non_zeros = len(non_zeros)
        if num_non_zeros == 0 :
            continuum_waves.append(blankwaves[j])
            continuum_matrix.append(R_matrix[j])

    #print(continuum_waves)
    #print(continuum_matrix)

    plt.figure()
    plt.plot(blankwaves, R_matrix, linewidth = 0.25)
    plt.scatter(continuum_waves, continuum_matrix, color = 'k', s=8)
    plotdepths = [0.5]*len(wavelengths_expected)
    plt.vlines(wavelengths_expected, plotdepths, 1, label = 'line list', alpha = 0.5, linewidth = 0.5)
    plt.show()

    coeffs=np.polyfit(continuum_waves, continuum_matrix, 2)
    poly = np.poly1d(coeffs)
    fit = poly(blankwaves)
    R_matrix_1 = R_matrix
    R_matrix = (R_matrix/fit)-1
    rms = rms/fit

    plt.figure()
    plt.plot(blankwaves, R_matrix_1)
    plt.plot(blankwaves, fit)
    plt.show()
    #print('Calculating Alpha...')
    id_matrix=np.identity(len(flux_obs))
    S_matrix=(1/rms)*id_matrix
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
    return velocities, profile, profile_errors, R_matrix

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
        #div = np.max(fluxes)
        #fluxes = fluxes/div
        #flux_error_order = flux_error_order/div

        #print(flux_error_order[idx])
        #plt.figure()
        #plt.plot(wavelengths, fluxes)

    #fluxes = continuumfit(fluxes, wavelengths, 2)
    return fluxes, wavelengths, flux_error_order, masked_waves, min_overlap, max_overlap ## for just LSD

def continuumfit(fluxes, wavelengths, poly_ord):

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

        coeffs=np.polyfit(wavelengths, fluxes, poly_ord)
        poly = np.poly1d(coeffs)
        fit = poly(full_wavelengths)
        flux_obs = fluxes/fit
        flux_error = errors/fit

        fig = plt.figure('fit')
        plt.plot(wavelengths, fluxes)
        plt.plot(wavelengths, fit)
        #plt.plot(wavelengths, flux_obs)
        plt.scatter(wavelength, fluxe, color = 'k', s=8)
        plt.show()

        return poly

# s1d or e2ds
file_type = 'e2ds'

# order or full(can't fit properly yet as continuum fit goes to zero)
spec_type = 'order'

save_path='/Users/lucydolan/Documents/Least_Squares_Deconvolution/LSD_HARPS_profiles/'

file = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/ADP.2014-09-17T11:19:48.123/HARPS.2007-08-29T00:52:34.128_%s_A.fits'%file_type
ccf_file = '/Users/lucydolan/Documents/CCF_method/HD189733_HARPS_CCFS/August2007/HARPS.2007-08-29T00:52:34.128_ccf_K5_A.fits'
#file = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/ADP.2014-09-17T11:21:39.427/HARPS.2007-08-29T00:19:25.377_e2ds_A.fits'
directory = '/Users/lucydolan/Documents/CCF_method/HD189733/August2007/'

linelist = '/Users/lucydolan/Documents/Least_Squares_Deconvolution/LSD/Archive_stuff/archive/fulllinelist018.txt'
masking = 'masked'
order = 35

#order_range = [26] ##for a single order with no masking/linelist variation
#order_range = np.arange(8,71)
# masked orders
#order_range = [8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 56, 60, 62, 63, 66, 67]
#order_range = [27, 28, 36, 37, 47, 48, 49, 55, 56, 57, 59, 60, 61, 62, 63, 64]
#order_range = reversed(order_range)
count=0

maskings = ['masked', 'unmasked']

linelists=['/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/linelist_varyoriginal.txt',
           #'/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/linelist_varylogg_4-48.txt',
           #'/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/linelist_varylogg_4-68.txt',
           #'/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/linelist_varyT_4772K.txt',
           #'/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/linelist_varyT_5272K.txt'
           ]

#hdr=fits.Header()                                       #making fits header for continuum corrected fits file
#hdu=fits.HDUList()                                      #making fits list for continuum corrected spectrum to go into

all_elements=[]
masked_orders=[]
profiles = []
labels = []

## below are options to compare with different linelists, masks and orders.

for linelist in linelists:
    print(linelist)
    list_name = linelist.replace('/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/linelist_vary', '')
    list_name1 = list_name.replace('.txt', '')
    list_name = list_name1.replace('_', '=')
    list_name = list_name.replace('-', '.')
    '''
    if count == 0:
        name = '- 0.000001'
        count = 1
        colour = 'b'
        marker = 'x'
        line = '-'
    elif count == 1:
        count = 2
        name = '- 0.018'
        colour = 'r'
        marker = '.'
        line = '-.'
    else:
        count = 3
        name = '- 0.1'
        colour = 'g'
        marker = '.'
        line = 'dotted'
    '''
    '''
    for masking in maskings:
        if count == 0:
            name = masking
            count = 1
            colour = 'b'
            marker = 'x'
            line = '-'
        elif count == 1:
            count = 2
            name = masking
            colour = 'r'
            marker = '.'
            line = '-.'
    '''

#for order in order_range:
    ### comment out following section if not using order_range

    #print(order)
    count = 1
    colour = 'r'
    marker = '.'
    line = '-.'
    name = ''

    ###

    #does a rough continuum fit and blaze correction(for e2ds)
    flux, wavelengths, flux_error, masked_waves, min_overlap, max_overlap = blaze_correct(file_type, spec_type, order, file, directory, masking)
    plt.figure()
    plt.plot(wavelengths, flux)
    #hdr['ORDER']=order                     #saving order in fits header
    #hdr['RMS']=rms
    #hdu.append(fits.PrimaryHDU(data=[wavelengths, flux], header = hdr)) #adding spectrum with specified header to fits file

    velocities, profile, profile_errors, flux = LSD(wavelengths, flux, flux_error, linelist)
    '''
    ## read in ccf instead of LSD profile ##
    ccf = fits.open(ccf_file)
    profile = ccf[0].data[order]
    #spectra = np.transpose(ccf_spec)
    ccf_spec = ccf[0].data[order]
    velocities=ccf[0].header['CRVAL1']+(np.arange(ccf_spec.shape[0]))*ccf[0].header['CDELT1']
    profile, errors = continuumfitt(profile, velocities, profile/100, 1)
    '''
    ## plots LSD profile for that order.
    fig1 = plt.figure('Profile - Order: %s, linelist: %s'%(order, list_name))
    plt.title('Order: %s, linelist: %s'%(order, list_name))
    #fig1 = plt.figure('%s to %s'%(wavelength_min, wavelength_max))
    hline = [0]*len(velocities)
    plt.plot(velocities, profile)
    profiles.append(profile)
    labels.append(list_name)
    plt.plot(velocities, hline, linestyle ='--')
    #plt.ylim(-0.3, 0.01)
    #plt.savefig('/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/profile_%s'%list_name1)
    #plt.show()

    m_wavelengths, m_flux, line_waves, line_depths, elements = convolve(velocities, profile, linelist, wavelengths)
    '''

    cropped_wavelengths_copy = []
    for element in wavelengths:
        element = element + element*2.2765/2.99792458e5
        cropped_wavelengths_copy.append(element)

    wavelengths = cropped_wavelengths_copy
    '''
    residuals = flux - m_flux

    '''
    ## plots original spectrum highlighting points above zero
    plt.figure(figsize=(16,9), num = 'Order %s - Continuum fit'%order)
    plt.title('Points above zero - order %s'%order)
    plt.plot(wavelengths, flux,color = 'k', label = 'data')
    for n in range(len(flux)):
        if flux[n]>0:
            plt.plot(wavelengths[n], flux[n], marker = 'x', color = 'r')
    '''

    ## plots the original spectrum as well as the various forward model spectra/spectrum.
    if count == 1:
        fig, ax = plt.subplots(2,figsize=(16,9), gridspec_kw={'height_ratios': [2, 1]}, num = 'Forward Model - Order: %s'%(order), sharex = True)
        plt.title('Order: %s, continuum fit in LSD'%(order))
        ax[0].plot(wavelengths, flux, '--', color = 'orange', label = 'data')
        plotdepths = np.array(line_depths)*(-1)
        ax[0].vlines(line_waves, plotdepths, 0, label = 'line list', alpha = 0.5, linewidth = 0.5)
        #ax[0].axvspan(np.min(wavelengths), min_overlap, alpha=0.5, color='b')
        #ax[0].axvspan(max_overlap, np.max(wavelengths), alpha=0.5, color='b')

        for i in range(len(plotdepths)):
            #count = 0
            if plotdepths[i]<=(-0.5): ## adding text to the deep lines
                ax[0].text(line_waves[i], plotdepths[i]-0.001, elements[i], fontsize = 'xx-small')
    c=0
    for masked_wave in masked_waves:
        if len(masked_wave>=0):
            if c==0:
                c==1
                masked_orders.append(order)
            ax[0].axvspan(np.min(masked_wave), np.max(masked_wave), alpha=0.5, color='red')

    ax[0].plot(m_wavelengths, m_flux, color = colour, linestyle = line, label = 'model %s'%name)
    ax[0].legend()
    hline = [0]*len(wavelengths)
    ax[1].plot(wavelengths, residuals, marker, color = colour)
    ax[1].plot(wavelengths, hline, linestyle = '--')
    ax[1].set_ylim([-0.3, 0.5])

    plotdepths = np.array(line_depths)*(-1)

    for i in range(len(elements)):
        if plotdepths[i]<=(-0.5):
            all_elements.append(elements[i])

    fig.savefig('/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/residuals_%s'%list_name1)
'''
check_elements = ["'Ba 2'", "'Ti 2'", "'Fe 1'", "'Cr 1'", "'Ti 1'", "'Ni 1'", "'Y 1'", "'Eu 2'", "'Co 1'", "'V 1'", "'Ca 1'", "'Ca 2'", "'CN 1'", "'Sr 2'", "'Mn 1'", "'Mg 1'", "'Zr 1'", "'Na 1'"]

for check_element in check_elements:
    occurances = 0
    #print('check element:%s'%check_element)
    for j in range(len(all_elements)):
        if all_elements[j]==check_element:occurances=occurances+1
    else:pass
    if occurances>=1:
        print('%s occurs %s times total'%(check_element, occurances))
    else: print('none')
 '''
#hdu.writeto('/Users/lucydolan/Documents/CCF_method/LSD_figures/Forward Model Comparisons/spectrum_corrected.fits', output_verify='fix', overwrite = 'True')
'''
profile_files = ['/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/metal_minus01.fits',
            '/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/metal_plus01.fits',
            '/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/micro2.fits',
            '/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/micro3.fits']

for prof in profile_files:
    file = fits.open(prof)
    data = file[0].data
    list_name = prof.replace('/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/', '')
    list_name1 = list_name.replace('.txt', '')
    list_name = list_name1.replace('_', '=')
    label = list_name.replace('-', '.')
    label = label.replace('.fits', '')
    #plt.plot(data)
    #print('here')
    profiles.append(data)
    labels.append(label)

'''
plt.figure('Profile - Order: %s'%(order))
plt.title('Order: %s - all linelists'%(order))
hline = [0]*len(velocities)
for n in range(len(profiles)):
    plt.plot(velocities, profiles[n], label = labels[n])

plt.plot(velocities, hline, linestyle ='--')
#plt.ylim(-0.4, 0.01)
plt.legend()
plt.savefig('/Users/lucydolan/Documents/CCF_method/HD189733/Linelists/all_linelists.png')


plt.show()

#print(masked_orders)
