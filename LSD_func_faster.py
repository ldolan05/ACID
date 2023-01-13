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
import itertools

def cvmt(wavelengths_o, vel, vlambda,vdepth):
    c = 2.99792458e5
    dv = vel[1]-vel[0]

    lenk = 3
    
    row2 = np.array([])
    column2 = np.array([])
    value2 = np.array([])
    
    b = len(wavelengths_o)//lenk
    
    for k in np.arange(lenk):
        #wvls in this chunk
        wavelengths_sub = wavelengths_o[b*k:b*(k+1)]
        # print(wavelengths_sub)
        #min wvl and max wvl to include
        lmin = wavelengths_sub.min()*(1.-(vel.max()+dv)/c)
        lmax = wavelengths_sub.max()*(1.-(vel.min()-dv)/c)
        
        #only include vald abslines that are withinn lmin and lmax
        idx1 = np.logical_and((vlambda>lmin), (vlambda<lmax))
        # print(idx1)
        # print(vlambda.shape)
        vlambdaincl = vlambda[idx1]
        vdepthincl = vdepth[idx1]
        
        #compute velocity shift between vald lines and wvls
        v_shift = c*(np.tile(wavelengths_sub,(len(vlambdaincl),1)) / (np.tile(vlambdaincl,(len(wavelengths_sub),1))).transpose()-1.)

        #only keep those with velocity shift within velocity grid
        # l = index of vald absorption line
        # i = index of affected flux/wvl
        l,i = np.where((v_shift<=(vel.max()+dv)) & (v_shift>=(vel.min()-dv)))
        value0 = v_shift[l,i]
        
        #index shift b*k due to wvl-chunk-splitting for this look
        i += b*k

        #velocity position between two points of velocity grid
        pos = ((value0-vel[0])%dv)/dv
        #left velocity grid point
        jleft = (value0-vel[0])//dv
        
        #check which ones falll within velocity grid
        woleft = (jleft>=0) & (jleft<len(vel))
        woright = (jleft>=-1) & (jleft<len(vel)-1)
        
        #stack information
        i1 = np.hstack((i[woleft],i[woright]))
        j1 = np.hstack((jleft[woleft],jleft[woright]+1))

        #contribution to flux from each vald absorption line
        vstl1 = ((1.-np.hstack((pos[woleft],1.-pos[woright])))*vdepthincl[np.hstack((l[woleft],l[woright]))])
        
        sorter = i1+j1/len(vel)/10.
        indexorder = np.argsort(sorter)
        i1 = i1[indexorder]
        j1 = j1[indexorder]
        vstl1 = vstl1[indexorder]
        sorter = sorter[indexorder]
        uniquejs,whereuniquejindex,samejs = np.unique(sorter,return_inverse=True,return_index=True)

        row2 = np.hstack((row2,i1[whereuniquejindex]))
        value2 = np.hstack((value2,np.bincount(samejs,vstl1)))
        column2 = np.hstack((column2,j1[whereuniquejindex]))
    return value2,row2,column2

def runLSD_inv(values, row_no, col_no, n, m, weights, fluxes):
    # MATRIX CALCULATIONS USING 'csc_matrix' TO IMPROVE EFFICIENCY
    # The uncertainties are computed here (necessitates inverting the MtWM matrix

    M = csc_matrix((values, (row_no, col_no)), shape=(n, m))
    Mt = M.transpose()
    MtW = Mt.dot(csc_matrix((weights, (np.arange(n), np.arange(n))), shape=(n, n)))
    MtWM = MtW.dot(M)

    A = MtWM

    B = MtW.dot(csc_matrix(fluxes).transpose())
    #Z = spsolve(A, B, use_umfpack=True)
    inverse = inv(csc_matrix(MtWM))
    Z = inverse.dot(B)
    return np.asarray(Z.todense()).ravel(), np.sqrt(np.diag(inverse.todense()))
    
def LSD(wavelengths, flux_obs, rms, linelist, adjust_continuum, poly_ord, sn, order, run_name, velocities):


    #idx = tuple([flux_obs>0])
    # in optical depth space
    rms = rms/flux_obs
    flux_obs = np.log(flux_obs)

    # plt.figure()
    # plt.plot(wavelengths, flux_obs)
    # plt.show()
    # flux_obs = flux_obs -1
    #wavelengths = wavelengths[idx]

    # width = 20
    # centre = -2.1

    # #vmin=-vmax
    
    # resol1 = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
    # deltav = resol1/(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
    # print(deltav)

    # resol1 = deltav*(wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2))*2.99792458e5
    # shift = int(centre/deltav)
    # centre1 = shift*deltav

    # vmin = int(centre1-(width/2))
    # vmax = int(centre1+(width/2))
    
    # # pix_size = 0.82
    # # vmin = -15
    # # vmax = 10
    # print(vmin, vmax, deltav)

    # velocities=np.linspace(vmin,vmax,no_pix)
    # velocities=np.arange(vmin,vmax,pix_size)
    # print(velocities[1]-velocities[0])
    deltav = velocities[1]-velocities[0]
    #print(deltav)
    #print(vgrid[1]-vgrid[0])
    #print('Matrix S has been set up')

    #### This is the EXPECTED linelist (for a slow rotator of the same spectral type) ####
    counter = 0
    for line in reversed(open(linelist).readlines()):
        if len(line.split(",")) > 10:
            break
        counter += 1
    num_lines = sum(1 for line in open(linelist))
    last_line = num_lines - counter + 3

    with open(linelist) as f_in:
        linelist_expected = np.genfromtxt(itertools.islice(f_in, 3, last_line, 4), dtype=str,delimiter=',')

    #linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
    wavelengths_expected1 =np.array(linelist_expected[:,1], dtype = 'float')
    depths_expected1 = np.array(linelist_expected[:,9], dtype = 'float')
    if np.max(depths_expected1)>1:
        depths_expected1 = np.array(linelist_expected[:,13], dtype = 'float')
    # print(len(depths_expected1))

    wavelength_min = np.min(wavelengths)
    wavelength_max = np.max(wavelengths)

    wavelengths_expected=[]
    depths_expected=[]
    no_line =[]
    for some in range(0, len(wavelengths_expected1)):
        line_min = 1/(3*sn)
        #print(line_min)
        # line_min = 0.25
        #line_min = np.log(1+line_min)
        #print(line_)
        #line_min = np.log(1+line_min)
        #print(line_min)
        if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max and depths_expected1[some]>=line_min:
            wavelengths_expected.append(wavelengths_expected1[some])
            #depths_expected.append(depths_expected1[some]+random.uniform(-0.1, 0.1))
            depths_expected.append(depths_expected1[some])
        else:
            pass

    # # ### TEST SECTION ####
    # count_range = np.array([len(wavelengths_expected)]*10)*np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    # count_range = np.array(count_range, dtype = int)

    # wavelengths_expected1 = wavelengths_expected
    # depths_expected1 = depths_expected
    # depths_expected = []
    # wavelengths_expected = []
    # for line in count_range:
    #     wavelengths_expected.append(wavelengths_expected1[line])
    #     depths_expected.append(depths_expected1[line])

    ######## END OF TEST SECTION ########

    ## depths from linelist in optical depth space
    depths_expected1 = np.array(depths_expected)
    #print(depths_expected)
    depths_expected = np.log(1+depths_expected1)
    # print(depths_expected)
    ## conversion for depths from SME
    #depths_expected = -np.log(1-depths_expected1)

    # print(len(depths_expected))

    blankwaves=wavelengths
    R_matrix=flux_obs

    alpha=np.zeros((len(blankwaves), len(velocities)))

    #limit=max(abs(velocities))*max(wavelengths_expected)/2.99792458e5
    # print(limit)

    for j in range(0, len(blankwaves)):
        for i in (range(0,len(wavelengths_expected))):
            vdiff = ((blankwaves[j] - wavelengths_expected[i])*2.99792458e5)/wavelengths_expected[i]
            # limit_up = (np.max(velocities)+deltav)*(wavelengths_expected[i]/2.99792458e5)
            # print(limit_up)
            # limit_down = (np.min(velocities)-deltav)*(wavelengths_expected[i]/2.99792458e5)
            # print(limit_down)
            if vdiff<=(np.max(velocities)+deltav) and vdiff>=(np.min(velocities)-deltav):
                diff=blankwaves[j]-wavelengths_expected[i]

                # id = np.logical_and(blankwaves<wavelengths_expected[i]+limit_up, blankwaves>wavelengths_expected[i]+limit_down)
                # print(id)
                # w = ((blankwaves - wavelengths_expected[i])*2.99792458e5)/wavelengths_expected[i]
                # p = flux_obs

                # print(blankwaves[id], flux_obs[id])
            
                # id2 = np.logical_and(blankwaves<wavelengths_expected[i]+limit_up+1, blankwaves>wavelengths_expected[i]+limit_down-1)
                if rms[j]<1:no_line.append(i)
                vel=2.99792458e5*(diff/wavelengths_expected[i])
                # plt.figure()
                # plt.errorbar(blankwaves[id2], flux_obs[id2], rms[id2], color = 'k')
                # plt.scatter(blankwaves[j], flux_obs[j], label = 'wavelength pixel')
                # plt.scatter(wavelengths_expected[i], flux_obs[j], label = 'linelist wavelegngth')
                # plt.plot(blankwaves[id], flux_obs[id], label = 'included area for line')
                for k in range(0, len(velocities)):
                    # if blankwaves[j]==blankwaves[0]:
                    #     dv = ((blankwaves[j+1] - blankwaves[j])*2.99792458e5)/blankwaves[j]
                    # else:
                    #     dv = ((blankwaves[j] - blankwaves[j-1])*2.99792458e5)/blankwaves[j-1]
                    x=abs((velocities[k]-vel)/deltav)
                    if x<1.:
                        delta_x=(1-x)
                        alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
                    # if -1.<x and x<0.:
                    #     delta_x=(1-x)
                    #     alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x
                    # elif 0.<=x and x<1.:
                    #     delta_x=(1-x)
                    #     alpha[j, k] = alpha[j, k]+depths_expected[i]*delta_x

                # print(alpha[j, :].shape)
                # print(w)
                # idx = tuple([alpha[j, :]!=0.])
                # idx2 = tuple([alpha[j, :]==0.])
                # w2 = 2.99792458e5*wavelengths_expected[i]/(2.99792458e5-velocities)
                # plt.scatter(w2[idx2], np.zeros(velocities[idx2].shape), label = 'velocitiy grid and delta function = 0')
                # plt.scatter(w2[idx], np.zeros(velocities[idx].shape), label = 'velocitiy grid and delta function !=0 ')
                # plt.legend()
                # plt.show()
            else:
                pass
    
    # value, row, column = cvmt(blankwaves, velocities, np.array(wavelengths_expected), np.array(depths_expected))
    # M = csc_matrix((value, (row, column)), shape=(len(blankwaves), len(velocities)))
    # M=np.array(M)
    # print(M.type)
    # print(alpha.type)
    # if np.sum(M-alpha)!=0.:
    #     print(np.sum(M-alpha))
    #     print('alpha and M do not agree')
    #     print('using M...')
    #     alpha = M.toarray()
    #     alpha = np.array(alpha)
    # else:print('THEY AGREE')

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

    # profile, profile_errors = runLSD_inv(value, row, column, len(blankwaves), len(velocities), 1/(rms**2), flux_obs)

    return velocities, profile, profile_errors, alpha, wavelengths_expected, depths_expected1, len(no_line)

def get_wave(data,header):

  wave=data*0.
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
          try:par=header['ESO DRS CAL TH COEFF LL%d' % idx]
          except:par=header['TNG DRS CAL TH COEFF LL%d' % idx]
          wave[o,:]=wave[o,:]+par*xx[i,:]
       #for x in range(npix):
       #  wave[o,x]=wave[o,x]+par*xx[i,x]#float(x)**float(i)

  return wave

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
        # plt.figure()
        # plt.plot(wavelengths1, fluxes1)
        # plt.plot(wavelengths1, fit)
    
        flux_obs = fluxes1/fit
        
        return flux_obs

# from MM-LSD code - give credit if needed
def upper_envelope(x, y):
    #used to compute the tapas continuum. find peaks then fit spline to it.
    peaks = find_peaks(y, height=0.2, distance=len(x) // 500)[0]
    # t= knot positions
    spl = LSQUnivariateSpline(x=x[peaks], y=y[peaks], t=x[peaks][5::10])
    return spl(x)

def blaze_correct(file_type, spec_type, order, file, directory, masking, run_name, berv_opt):
    #### Inputing spectrum depending on file_type and spec_type #####

    if file_type == 's1d':
        #### Finding min and max wavelength from e2ds for the specified order ######
        file_e2ds = file.replace('s1d', 'e2ds')
        print(file_e2ds)
        hdu=fits.open('%s'%file_e2ds)
        try:sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        except:sn = hdu[0].header['HIERARCH TNG DRS SPE EXT SN%s'%order]
        spec=hdu[0].data
        header=hdu[0].header
        try:brv=header['ESO DRS BERV']
        except:brv=header['TNG DRS BERV']
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

        # print(min_overlap)
        # print(max_overlap)
        # plt.figure()
        # plt.plot(wave[order-1], spec[order-1])
        # plt.plot(wave[order], spec[order])
        # plt.plot(wave[order+1], spec[order+1])
        # plt.show()

        # idx_ = tuple([wavelengths>min_overlap])
        # wavelength_min = 5900
        # wavelength_max = wavelength_min+200    ###### if you want to do a WAVELENGTH RANGE just input min and max here ######
        #print(wavelength_max)
        hdu.close()

        # plt.figure('e2ds vs s1d, straight from fits')
        # plt.plot(wavelengths_order, spec[order], label = 'e2ds')

        #### Now reading in s1d file ########
        print(file)
        hdu=fits.open('%s'%file)
        spec=hdu[0].data
        header=hdu[0].header
        spec_check = spec[spec<=0]

        # print(hdu[0].header['CRVAL1'])
        # print(hdu[0].header['CRPIX1'])
        # print(hdu[0].header['CDELT1'])
        # print(spec.shape[0])
        # print((hdu[0].header['CRPIX1']+np.arange(spec.shape[0])))
        # print((np.arange(spec.shape[0])))
        # print((hdu[0].header['CRPIX1']+np.arange(spec.shape[0]))*hdu[0].header['CDELT1'])
        # print((np.arange(spec.shape[0]))*hdu[0].header['CDELT1'])

        # wave=hdu[0].header['CRVAL1']+(hdu[0].header['CRPIX1']+np.arange(spec.shape[0]))*hdu[0].header['CDELT1']
        
        wave=hdu[0].header['CRVAL1']+(np.arange(spec.shape[0]))*hdu[0].header['CDELT1']

        print(wave)

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

            if len(wavelengths)>4681:
                wavelengths = wavelengths[:4681]
                fluxes = fluxes[:4681]
            # print(len(wavelengths))
            # print(np.max(wavelengths), np.min(wavelengths))
            # print(min_overlap)
            # print(max_overlap)
            # idx_overlap = np.logical_and(wavelengths>=min_overlap, wavelengths<=max_overlap)
            # idx_overlap = tuple([idx_overlap==False])
            overlap = []

            # plt.figure()
            # plt.plot(wavelengths, fluxes, label = 'Flux')
            # # plt.plot(wavelengths[idx_overlap], fluxes[idx_overlap])
            # plt.show()

            # plt.plot(wavelengths, fluxes, label = 's1d')
            # plt.legend()
            # plt.show()

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

            # ## test - s1d interpolated onto e2ds wavelength grid ##
            # hdu_e2ds=fits.open('%s'%file.replace('s1d', 'e2ds'))
            # spec_e2ds=hdu_e2ds[0].data
            # header_e2ds=hdu_e2ds[0].header

            # wave_e2ds=get_wave(spec_e2ds, header_e2ds)*(1.+brv/2.99792458e5)

            # # plt.figure()
            # # plt.scatter(np.arange(len(wave_e2ds[order][:-1])), wave_e2ds[order][1:]-wave_e2ds[order][:-1], label = 'e2ds wave (after berv)')
            # # plt.scatter(np.arange(len(wave_e2ds[order][:-1])), get_wave(spec_e2ds, header_e2ds)[order][1:]-get_wave(spec_e2ds, header_e2ds)[order][:-1], label = 'e2ds wave (before berv)')
            # # # plt.scatter(np.arange(len(wavelengths[:-1])), wavelengths[:-1]-wavelengths[1:], label = 's1d wave')
            # # plt.legend()
            # # plt.show()

            # # id = np.logical_and(wave_e2ds<np.max(wavelengths), wave_e2ds>np.min(wavelengths))
            # # print(wave_e2ds*u.AA)
            # # print(wavelengths*u.AA)
            # # print(fluxes*u.photon)

            # blaze_file = glob.glob('%sblaze_folder/**blaze_A*.fits'%(directory))
            # # print('%sblaze_folder/**blaze_A*.fits'%(directory))
            # # print(blaze_file)
            # blaze_file = blaze_file[0]
            # blaze =fits.open('%s'%blaze_file)
            # blaze_func = blaze[0].data
            # spec_e2ds = spec_e2ds/blaze_func
        
            # diff_arr = wavelengths[1:] - wavelengths[:-1]
            # print(diff_arr)
            # wavelengths = wavelengths[:-1]
            # fluxes = fluxes[:-1]/diff_arr

            # s1d_spec = Spectrum1D(spectral_axis = wavelengths*u.AA, flux = fluxes*u.Unit('photon AA-1'))
            # fluxcon = FluxConservingResampler()
            # new_spec = fluxcon(s1d_spec, wave_e2ds[order]*u.AA)

            # wavelengths = new_spec.spectral_axis
            # fluxes = new_spec.flux

            # wavelengths = wavelengths[10:len(wave_e2ds[order])-9]/u.AA
            # fluxes = fluxes[10:len(wave_e2ds[order])-9]/u.Unit('photon AA-1')
            # flux_error_order = flux_error_order[10:len(wave_e2ds[order])-10]

            # diff_arr = wavelengths[1:] - wavelengths[:-1]
            # print(diff_arr)
            # wavelengths = wavelengths[:-1]
            # fluxes = fluxes[:-1]*diff_arr

            # print(wavelengths)
            # print(fluxes)

            # plt.figure()
            # plt.title('interpolated s1d comapred to actual e2ds spectrum')
            # plt.plot(wavelengths, fluxes, label = 'interpolated s1d on e2ds wave grid')
            # plt.plot(wave_e2ds[order], spec_e2ds[order], label = 'e2ds spectrum')
            # plt.legend()
            # plt.show()


            # ## end of test ##

            # ## test2 - synthetic s1d spectrum - using wavelength grid ##
            # def gauss(x1, rv, sd, height, cont):
            #     y1 = height*np.exp(-(x1-rv)**2/(2*sd**2)) + cont
            #     return y1

            # wavelength_grid = wavelengths
            # flux_grid = np.ones(wavelength_grid.shape)

            # linelist = '/home/lsd/Documents/fulllinelist0001.txt'
            
            # linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
            # wavelengths_expected1 =np.array(linelist_expected[:,0])
            # depths_expected1 = np.array(linelist_expected[:,1])
            # # print(len(depths_expected1))

            # wavelength_min = np.min(wavelengths)
            # wavelength_max = np.max(wavelengths)

            # print(wavelength_min, wavelength_max)

            # wavelengths_expected=[]
            # depths_expected=[]
            # no_line =[]
            # for some in range(0, len(wavelengths_expected1)):
            #     line_min = 0.25
            #     if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max and depths_expected1[some]>=line_min:
            #         wavelengths_expected.append(wavelengths_expected1[some])
            #         #depths_expected.append(depths_expected1[some]+random.uniform(-0.1, 0.1))
            #         depths_expected.append(depths_expected1[some])
            #     else:
            #         pass
            
            # count_range = np.array([len(wavelengths_expected)]*10)*np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            # count_range = np.array(count_range, dtype = int)
            # print(count_range)
            # vgrid = np.linspace(-21,18,48)
            # try: ccf = fits.open(file.replace('s1d', 'ccf_K5'))
            # except: ccf = fits.open(file.replace('s1d', 'ccf_G2'))
            # rv = ccf[0].header['ESO DRS CCF RV']

            
            # for line in count_range:
            #     mid_wave = wavelengths_expected[line]
            #     wgrid = 2.99792458e5*mid_wave/(2.99792458e5-vgrid)
            #     id = np.logical_and(wavelength_grid<np.max(wgrid), wavelength_grid>np.min(wgrid))
            #     prof_wavelength_grid = wavelength_grid[id]
            #     prof_v_grid = ((prof_wavelength_grid - mid_wave)*2.99792458e5)/prof_wavelength_grid
            #     prof = gauss(prof_v_grid, rv, 2.47, -depths_expected[line], 1.)
            #     # plt.figure()
            #     # plt.plot(prof_wavelength_grid, prof)
            #     # plt.show()
            #     flux_grid[id] = prof

            # coeffs=np.polyfit(wavelengths, fluxes/fluxes[0], 3)
            # poly = np.poly1d(coeffs*fluxes[0])
            # fit = poly(wavelengths)

            # wavelengths = wavelength_grid
            # fluxes = flux_grid * fit

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

                    plt.figure('telluric masking')
                    plt.title('Spectrum - after telluric masking')
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
        try:sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
        except:sn = hdu[0].header['HIERARCH TNG DRS SPE EXT SN%s'%order]
        print('S/N: %s'%sn)
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
        # file_ccf = fits.open(file.replace('e2ds', 'ccf_G2'))
        # print(file_ccf[0].header['ESO DRS BERV'])
        try:brv=header['ESO DRS BERV']
        except:brv=header['TNG DRS BERV']
        # print(brv)
        wave_nonad=get_wave(spec, header)
        # if berv_opt == 'y':
        #     print('BERV corrected')
        wave = wave_nonad#*(1.+brv/2.99792458e5)
    
        # rv_drift=header['ESO DRS DRIFT RV'] 
        # print(rv_drift)
        wave_corr = (1.+brv/2.99792458e5)
        print(brv, (wave_corr-1)*2.99792458e5)

        blaze_file = glob.glob('%s/%s'%(directory, hdu[0].header['ESO DRS BLAZE FILE']))
        blaze_file = blaze_file[0]
        blaze =fits.open('%s'%blaze_file)
        blaze_func = blaze[0].data
        spec = spec/blaze_func
        flux_error = flux_error/blaze_func

        fluxes = spec[order]
        flux_error_order = flux_error[order]
        wavelengths = wave[order]

        # ## resampling flux into brv corrected frame - flux conserving
        # diff_arr = wavelengths[1:] - wavelengths[:-1]
        # print(diff_arr)
        # wavelengths = wavelengths[:-1]
        # fluxes = fluxes[:-1]

        # fluxes = fluxes/diff_arr

        # spec = Spectrum1D(spectral_axis = wavelengths*u.AA, flux = fluxes*u.Unit('photon AA-1'))
        # fluxcon = FluxConservingResampler()
        # new_spec = fluxcon(spec, (wavelengths*wave_corr)*u.AA)

        # wavelengths = new_spec.spectral_axis
        # fluxes = new_spec.flux

        # wavelengths = wavelengths[:4097]/u.AA
        # fluxes = fluxes[:4097]/u.Unit('photon AA-1')

        # diff_arr = wavelengths[1:] - wavelengths[:-1]
        # wavelengths = wavelengths[:-1]
        # fluxes = fluxes[:-1]*diff_arr

        # # test - e2ds interpolated onto s1d wavelength grid ##
        # hdu_s1d=fits.open('%s'%file.replace('e2ds', 's1d'))
        # spec_s1d=hdu_s1d[0].data
        # header_s1d=hdu_s1d[0].header

        # wave_s1d=header_s1d['CRVAL1']+(header_s1d['CRPIX1']+np.arange(spec_s1d.shape[0]))*header_s1d['CDELT1']
        # id = np.logical_and(wave_s1d<np.max(wavelengths), wave_s1d>np.min(wavelengths))

        # ## these fluxes are in photons per bin - I need them in photons per Angstrom
        # ## therefore i do flux/angstroms in pixel
        # diff_arr = wavelengths[1:] - wavelengths[:-1]
        # print(diff_arr)
        # wavelengths = wavelengths[:-1]
        # fluxes = fluxes[:-1]
        # # plt.figure('changing flux units')
        # # plt.plot(wavelengths, fluxes, label = 'flux per pixel')

        # fluxes = fluxes/diff_arr

        # # plt.plot(wavelengths, fluxes, label = ' flux per A')

        # e2ds_spec = Spectrum1D(spectral_axis = wavelengths*u.AA, flux = fluxes*u.Unit('photon AA-1'))
        # fluxcon = FluxConservingResampler()
        # new_spec = fluxcon(e2ds_spec, wave_s1d[id]*u.AA)

        # wavelengths = new_spec.spectral_axis
        # fluxes = new_spec.flux

        # wavelengths = wavelengths[:4097]/u.AA
        # fluxes = fluxes[:4097]/u.Unit('photon AA-1')

        # diff_arr = wavelengths[1:] - wavelengths[:-1]
        # wavelengths = wavelengths[:-1]
        # fluxes = fluxes[:-1]*diff_arr

        # find overlapping regions 
        last_wavelengths = wave[order-1]
        next_wavelengths = wave[order+1]
        last_spec = spec[order-1]
        next_spec = spec[order+1]
        last_error = flux_error[order-1]
        next_error = flux_error[order+1]
        min_overlap = np.min(wavelengths)
        max_overlap = np.max(wavelengths)
        
        # idx_ = tuple([wavelengths>min_overlap])
        last_idx = np.logical_and(last_wavelengths>min_overlap, last_wavelengths<max_overlap)
        next_idx = np.logical_and(next_wavelengths>min_overlap, next_wavelengths<max_overlap)
        
        overlap = np.array(([list(last_wavelengths[last_idx]), list(last_spec[last_idx]), list(last_error[last_idx])], [list(next_wavelengths[next_idx]), list(next_spec[next_idx]), list(next_error[next_idx])]))
        

    ## telluric correction
    tapas = fits.open('tapas_000001.fits')
    tapas_wvl = (tapas[1].data["wavelength"]) * 10.0
    tapas_trans = tapas[1].data["transmittance"]
    tapas.close()
    #try:brv=header['ESO DRS BERV']
    tapas_wvl = tapas_wvl[::-1]/(1.+brv/2.99792458e5)
    tapas_trans = tapas_trans[::-1]

    background = upper_envelope(tapas_wvl, tapas_trans)
    f = interp1d(tapas_wvl, tapas_trans / background, bounds_error=False)

    # plt.figure('telluric spec and real spec')
    # plt.plot(wavelengths, continuumfit(wavelengths, fluxes, 3))
    # plt.plot(wavelengths, f(wavelengths))
    # plt.show()
    
    # plt.figure()
    # plt.plot(tapas_wvl, tapas_trans)
    # plt.show()
    print('overlap accounted for')

    return np.array(fluxes), np.array(wavelengths), np.array(flux_error_order), sn, np.median(wavelengths), f(wavelengths), overlap ## for just LSD
############################################################################################################
