# Old LSD alpha matrix code, iterative step:
# The below I think works similarly to above, but but uses only for loops, rather then using a for loop
# that only involves numpy calculations. With testing, the above is much faster.
else:
    warnings.warn('Large wavelength ranges give large computation time. Seperate wavelength range into smaller chunks for faster computation.', DeprecationWarning, stacklevel=2)
    alpha = np.zeros((len(blankwaves), len(velocities)))

    for j in tqdm(range(0, len(blankwaves)), desc='Calculating alpha matrix'):
        for i in (range(0, len(self.wavelengths_expected))):
            vdiff = ((blankwaves[j] - self.wavelengths_expected[i]) * ckms) / self.wavelengths_expected[i]
            if vdiff <= (np.max(velocities) + deltav) and vdiff >= (np.min(velocities) - deltav):
                diff = blankwaves[j] - self.wavelengths_expected[i]
                vel = const.c / 1e3 * (diff / self.wavelengths_expected[i])
                for k in range(0, len(velocities)):
                    x = (velocities[k] - vel) / deltav
                    if -1. < x and x < 0.:
                        delta_x = (1 + x)
                        alpha[j, k] = alpha[j, k] + depths_expected[i] * delta_x
                    elif 0. <= x and x < 1.:
                        delta_x = (1 - x)
                        alpha[j, k] = alpha[j, k] + depths_expected[i] * delta_x
            else:
                pass

# Vectorized old part:
# The below was simplified using np.clip (as shown above)
            alpha_mask_1 = np.logical_and(-1. < x, x < 0.)
            alpha_mask_2 = np.logical_and(0. <= x, x < 1.)
            delta_x_1 = 1 + x
            delta_x_2 = 1 - x
            delta_x_1[alpha_mask_1==False] = 0
            delta_x_2[alpha_mask_2==False] = 0
            # Update alpha array using calculated delta_x values
            alpha = np.zeros((len(blankwaves), len(velocities)))
            alpha += (depths_expected[:, np.newaxis] * delta_x_1).sum(axis=1)
            alpha += (depths_expected[:, np.newaxis] * delta_x_2).sum(axis=1)

# Old blaze correct code part 1:
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

# Old blaze correct code part 2:

# plt.figure()
            # plt.figure('blaze for orders 28, 29 and 30')
            # plt.plot(wave[28], blaze[0].data[28])
            # plt.plot(wave[29], blaze[0].data[29])
            # plt.plot(wave[30], blaze[0].data[30])

            # plt.figure()
            # plt.title('blaze for orders 28, 29 and 30 summed together')
            # pixel_grid = np.linspace(np.min(wave[28]), np.max(wave[30]), len(np.unique(wave[28:30])))
            # blaze_sum = np.zeros(pixel_grid.shape)

            # f28 = interp1d(wave[28], blaze[0].data[28], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
            # f29 = interp1d(wave[29], blaze[0].data[29], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
            # f30 = interp1d(wave[30], blaze[0].data[30], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')

            # idx28 = np.logical_and(pixel_grid>np.min(wave[28]), pixel_grid<np.max(wave[28]))
            # idx29 = np.logical_and(pixel_grid>np.min(wave[29]), pixel_grid<np.max(wave[29]))
            # idx30 = np.logical_and(pixel_grid>np.min(wave[30]), pixel_grid<np.max(wave[30]))

            # blaze_28 = f28(pixel_grid[idx28])
            # blaze_29 = f29(pixel_grid[idx29])
            # blaze_30 = f30(pixel_grid[idx30])

            # wave28 = pixel_grid[idx28]
            # wave29 = pixel_grid[idx29]
            # wave30 = pixel_grid[idx30]

            # for pixel in range(len(pixel_grid)):
            #     wavell = pixel_grid[pixel]
                
            #     idx28 = tuple([wave28==wavell])
            #     idx29 = tuple([wave29==wavell])
            #     idx30 = tuple([wave30==wavell])
                
            #     try: b = blaze_28[idx28][0]
            #     except: b=0
            #     try: b1 = blaze_29[idx29][0]
            #     except: b1=0
            #     try: b2 = blaze_30[idx30][0]
            #     except: b2=0

            #     print(wavell, b, b1, b2)

            #     blaze_sum[pixel] = b + b1 + b2 

            # plt.plot(pixel_grid, blaze_sum)
            # plt.show()

            # plt.figure()
            # plt.title('e2ds after blaze orders 28, 29 and 30')
            # plt.plot(wave[28], spec[28])
            # plt.plot(wave[29], spec[29])
            # plt.plot(wave[30], spec[30])
            
            ## TEST - adjusting e2ds spectrum onto s1d continuum ##

            # ## first interpolate s1d onto e2ds wavelength grid ##
            # s1d_file = fits.open(file.replace('e2ds', 's1d'))
            # s1d_spec = s1d_file[0].data
            # wave_s1d = s1d_file[0].header['CRVAL1']+(np.arange(s1d_spec.shape[0]))*s1d_file[0].header['CDELT1']

            # wavelengths = wave_s1d
            # fluxes = s1d_spec

            # plt.figure()
            # plt.plot(wavelengths, fluxes, label = 'e2ds spectrum - corrected to s1d continuum')
            # plt.plot(wave_s1d, s1d_spec, label = 's1d spectrum')
            # plt.legend()
            # plt.show()

            # plt.figure()
            # plt.plot(wavelengths, fluxes, label = 's1d')
            # plt.legend()

            # diff_arr = wavelengths[1:] - wavelengths[:-1]
            # # print(diff_arr)
            # wavelengths = wavelengths[:-1]
            # fluxes = fluxes[:-1]

            # fluxes = fluxes/diff_arr

            # fluxes = np.ones(fluxes.shape)
            # for i in range(len(fluxes)):
            #     fluxes[i] = fluxes[i]*8000
            # plt.figure()
            # plt.plot(wavelengths, fluxes, label = 's1d in photons AA-1')
            # #plt.legend()

            # interpolate s1d onto e2ds wavlengths - non flux conserving
            # s1dd_spec = Spectrum1D(spectral_axis = wavelengths*u.AA, flux = fluxes*u.Unit('photon AA-1'))
            # fluxcon = FluxConservingResampler()
            # extended_e2ds_wave = np.concatenate((wave[order], [wave[order][-1]+0.01]))

            ## MM-LSD way

            # fluxes = spec[order]
            # flux_error_order = flux_error[order]
            # wavelengths = wave[order]

            # for i in range(len(fluxes)):
            #     if i ==0:
            #         # print(fluxes[i])
            #         fluxes[i] = fluxes[i]*(0.01/(wavelengths[1]-wavelengths[0]))
            #         # print(0.01/(2.99792458e5*(wavelengths[1]-wavelengths[0])))
            #         # print(fluxes[i])
            #     else:
            #         # print(fluxes[i])
            #         fluxes[i] = fluxes[i]*(0.01/(wavelengths[i]-wavelengths[i-1]))
            #         # print(0.01/(2.99792458e5*(wavelengths[1]-wavelengths[0])))
            #         # print(fluxes[i])

            # ## end of MM-LSD way

            # new_spec = fluxcon(s1dd_spec, wavelengths*u.AA)

            # reference_wave = new_spec.spectral_axis/u.AA
            # reference_flux = new_spec.flux/u.Unit('photon AA-1')

            # # print(len(wavelengths_new))
            # # print(len(wave[order]))

            # # # plt.plot(wavelengths_new, fluxes, label = 'interpolated s1d in photons AA-1')
            # # # plt.legend()

            # #diff_arr = wavelengths_new[1:] - wavelengths_new[:-1]
            # #reference_wave = wavelengths_new[:-1]
            # #reference_flux = fluxes[:-1]*diff_arr
            
            # # # print(len(reference_wave))
            # # # print(len(wave[order]))

            # # # print(reference_wave-wave[order])

            # # # plt.figure()
            # # # plt.plot(reference_wave, reference_flux, label = 'interpolated s1d in photons per bin')
            # # # plt.legend()
            # # # plt.show()
            # # ## divide e2ds spectrum by interpolated s1d and fit polynomial to result

            # reference_wave = np.array(reference_wave, dtype = float)
            # reference_flux = np.array(reference_flux, dtype = float)
            # div_frame = fluxes/reference_flux

            # # plt.figure()
            # # plt.plot(reference_wave, div_frame)
            # # plt.show()

            # # # ### creating windows to fit polynomial to
            # # # binned = np.zeros(int(len(div_frame)/2))
            # # # binned_waves = np.zeros(int(len(div_frame)/2))
            # # # for i in range(0, len(div_frame)-1, 2):
            # # #     pos = int(i/2)
            # # #     binned[pos] = (div_frame[i]+div_frame[i+1])/2
            # # #     binned_waves[pos] = (reference_wave[i]+reference_wave[i+1])/2

            # # # plt.plot(frame_wavelengths[n], frames_unadjusted[n], color = 'b', label = 'unadjusted')
            # # # plt.figure()
            # # # plt.plot(frame_wavelengths[n], frames[n])
            # # # plt.show()

            # ### fitting polynomial to div_frame
            # coeffs=np.polyfit(reference_wave, div_frame, 3)
            # poly = np.poly1d(coeffs)
            # # print(coeffs)
            # inputs = coeffs[::-1]
            # # print(inputs)

            # wavelengths = reference_wave

            # fit = poly(wavelengths)
        
            # # # plt.figure()
            # # # plt.plot(reference_wave, reference_flux, label= 'reference')
            # # # plt.plot(wave[order], spec[order], label = 'e2ds')
            # # # plt.legend()

            # # # plt.figure()
            # # # plt.plot(reference_wave-wavelengths[:-1], label = 'reference_wave-wavelengths')
            # # # plt.legend()
            # # # plt.show()

            # # plt.figure()
            # # plt.scatter(reference_wave, div_frame, label = 'div flux')
            # # plt.plot(wavelengths, fit, label = 'fit')
            # # # plt.plot(reference_wave, poly(reference_wave), label = 'poly(reference_wave)')
            # # plt.legend()
            # # plt.show()
            
            # fluxes = spec[order]/fit
            # flux_error_order = flux_error[order]/fit

            # # plt.figure()
            # # plt.plot(wavelengths, fluxes/reference_flux, label = 'continuum adjusted e2ds/s1d')
            # # plt.legend()
            # # plt.show()

            # # plt.figure()
            # # plt.plot(wavelengths, fluxes-reference_flux, label = 'continuum adjusted e2ds-s1d')
            # # plt.legend()
            # # plt.show()
            # # plt.figure()
            # # plt.plot(wavelengths, spec[order], label = 'before')
            # # plt.plot(wavelengths, fit)
            # # plt.plot(wavelengths, fluxes, label= 'after')
            # # plt.legend()
            # # plt.show()

            # # idx_full = np.logical_and(wave_s1d>np.min(wave[28]), wave_s1d<np.max(wave[30]))
            # # plt.plot(wave_s1d[idx_full], s1d_spec[idx_full])
            # # plt.show()

            # blaze.close()
            # # plt.figure('after blaze correction - e2ds vs s1d - after berv correction')
            # # plt.plot(wave[order], spec[order], label = 'e2ds')
            # # plt.plot(wave_s1d[wave_s1d>np.max(wave[order])], spec_s1d[wave_s1d>np.max(wave[order])], label = 's1d')
            # # plt.show()

            

            # # # test - e2ds interpolated onto s1d wavelength grid ##
            # # hdu_s1d=fits.open('%s'%file.replace('e2ds', 's1d'))
            # # spec_s1d=hdu_s1d[0].data
            # # header_s1d=hdu_s1d[0].header

            # # wave_s1d=header_s1d['CRVAL1']+(header_s1d['CRPIX1']+np.arange(spec_s1d.shape[0]))*header_s1d['CDELT1']
            # # id = np.logical_and(wave_s1d<np.max(wavelengths), wave_s1d>np.min(wavelengths))
            # # print(wave_s1d*u.AA)
            # # print(wavelengths*u.AA)
            # # print(fluxes*u.Unit('erg cm-2 s-1 AA-1'))
            # # # plt.figure('s1d compared to interpolated e2ds')
            # # # plt.title('s1d compared to interpolated e2ds')
            # # # plt.plot(wave_s1d, spec_s1d, label = 's1d spectrum')

            # # ## these fluxes are in photons per bin - I need them in photons per Angstrom
            # # ## therefore i do flux/angstroms in pixel
            # # diff_arr = wavelengths[1:] - wavelengths[:-1]
            # # print(diff_arr)
            # # wavelengths = wavelengths[:-1]
            # # fluxes = fluxes[:-1]
            # # # plt.figure('changing flux units')
            # # # plt.plot(wavelengths, fluxes, label = 'flux per pixel')

            # # fluxes = fluxes/diff_arr

            # # # plt.plot(wavelengths, fluxes, label = ' flux per A')

            # # e2ds_spec = Spectrum1D(spectral_axis = wavelengths*u.AA, flux = fluxes*u.Unit('photon AA-1'))
            # # fluxcon = FluxConservingResampler()
            # # new_spec = fluxcon(e2ds_spec, wave_s1d[id]*u.AA)

            # # wavelengths = new_spec.spectral_axis
            # # fluxes = new_spec.flux

            # # wavelengths = wavelengths[:4097]/u.AA
            # # fluxes = fluxes[:4097]/u.Unit('photon AA-1')

            # # diff_arr = wavelengths[1:] - wavelengths[:-1]
            # # wavelengths = wavelengths[:-1]
            # # fluxes = fluxes[:-1]*diff_arr

            # # # plt.figure('s1d compared to interpolated e2ds') 
            # # # plt.plot(wavelengths, fluxes, label = 'interpolated e2ds onto s1d')
            # # # plt.xlim(np.min(wavelengths), np.max(wavelengths))
            # # # plt.legend()
            # # # plt.show() 

            # # print(wavelengths)
            # # print(fluxes)

            # #end of test ##

            # # # test2 - synthetic e2ds spectrum - using wavelength grid ##
            # # def gauss(x1, rv, sd, height, cont):
            # #     y1 = height*np.exp(-(x1-rv)**2/(2*sd**2)) + cont
            # #     return y1

            # # wavelength_grid = wavelengths
            # # flux_grid = np.ones(wavelength_grid.shape)

            # # linelist = '/home/lsd/Documents/fulllinelist0001.txt'

            # # linelist_expected = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
            # # wavelengths_expected1 =np.array(linelist_expected[:,0])
            # # depths_expected1 = np.array(linelist_expected[:,1])
            # # # print(len(depths_expected1))

            # # wavelength_min = np.min(wavelengths)
            # # wavelength_max = np.max(wavelengths)

            # # print(wavelength_min, wavelength_max)

            # # wavelengths_expected=[]
            # # depths_expected=[]
            # # no_line =[]
            # # for some in range(0, len(wavelengths_expected1)):
            # #     line_min = 0.25
            # #     if wavelengths_expected1[some]>=wavelength_min and wavelengths_expected1[some]<=wavelength_max and depths_expected1[some]>=line_min:
            # #         wavelengths_expected.append(wavelengths_expected1[some])
            # #         #depths_expected.append(depths_expected1[some]+random.uniform(-0.1, 0.1))
            # #         depths_expected.append(depths_expected1[some])
            # #     else:
            # #         pass

            # # count_range = np.array([len(wavelengths_expected)]*10)*np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            # # count_range = np.array(count_range, dtype = int)
            # # print(count_range)
            # # vgrid = np.linspace(-21,18,48)
            # # try: ccf = fits.open(file.replace('e2ds', 'ccf_K5'))
            # # except: ccf = fits.open(file.replace('e2ds', 'ccf_G2'))
            # # rv = ccf[0].header['ESO DRS CCF RV']


            # # for line in count_range:
            # #     mid_wave = wavelengths_expected[line]
            # #     wgrid = 2.99792458e5*mid_wave/(2.99792458e5-vgrid)
            # #     id = np.logical_and(wavelength_grid<np.max(wgrid), wavelength_grid>np.min(wgrid))
            # #     prof_wavelength_grid = wavelength_grid[id]
            # #     prof_v_grid = ((prof_wavelength_grid - mid_wave)*2.99792458e5)/prof_wavelength_grid
            # #     prof = gauss(prof_v_grid, rv, 2.47, -depths_expected[line], 1.)
            # #     # id = tuple([prof_v_grid<-0.99])
            # #     # plt.figure()
            # #     # plt.plot(prof_wavelength_grid, prof)
            # #     # plt.show()
            # #     flux_grid[id] = prof

            # # # plt.figure()
            # # # plt.plot(wavelength_grid, flux_grid)
            # # # plt.show()

            # # coeffs=np.polyfit(wavelengths, fluxes/fluxes[0], 3)
            # # poly = np.poly1d(coeffs*fluxes[0])
            # # fit = poly(wavelengths)

            # # wavelengths = wavelength_grid
            # # fluxes = flux_grid * fit

            # # plt.figure()
            # plt.plot(wavelengths, fluxes)
            # plt.show()

            # find overlapping regions 
            # last_wavelengths = wave[order-1]
            # next_wavelengths = wave[order+1]
            # last_spec = spec[order-1]
            # next_spec = spec[order+1]
            # last_error = flux_error[order-1]
            # next_error = flux_error[order+1]
            # min_overlap = np.min(wavelengths)
            # max_overlap = np.max(wavelengths)

            
            # idx_ = tuple([wavelengths>min_overlap])
            # last_idx = np.logical_and(last_wavelengths>min_overlap, last_wavelengths<max_overlap)
            # next_idx = np.logical_and(next_wavelengths>min_overlap, next_wavelengths<max_overlap)
            
            # overlap = np.array(([list(last_wavelengths[last_idx]), list(last_spec[last_idx]), list(last_error[last_idx])], [list(next_wavelengths[next_idx]), list(next_spec[next_idx]), list(next_error[next_idx])]))
            
            # # overlap[0, 0] = list(last_wavelengths[last_idx])
            # # overlap[0, 1] = list(last_spec[last_idx])
            # # overlap[1, 0] = list(next_wavelengths[next_idx])
            # # overlap[1, 1] = list(next_spec[next_idx])

            # print(overlap)
            # plt.figure()
            # plt.plot(wavelengths, fluxes)
            # plt.plot(wavelengths[idx_overlap], fluxes[idx_overlap])
            # plt.show()
            # wavelengths = wavelengths[idx]
            # fluxes = fluxes[idx]
            # flux_error_order = flux_error_order[idx]

# Telluric correction:
## telluric correction
        # tapas = fits.open('/Users/lucydolan/Starbase/tapas_000001.fits')
        # tapas_wvl = (tapas[1].data["wavelength"]) * 10.0
        # tapas_trans = tapas[1].data["transmittance"]
        # tapas.close()
        # brv=header['ESO DRS BERV']
        # tapas_wvl = tapas_wvl[::-1]/(1.+brv/2.99792458e5)
        # tapas_trans = tapas_trans[::-1]

        # background = upper_envelope(tapas_wvl, tapas_trans)
        # f = interp1d(tapas_wvl, tapas_trans / background, bounds_error=False)

        # plt.figure('telluric spec and real spec')
        # plt.plot(wavelengths, continuumfit(wavelengths, fluxes, 3))
        # plt.plot(wavelengths, f(wavelengths))
        # plt.show()
        
        # plt.figure()
        # plt.plot(tapas_wvl, tapas_trans)
        # plt.show()
        # print('overlap accounted for')

# Overlap stuff?
# print(len(wavelengths))
                # print(np.max(wavelengths), np.min(wavelengths))
                # print(min_overlap)
                # print(max_overlap)
                # idx_overlap = np.logical_and(wavelengths>=min_overlap, wavelengths<=max_overlap)
                # idx_overlap = tuple([idx_overlap==False])
                # overlap = []

                # plt.figure()
                # plt.plot(wavelengths, fluxes, label = 'Flux')
                # # plt.plot(wavelengths[idx_overlap], fluxes[idx_overlap])
                # plt.show()

                # plt.plot(wavelengths, fluxes, label = 's1d')
                # plt.legend()
                # plt.show()