import numpy as np
from scipy import linalg
from astropy.io import  fits
import glob, time, warnings, sys, psutil, os, inspect
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, LSQUnivariateSpline
from tqdm import tqdm
ckms = float(const.c/1e3)  # speed of light in km/s

class LSD:

    def __init__(self, ACID=None):

        self.verbose = True
        self.adjust_continuum = None
        self.slurm = "SLURM_JOB_ID" in os.environ
        if not ACID:
            return
        self.wavelengths = ACID.combined_wavelengths
        self.flux_obs = ACID.fluxes_order1
        self.rms = ACID.flux_error_order1
        self.linelist = ACID.linelist_path
        self.poly_ord = ACID.poly_ord
        self.sn = ACID.combined_sn
        self.order = ACID.order
        self.run_name = ACID.name
        self.velocities = ACID.velocities
        self.verbose = ACID.verbose

    def run_LSD(self, *args, **kwargs):
        # Nothing the args use to be:
        # wavelengths, flux_obs, rms, linelist, adjust_continuum, poly_ord, sn, order, run_name, velocities
        # and kwargs:
        # verbose=False
        # These can be overridden by passing in different values,
        # but by default use the ones from the class init
        arg_names = [
            "wavelengths", "flux_obs", "rms", "linelist", "adjust_continuum",
            "poly_ord", "sn", "order", "run_name", "velocities", "verbose"
        ] # legacy arg names

        params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in arg_names
        ]
        sig = inspect.Signature(params)

        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError as e:
            # More informative error if user duplicates an argument
            raise TypeError(str(e) + " — you may be passing both a positional and keyword argument for the same name.")

        # Apply valid arguments
        for name, value in bound.arguments.items():
            setattr(self, name, value)

        # Check for unexpected kwargs (not in arg_names)
        unexpected = set(kwargs) - set(arg_names)
        if unexpected:
            raise TypeError(f"Unexpected argument(s): {', '.join(unexpected)}")

        #idx = tuple([flux_obs>0])
        # Converting to optical depth space
        self.rms = self.rms / self.flux_obs
        self.flux_obs = np.log(self.flux_obs)

        deltav = self.velocities[1] - self.velocities[0]

        #### This is the EXPECTED linelist (for a slow rotator of the same spectral type) ####
        # Ben - Reading the linelist
        if isinstance(self.linelist, str):
            linelist_expected = np.genfromtxt('%s'%self.linelist, skip_header=4, delimiter=',', usecols=(1,9))
            wavelengths_expected_all = np.array(linelist_expected[:,0])
            depths_expected_all = np.array(linelist_expected[:,1])
        else: # If user input linelist_wl and depths in ACID
            wavelengths_expected_all = np.array(self.linelist["wavelength"])
            depths_expected_all = np.array(self.linelist["depth"])

        # Selecting lines within the wavelength range of the observed spectrum
        wavelength_min = np.min(self.wavelengths)
        wavelength_max = np.max(self.wavelengths)
        idx = np.logical_and(wavelengths_expected_all >= wavelength_min,
                            wavelengths_expected_all <= wavelength_max)
        self.wavelengths_expected = wavelengths_expected_all[idx]
        self.depths_expected = depths_expected_all[idx]

        # Selecting lines deeper than 1/(3*sn)
        line_min = 1 / (3 * self.sn)
        idx = (self.depths_expected >= line_min)
        self.wavelengths_expected = self.wavelengths_expected[idx]
        self.depths_expected = self.depths_expected[idx]
        self.no_line = len(self.depths_expected)
        
        # Converting to log space for depths
        self.depths_expected = -np.log(1-self.depths_expected)

        blankwaves = self.wavelengths
        R_matrix = self.flux_obs

        # Find differences and velocities
        diff = blankwaves[:, np.newaxis] - self.wavelengths_expected
        vel = ckms * (diff / self.wavelengths_expected)

        # We can calculate the alpha matrix in one pass if the number of wavelengths is small enough
        
        # Note this used to be "if len(wavelengths)<6000", which I believe is way too high
        # for 16GB RAM. With testing, I found that if the matrix was half the available memory,
        # it would always be fast, otherwise seperate into blocks as the else statment below.
        if self.slurm:
            available_memory = os.environ.get('SLURM_MEM_PER_NODE')
            print(available_memory)
            sys.exit() # Remove this when tested
        else:
            available_memory = psutil.virtual_memory().available
        mat_size = len(self.wavelengths_expected) * len(self.velocities) * len(blankwaves) * 8 * 1e-9 # Matrix size in GB
        m_available = available_memory * 1e-9 / 2  # Available memory in GB (divided by 2 to be safe)
        if mat_size < m_available:

            x = (vel[:, :, np.newaxis] - self.velocities) / deltav
            delta = np.clip(1.0 - np.abs(x), 0.0, 1.0)
            self.alpha = (self.depths_expected[:, None] * delta).sum(axis=1)  # (nb, n_vel)

        else:
            n_blank = len(blankwaves)
            n_vel   = len(self.velocities)
            mem_size = available_memory // 2
            bytes_per_row = n_blank * n_vel * 8 * 3 # *8 for float64, *3 for vel, x, delta in a row
            max_block = max(1, mem_size // bytes_per_row)
            block = min(max_block, len(self.wavelengths_expected))
            # block = 512 # after initial testing, this value is a good compromise between memory use and speed
            self.alpha  = np.zeros((len(blankwaves), len(self.velocities)), dtype=np.float64)
            if self.verbose:
                iterable = tqdm(range(0, len(self.wavelengths_expected), block), desc='Calculating alpha matrix')
            else:
                iterable = range(0, len(self.wavelengths_expected), block)
            for start_pos in iterable:
                end_pos = min(start_pos + block, len(self.wavelengths_expected)) # ensure we don't go out of bounds on last iteration
                wl  = self.wavelengths_expected[start_pos:end_pos]
                dep = self.depths_expected[start_pos:end_pos]

                vel_blk = ckms * (blankwaves[:, None] - wl) / wl
                x_blk   = (vel_blk[:, :, None] - self.velocities) / deltav
                delta   = np.clip(1.0 - np.abs(x_blk), 0.0, 1.0)                    

                self.alpha += (dep[:, None] * delta).sum(axis=1)

        id_matrix = np.identity(len(self.flux_obs))
        S_matrix = (1/self.rms) * id_matrix

        S_squared = np.dot(S_matrix, S_matrix)
        alpha_transpose = (np.transpose(self.alpha))

        RHS_1 = np.dot(alpha_transpose, S_squared)
        RHS_final = np.dot(RHS_1, R_matrix)

        LHS_preinvert = np.dot(RHS_1, self.alpha)
        LHS_prep = np.matrix(LHS_preinvert)

        P, L, U = linalg.lu(LHS_prep)

        n = len(LHS_prep)
        B = np.identity(n)
        Z = linalg.solve_triangular(L, B, lower=True)
        X = linalg.solve_triangular(U, Z, lower=False)
        LHS_final = np.matmul(X, np.transpose(P))
    
        self.profile = np.dot(LHS_final, RHS_final)
        self.profile_errors_squared = np.diagonal(LHS_final)
        self.profile_errors = np.sqrt(self.profile_errors_squared)

        return

    def get_wave(self, data, header):

        wave = np.array(data*0., dtype = 'longdouble')
        no = data.shape[0]
        npix = data.shape[1]
        d = header['ESO DRS CAL TH DEG LL']
        xx0 = np.arange(npix)
        xx = []
        for i in range(d+1):
            xx.append(xx0**i)
        xx = np.asarray(xx, dtype = 'longdouble')

        for o in range(no):
            for i in range(d+1):
                idx = i + o * (d + 1)
                par = np.longdouble(header['ESO DRS CAL TH COEFF LL%d' % idx])
                wave[o, :] = wave[o, :] + par * xx[i, :]
            #for x in range(npix):
            #  wave[o,x]=wave[o,x]+par*xx[i,x]#float(x)**float(i)

        return wave

    def continuumfit(self, wavelengths1, fluxes1, poly_ord):

            fluxes = fluxes1
            wavelengths = wavelengths1

            idx = wavelengths.argsort()
            wavelength = wavelengths[idx]
            fluxe = fluxes[idx]
            clipped_flux = []
            clipped_waves = []
            binsize = 100
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

    def upper_envelope(self, x, y):
        # from MM-LSD code - give credit if needed
        # used to compute the tapas continuum. find peaks then fit spline to it.
        peaks = find_peaks(y, height=0.2, distance=len(x) // 500)[0]
        # t= knot positions
        spl = LSQUnivariateSpline(x=x[peaks], y=y[peaks], t=x[peaks][5::10])
        return spl(x)

    def blaze_correct(self, file_type, spec_type, order, file, directory, masking, run_name, berv_opt):
        #### Inputing spectrum depending on file_type and spec_type #####

        if file_type == 's1d':
            #### Finding min and max wavelength from e2ds for the specified order ######
            file_e2ds = file.replace('s1d', 'e2ds')
            # print(file_e2ds)
            hdu=fits.open('%s'%file_e2ds)
            sn = hdu[0].header['HIERARCH ESO DRS SPE EXT SN%s'%order]
            spec = hdu[0].data
            header = hdu[0].header
            brv = header['ESO DRS BERV']
            # print('hi')
            spec_check = spec[spec<=0]
            # if len(spec_check)>0:
                # print('WARNING NEGATIVE/ZERO FLUX - corrected')

            where_are_zeros = (spec<=0)
            spec[where_are_zeros] = 1000000000000
            flux_error = np.sqrt(spec)

            wave=self.get_wave(spec, header)*(1.+brv/2.99792458e5)
            wavelengths_order = wave[order]
            wavelength_min = np.min(wavelengths_order)
            wavelength_max = np.max(wavelengths_order)

            ## remove overlapping region (always remove the overlap at the start of the order, i.e the min_overlap)
            last_wavelengths = wave[order-1]
            next_wavelengths = wave[order+1]
            min_overlap = np.max(last_wavelengths)
            max_overlap = np.min(next_wavelengths)

            # idx_ = tuple([wavelengths>min_overlap])
            # wavelength_min = 5900
            # wavelength_max = wavelength_min+200    ###### if you want to do a WAVELENGTH RANGE just input min and max here ######
            hdu.close()

            #### Now reading in s1d file ########
            hdu=fits.open('%s'%file)
            spec=hdu[0].data
            header=hdu[0].header
            spec_check = spec[spec<=0]
            
            wave=hdu[0].header['CRVAL1']+(np.arange(spec.shape[0]))*hdu[0].header['CDELT1']
            
            where_are_zeros = (spec<=0)
            spec[where_are_zeros] = 1000000000000
            flux_error = np.sqrt(spec)

            if spec_type == 'order':
                wavelengths=[]
                fluxes=[]
                errors = []
                for value in range(0, len(wave)):
                    l_wave=wave[value]
                    l_flux=spec[value]
                    l_error=flux_error[value]
                    if l_wave>=wavelength_min and l_wave<=wavelength_max:
                        wavelengths.append(l_wave)
                        fluxes.append(l_flux)
                        errors.append(l_error)
                wavelengths = np.array(wavelengths)
                fluxes = np.array(fluxes)
                errors = np.array(errors)


                if len(wavelengths)>5144:
                    wavelengths = wavelengths[:5144]
                    fluxes = fluxes[:5144]
                    flux_error = errors[:5144]

                spec_check = fluxes[fluxes<=0]
                # if len(spec_check)>0:
                #     print('WARNING NEGATIVE/ZERO FLUX - corrected')

                # where_are_zeros = (spec<=0)
                # spec[where_are_zeros] = 1000000000000
                # flux_error = np.sqrt(spec)
                
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

                        plt.figure('telluric masking')
                        plt.title('Spectrum - after telluric masking')
                        plt.plot(wavelengths, fluxes)
                        # print(masked_waves)
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
            # print('S/N: %s'%sn)
            spec_check = spec[spec<=0]
            # if len(spec_check)>0:
            #     print('WARNING NEGATIVE/ZERO FLUX - corrected')


            # where_are_NaNs = np.isnan(flux_error)
            # flux_error[where_are_NaNs] = 1000000000000
            where_are_zeros = (spec<=0)
            spec[where_are_zeros] = 1000000000000
            flux_error = np.sqrt(spec)
            '''
            flux_error1 = header['HIERARCH ESO DRS SPE EXT SN%s'%order]

            flux_error = header['HIERARCH ESO DRS CAL TH RMS ORDER%s'%order]
            print(flux_error, flux_error1)

            flux_error = flux_error*np.ones(np.shape(spec))
            '''
            # file_ccf = fits.open(file.replace('e2ds', 'ccf_G2'))
            # print(file_ccf[0].header['ESO DRS BERV'])
            brv = np.longdouble(header['ESO DRS BERV'])
            # print(brv)
            wave_nonad = self.get_wave(spec, header)
            # if berv_opt == 'y':
            #     print('BERV corrected')
            wave = wave_nonad*(1.+brv/2.99792458e5)
            wave = np.array(wave, dtype = 'double')
            # if berv_opt == 'n':
            #     print('BERV not corrected')
            # wave = wave_nonad
        
            # rv_drift=header['ESO DRS DRIFT RV'] 
            # print(rv_drift)
            wave_corr = (1.+brv/2.99792458e5)
            # print(brv, (wave_corr-1)*2.99792458e5)

            # inp = input('Enter to continue...')
            '''
            plt.figure('Spectrum directly from fits file')
            plt.title('Spectrum directly from fits file')
            plt.errorbar(wave[order], spec[order], yerr = flux_error[order])
            plt.xlabel('wavelengths')
            plt.ylabel('flux')
            plt.show()
            '''

            blaze_file = glob.glob('tests/data/*blaze_B*.fits')

            blaze_file = blaze_file[0]
            # try:
            #     blaze_file = glob.glob('./**blaze_A*.fits')
            #     # print('%sblaze_folder/**blaze_A*.fits'%(directory))
            #     # print(blaze_file)
            #     blaze_file = blaze_file[0]
            # except: 
            #     try:
            #         blaze_file = glob.glob('/Users/lucydolan/Starbase/problem_frames/**blaze_A*.fits')
            #         blaze_file = blaze_file[0]
            #     except:
            #         blaze_file = glob.glob('tests/data/**blaze_A*.fits')
            blaze = fits.open('%s'%blaze_file)
            blaze_func = blaze[0].data
            min_rows = min(spec.shape[0], blaze_func.shape[0], flux_error.shape[0])
            spec, blaze_func, flux_error = spec[:min_rows], blaze_func[:min_rows], flux_error[:min_rows]
            spec = spec/blaze_func
            flux_error = flux_error/blaze_func

            
        
            fluxes = spec[order]
            flux_error_order = flux_error[order]
            wavelengths = wave[order]

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
                        #print(np.max(mask), np.min(mask))
                        idx = np.logical_and(wavelengths>=np.min(mask), wavelengths<=np.max(mask))
                        #print(flux_error_order[idx])
                        flux_error_order[idx] = 10000000000000000000

                        if len(wavelengths[idx])>0:
                            masked_waves.append(wavelengths[idx])

                    plt.figure('Telluric masking')
                    plt.title('Spectrum - telluric masking')
                    plt.plot(wavelengths, fluxes)
                    # print(masked_waves)
                    for masked_wave in masked_waves:
                        plt.axvspan(np.min(masked_wave), np.max(masked_wave), alpha=0.5, color='red')
                    #print('new version')
                    plt.savefig('/home/lsd/Documents/LSD_Figures/masking_plots/order%s_masks_%s'%(order, run_name))
                    plt.ylabel('flux')
                    plt.xlabel('wavelength')

                if response == 'n':
                    print('yay!')


            elif masking == 'unmasked':
                masked_waves = []
                masked_waves = np.array(masked_waves)

            else: print('WARNING - masking not catching - must be either "masked" or "unmasked"')

            hdu.close()

        #flux_error_order = (flux_error_order)/(np.max(fluxes)-np.min(fluxes))
        #print('flux error: %s'%flux_error_order)
        #fluxes = (fluxes - np.min(fluxes))/(np.max(fluxes)-np.min(fluxes))
        #idx = tuple([fluxes>0])

        return np.array(fluxes), np.array(wavelengths), np.array(flux_error_order), sn ## for just LSD
