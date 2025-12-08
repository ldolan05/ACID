import numpy as np
from scipy import linalg
from astropy.io import  fits
import glob, time, warnings, sys, psutil, os, inspect
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d, LSQUnivariateSpline
from tqdm import tqdm
from numpy import integer as npint
from scipy.linalg import cho_factor, cho_solve
from beartype import beartype
c_kms = float(const.c/1e3)  # speed of light in km/s

@beartype
class LSD:
    """Class containing all useful functions for performing least-squares deconvolution.
    This does not simultaneously fit continuum and perform LSD (which ACID does). It is used
    for the initial parameters of the ACID mcmc run and for obtaining final profiles. It 
    can also be used as a standalone LSD implementation.
    """
    def __init__(self, Acid:object|None=None):
        """Intilialises the LSD class, optionally with an ACID instance to take parameters from.

        Parameters
        ----------
        Acid : object | None, optional
            The Acid instanse to draw parameters from, by default None
        """
        self.verbose          = getattr(Acid, 'verbose', 2) # class default of 2
        self.slurm            = "SLURM_JOB_ID" in os.environ
        self.velocities       = getattr(Acid, 'velocities', None)
        self.linelist         = getattr(Acid, 'linelist_path', None)
        self.order            = getattr(Acid, 'order', None)
        self.run_name         = getattr(Acid, 'name', None)
        self.adjust_continuum = None

    def run_LSD(
        self,
        wavelengths : np.ndarray,
        flux        : np.ndarray,
        errors      : np.ndarray,
        sn          : float|int|npint|np.ndarray,
        linelist    : str|dict        = None,
        velocities  : np.ndarray      = None,
        verbose     : int|npint|None  = None,
        ):
        """Runs the LSD algorithm to extract the average line profile from the observed spectrum.

        Parameters
        ----------
        wavelengths : np.ndarray
            Array of wavelengths of the observed spectrum in Angstroms
        flux : np.ndarray
            Array of flux values corresponding to the wavelengths (in linear space, and should be continuum normalized)
        errors : np.ndarray
            Array of error values corresponding to the flux
        sn : float | int
            Signal-to-noise ratio of the observed spectrum
        linelist : str | dict, optional
            Path to the linelist file or a dictionary containing 'wavelengths' and 'depths'. If the class was 
            not initialised with an Acid instance, this is required, by default None
        velocities : np.ndarray, optional
            Array of velocities corresponding to the observed spectrum. If the class was not initialised with 
            an Acid instance, this is required, by default None
        verbose : int | None, optional
            Verbosity level, if None, uses the class default of 2. See the Acid class for more information about
            verbosity integer levels, by default None
        """

        if not wavelengths.shape == flux.shape == errors.shape:
            raise ValueError("Input wavelengths, flux, and errors must have the same shape.")
        if wavelengths.ndim != 1:
            raise ValueError("Input wavelengths, flux, and errors must be 1D arrays.")        

        self.velocities = velocities if velocities is not None else self.velocities
        self.verbose    = verbose    if verbose    is not None else self.verbose # not from self.Acid
        self.linelist   = linelist   if linelist   is not None else self.linelist

        #### Read the EXPECTED linelist (for a slow rotator of the same spectral type) ####
        wavelengths_linelist, depths_linelist = self.read_linelist(self.linelist)

        # Clip linelist to wavelength range of spectrum
        wavelengths_linelist, depths_linelist = self.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist)

        # Apply S/N cut (of 1/(3*SN)) to linelist
        wavelengths_linelist, depths_linelist = self.sn_clip(wavelengths_linelist, depths_linelist, sn)

        # Convert to optical depth space for the linelist and the spectrum (may move to own function)
        errors = errors / flux
        flux = np.log(flux)
        depths_linelist = -np.log(1-depths_linelist)

        # Calculates alpha in optical depth, selects lines greater than 1/(3*sn)
        self.alpha = self.calc_alpha(wavelengths, wavelengths_linelist, depths_linelist)

        # Now solve for profile using Cholesky decomposition
        c_factor = self.calc_cholesky(self.alpha, errors)

        # Solve for profile and profile errors using Cholesky factors
        self.profile, self.profile_errors = self.solve_z(self.alpha, flux, errors, c_factor)

        return

    def read_linelist(
        self,
        linelist : str|dict):
        """Reads in the linelist from a file or dictionary.

        Parameters
        ----------
        linelist : str or dict
            Path to the linelist file or a dictionary containing 'wavelengths' and 'depths'
        
        Returns
        ----------
        wavelengths_linelist : np.ndarray
            Wavelengths from the linelist
        depths_linelist : np.ndarray
            Fluxes from the linelist
        """
        # Reading the linelist file or dictionary
        if isinstance(linelist, str):
            full_linelist = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9))
            wavelengths_linelist = np.array(full_linelist[:,0])
            depths_linelist = np.array(full_linelist[:,1])

        # If user input linelist_wl and linelist_depths in Acid
        else:
            wavelengths_linelist = np.array(linelist["wavelengths"])
            depths_linelist = np.array(linelist["depths"])

        # Save for debugging
        self._input_linelist_wavelengths = wavelengths_linelist
        self._input_linelist_depths = depths_linelist

        return wavelengths_linelist, depths_linelist

    @staticmethod
    def clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist):
        """Clips the linelist to only include lines within the wavelength range of the observed spectrum.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelengths of the observed spectrum
        wavelengths_linelist : np.ndarray
            Wavelengths from the linelist
        depths_linelist : np.ndarray
            Depths from the linelist

        Returns
        -------
        wavelengths_linelist : np.ndarray
            Clipped wavelengths from the linelist
        depths_linelist : np.ndarray
            Clipped depths from the linelist
        """
        lower, upper = wavelengths.min(), wavelengths.max()
        idx = (wavelengths_linelist >= lower) & (wavelengths_linelist <= upper)
        return wavelengths_linelist[idx], depths_linelist[idx]

    def sn_clip(self, wavelengths_linelist, depths_linelist, sn):
        """Applies a signal-to-noise cut to the linelist, removing lines shallower than 1/(3*sn).

        Parameters
        ----------
        wavelengths_linelist : np.ndarray
            Wavelengths from the linelist
        depths_linelist : np.ndarray
            Depths from the linelist
        sn : float
            Signal-to-noise ratio threshold

        Returns
        -------
        np.ndarray
            Clipped wavelengths from the linelist
        np.ndarray
            Clipped depths from the linelist
        """
        # Selecting lines deeper than 1/(3*sn)
        idx = (depths_linelist >= 1/(3*sn))
        wavelengths_linelist = wavelengths_linelist[idx]
        depths_linelist = depths_linelist[idx]
        if self.verbose > 0:
            ncut = np.sum(~idx)
            nrest = np.sum(idx)
            perc = 100 * nrest / (nrest + ncut)
            if perc < 5:
                print("Warning: Less than 5% of lines remain after S/N cut. Check your linelist and S/N value.")
        if self.verbose > 2:
            print(f"{perc:.2f}% of lines used in LSD: {nrest} out of {nrest + ncut} remain from S/N cut.")
        return wavelengths_linelist, depths_linelist

    def calc_alpha(
        self,
        wavelengths         : np.ndarray,
        wavelengths_linelist: np.ndarray,
        depths_linelist     : np.ndarray,
        velocities          : np.ndarray     = None,
        verbose             : int|npint|None = None,
        ):
        """Calculates the alpha matrix given flux and errors in OD space, and a line_list in OD space.
        Note that if this function is called without using run_LSD, there is no selection of lines deeper than 1/(3*sn).
        If you still wish to do this, it needs to be done in linear space with the sn_clip function before converting to OD space.

        Parameters
        ----------
        wavelengths : np.ndarray
            Array of wavelengths of the observed spectrum in optical depth space
        wavelengths_linelist : np.ndarray
            Array of wavelengths from the linelist in optical depth space
        depths_linelist : np.ndarray
            Array of depths from the linelist in optical depth space
        velocities : np.ndarray, optional
            Array of velocities, needs to either be initialised by class with Acid instance, or input here, by default None
        verbose : int | None, optional
            Verbosity level, uses the class default of 2 if None, by default None
        """

        self.velocities = velocities if velocities is not None else self.velocities
        self.verbose    = verbose    if verbose    is not None else self.verbose

        # Calculate velocity pixel size
        deltav = self.velocities[1] - self.velocities[0]

        # Clip linelist to wavelength range of spectrum (again just in case this is called without run_LSD)
        wavelengths_linelist, depths_linelist = self.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist)

        # Find differences and velocities
        blankwaves = wavelengths
        diff = blankwaves[:, np.newaxis] - wavelengths_linelist
        vel = c_kms * (diff / wavelengths_linelist)

        if self.slurm:
            available_memory = int(os.environ.get('SLURM_MEM_PER_NODE')) # in MB
            available_memory *= 1e6  # Convert to bytes as in the else statement below
        else:
            available_memory = psutil.virtual_memory().available

        mat_size = len(wavelengths_linelist) * len(self.velocities) * len(blankwaves) * 8 * 1e-9 # Matrix size in GB
        m_available = available_memory * 1e-9 / 2  # Available memory in GB (divided by 2 to be safe)

        # We can calculate the alpha matrix in one pass if the number of wavelengths is small enough
        if mat_size < m_available:
            # Calculating entire alpha matrix at once
            x = (vel[:, :, np.newaxis] - self.velocities) / deltav
            delta = np.clip(1.0 - np.abs(x), 0.0, 1.0)
            alpha = (depths_linelist[:, None] * delta).sum(axis=1)  # (nb, n_vel)

        # Else, calculate in blocks to save memory
        else:
            n_blank = len(blankwaves)
            n_vel   = len(self.velocities)
            mem_size = available_memory // 2
            bytes_per_row = n_blank * n_vel * 8 * 3 # *8 for float64, *3 for vel, x, delta in a row
            max_block = max(1, mem_size // bytes_per_row)
            block = int(min(max_block, len(wavelengths_linelist)))
            # Set initial alpha matrix to np.zeros
            alpha  = np.zeros((len(blankwaves), len(self.velocities)), dtype=np.float64)

            # Use tqdm progress bar if verbose
            if self.verbose>1:
                iterable = tqdm(range(0, len(wavelengths_linelist), block), desc='Calculating alpha matrix')
            else:
                iterable = range(0, len(wavelengths_linelist), block)

            for start_pos in iterable:
                # Ensure we don't go out of bounds on last iteration
                end_pos = min(start_pos + block, len(wavelengths_linelist))
                wl  = wavelengths_linelist[start_pos:end_pos]
                dep = depths_linelist[start_pos:end_pos]

                # Perform calculations for this block
                vel_blk = c_kms * (blankwaves[:, None] - wl) / wl
                x_blk   = (vel_blk[:, :, None] - self.velocities) / deltav
                delta   = np.clip(1.0 - np.abs(x_blk), 0.0, 1.0)                    

                alpha += (dep[:, None] * delta).sum(axis=1)
        return alpha

    @staticmethod
    def calc_cholesky(
        alpha : np.ndarray,
        error : np.ndarray,
        ):
        """Calculates the LHS Cholesky factorisation matrix given the alpha matrix and flux errors (in optical depth space)


        Parameters
        ----------
        alpha : np.ndarray
            The precomputed alpha matrix
        error : np.ndarray
            Flux errors in optical depth space

        Returns
        -------
        c_factor : tuple
            Cholesky factorisation matrix and lower/upper flag, to be put straight into solve_z as c_factor
        """
        V = 1.0 / (error ** 2) # variance vector in log space, error already in log space

        # M = αT V α,  b = αT V R
        AVA = alpha.T @ (V[:, None] * alpha)

        # Cholesky factorisation of M
        c_factor = cho_factor(AVA, overwrite_a=True)
        return c_factor

    @staticmethod
    def solve_z(
        alpha,
        flux,
        error,
        c_factor):
        """Solves for the LSD profile and its errors using the Cholesky factors.

        Parameters
        ----------
        alpha : np.ndarray
            The precomputed alpha matrix
        flux : np.ndarray
            The observed flux values in optical depth space
        error : np.ndarray
            The flux errors in optical depth space
        c_factor : tuple
            Cholesky factorisation matrix and lower/upper flag, to be put straight into solve_z as c_factor

        Returns
        -------
        profile, profile_errors : tuple
            LSD profile and its errors
        """
        V = 1.0 / (error ** 2) # variance vector in log space, error already in log space
        R = flux         # R matrix in log space

        # M = αT V α,  b = αT V R
        AVR = alpha.T @ (V * R)
        AVA = alpha.T @ (V[:, None] * alpha)

        # z = M⁻¹ b
        profile = cho_solve(c_factor, AVR)

        # Find error, cov(z) = M⁻¹, take diagonal, as in ACID v1
        cov_z = cho_solve(c_factor, np.eye(AVA.shape[0]))
        profile_errors = np.sqrt(np.diag(cov_z))
        return profile, profile_errors

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
