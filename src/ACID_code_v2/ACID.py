import sys, emcee, warnings, os, time, importlib, inspect
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
import multiprocessing as mp
from multiprocessing import Pool
from . import utils
from . import LSD
from . import mcmc_utils

warnings.filterwarnings("ignore")
importlib.reload(LSD)
importlib.reload(utils)


class ACID:

    def __init__(self, tell_lines=None):
        self.all_frames = None
        self.order_range = [1]
        self.velocities = None
        self.linelist_path = None
        self.frames = None
        self.frame_wavelengths = None
        self.frame_flux = None
        self.frame_errors = None
        self.frame_sns = None
        self.combined_wavelengths = None
        self.combined_fluxes = None
        self.combined_flux_error = None
        self.combined_sn = None
        self.poly_inputs = None
        self.poly_ord = None
        self.run_name = None
        if tell_lines is not None:
            self.telluric_lines = tell_lines
        else:
            self.telluric_lines = [3820.33, 3933.66, 3968.47, 4327.74, 4307.90, 4383.55, 4861.34,
                                   5183.62, 5270.39, 5889.95, 5895.92, 6562.81, 7593.70, 8226.96]
        self.alpha = None
        self.initial_profile = None
        self.initial_profile_errors = None
        self.k_max = None
        self.model_inputs = None
        self.ndim = None
        self.nwalkers = None
        self.initial_state = None
        self.nsteps = None
        self.cores = None
        self.all_frames = None
        self.flat_samples = None
        self.dev_perc = None
        self.pix_chunk = None
        self.n_sigma = None
        self.profile = None
        self.poly_cos = None
        self.profile_err = None
        self.poly_cos_err = None
        self.continuum_error = None
        pass

    def _get_normalisation_coeffs(self, wl):
        a = 2 / (np.max(wl)-np.min(wl))
        b = 1 - a * np.max(wl)
        return a, b

    def continuumfit(self, fluxes, wavelengths, errors, poly_ord):
            
            cont_factor = fluxes[0]
            if cont_factor == 0: 
                cont_factor = np.mean(fluxes)
            idx = wavelengths.argsort()
            wavelength = wavelengths[idx]
            fluxe = fluxes[idx] / cont_factor
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
            coeffs = np.polyfit(clipped_waves, clipped_flux, poly_ord)
            poly = np.poly1d(coeffs)
            fit = poly(wavelengths) * cont_factor
            flux_obs = fluxes / fit
            new_errors = errors / fit
            poly_coeffs = np.concatenate((np.flip(coeffs), [cont_factor]))
            return poly_coeffs, flux_obs, new_errors

    def read_in_frames(self, order, filelist, file_type, directory=None):
        # read in first frame
        fluxes, wavelengths, flux_error_order, sn = LSD.LSD().blaze_correct(
            file_type, 'order', order, filelist[0], directory, 'unmasked', self.run_name, 'y')
        # fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct(
        #     file_type, 'order', order, filelist[0], directory, 'unmasked', self.run_name, 'y')

        frames = np.zeros((len(filelist), len(wavelengths)))
        errors = np.zeros((len(filelist), len(wavelengths)))
        frame_wavelengths = np.zeros((len(filelist), len(wavelengths)))
        sns = np.zeros((len(filelist), ))

        frames[0] = fluxes
        errors[0] = flux_error_order
        frame_wavelengths[0] = wavelengths
        sns[0] = sn

        def task_frames(frames, errors, frame_wavelengths, sns, i):
            file = filelist[i]
            frames[i], frame_wavelengths[i], errors[i], sns[i] = LSD.LSD().blaze_correct(
                file_type, 'order', order, file, directory, 'unmasked', self.run_name, 'y')
            # print(i, frames)
            return frames, frame_wavelengths, errors, sns
        
        ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
        for i in range(len(filelist[1:])+1):
            # print(i)
            frames, frame_wavelengths, errors, sns = task_frames(frames, errors, frame_wavelengths, sns, i)
            
        ### finding highest S/N frame, saves this as reference frame

        idx = (sns==np.max(sns))
        # global reference_wave
        reference_wave = frame_wavelengths[idx][0]
        reference_frame = frames[idx][0]
        reference_frame[reference_frame == 0] = 0.001
        reference_error = errors[idx][0]
        reference_error[reference_frame == 0] = 1000000000000000000

        # global frames_unadjusted
        frames_unadjusted = frames
        # global frame_errors_unadjusted
        frame_errors_unadjusted = errors

        ### each frame is divided by reference frame and then adjusted so that all spectra lie at the same continuum
        for n in range(len(frames)):
            f2 = interp1d(frame_wavelengths[n], frames[n], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
            div_frame = f2(reference_wave)/reference_frame

            idx_ref = (reference_frame<=0)
            div_frame[idx_ref]=1

            binned = []
            binned_waves = []
            binsize = int(round(len(div_frame)/5, 1))
            for i in range(0, len(div_frame), binsize):
                if i+binsize<len(reference_wave):
                    waves = reference_wave[i:i+binsize]
                    flux = div_frame[i:i+binsize]
                    waves = waves[abs(flux-np.median(flux))<0.1]
                    flux = flux[abs(flux-np.median(flux))<0.1]
                    binned.append(np.median(flux))
                    binned_waves.append(np.median(waves))

            binned = np.array(binned)
            binned_waves = np.array(binned_waves)
        
            ### fitting polynomial to div_frame
            try:coeffs = np.polyfit(binned_waves, binned, 4)
            except:coeffs = np.polyfit(binned_waves, binned, 2)
            poly = np.poly1d(coeffs)
            fit = poly(frame_wavelengths[n])
            frames[n] = frames[n]/fit
            errors[n] = errors[n]/fit
            idx = (frames[n] == 0)
            frames[n][idx] = 0.00001
            errors[n][idx] = 1000000000

        return frame_wavelengths, frames, errors, sns

    def calc_deltav(self, wavelengths):
        """Calculates velocity pixel size

        Calculates the velocity pixel size for the LSD velocity grid based off the spectral wavelengths.

        Args:
            wavelengths (array): Wavelengths for ACID input spectrum (in Angstroms).
            
        Returns:
            float: Velocity pixel size in km/s
        """
        resol = (wavelengths[-1]-wavelengths[0])/len(wavelengths)
        return resol / (wavelengths[0]+((wavelengths[-1]-wavelengths[0])/2)) * LSD.ckms

    def combine_spec(self, frame_wavelengths=None, frame_flux=None, frame_errors=None, frame_sns=None, _output=True):
        """Combines multiple spectral frames into one spectrum

        Parameters
        ----------
        frame_wavelengths : array, optional
            Wavelengths for the spectral frames, by default None
        frame_flux : array, optional
            Fluxes for the spectral frames, by default None
        frame_errors : array, optional
            Errors for the spectral frames, by default None
        frame_sns : array, optional
            Signal-to-noise ratio for the spectral frames, by default None
        _output : bool, optional
            Whether to output the combined spectrum, by default True

        Returns
        -------
        combined_wavelengths : array
            Wavelengths for the combined spectrum
        combined_spectrum : array
            Fluxes for the combined spectrum
        combined_errors : array
            Errors for the combined spectrum
        combined_sn : float
            Signal-to-noise ratio for the combined spectrum
        """
        # Inputs are now self.frame_wavelengths, self.frame_flux, self.frame_errors, self.frame_sns
        # they were: wavelengths_f, spectra_f, errors_f, sns_f):
        if frame_wavelengths:
            self.frame_wavelengths = frame_wavelengths
            self.frame_flux = frame_flux
            self.frame_errors = frame_errors
            self.frame_sns = frame_sns

        if len(self.frame_wavelengths)==1:
            self.combined_wavelengths = self.frame_wavelengths[0]
            self.combined_spectrum = self.frame_flux[0]
            self.combined_errors = self.frame_errors[0]
            self.combined_sn = self.frame_sns[0]

        else:
            self.combined_spectrum = np.copy(self.frame_flux)

            # combine all spectra to one spectrum
            for n in range(len(self.combined_spectrum)):

                self.combined_wavelengths = self.frame_wavelengths[np.argmax(self.frame_sns)]

                idx = np.where(self.frame_wavelengths[n] != 0)[0]

                f2 = interp1d(self.frame_wavelengths[n][idx], self.combined_spectrum[n][idx], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
                f2_err = interp1d(self.frame_wavelengths[n][idx], self.frame_errors[n][idx], kind = 'linear', bounds_error=False, fill_value = 'extrapolate')
                self.combined_spectrum[n] = f2(self.combined_wavelengths)
                self.frame_errors[n] = f2_err(self.combined_wavelengths)

                ## mask out out extrapolated areas
                idx_ex = np.logical_and(self.combined_wavelengths<=np.max(self.frame_wavelengths[n][idx]),
                                        self.combined_wavelengths>=np.min(self.frame_wavelengths[n][idx]))
                idx_ex = tuple([idx_ex==False])

                self.combined_spectrum[n][idx_ex] = 1.
                self.frame_errors[n][idx_ex] = 1000000000000

                ## mask out nans and zeros (these do not contribute to the main spectrum)
                where_are_NaNs = np.isnan(self.combined_spectrum[n])
                self.frame_errors[n][where_are_NaNs] = 1000000000000
                where_are_zeros = np.where(self.combined_spectrum[n] == 0)[0]
                self.frame_errors[n][where_are_zeros] = 1000000000000

                where_are_NaNs = np.isnan(self.frame_errors[n])
                self.frame_errors[n][where_are_NaNs] = 1000000000000
                where_are_zeros = np.where(self.frame_errors[n] == 0)[0]
                self.frame_errors[n][where_are_zeros] = 1000000000000

            width = len(self.combined_wavelengths)
            spectrum_f = np.zeros((width,))
            self.combined_errors = np.zeros((width,))

            for n in range(0,width):
                temp_spec_f = self.combined_spectrum[:, n]
                temp_err_f = self.frame_errors[:, n]

                weights_f = (1/temp_err_f**2)

                idx = tuple([temp_err_f>=1000000000000])
                weights_f[idx] = 0.

                if sum(weights_f) > 0:
                    weights_f = weights_f / np.sum(weights_f)

                    spectrum_f[n] = sum(weights_f * temp_spec_f)
                    self.combined_sn = sum(weights_f * self.frame_sns) / sum(weights_f)

                    self.combined_errors[n] = 1 / (sum(weights_f ** 2)) # TODO! CHECK this is the right calculation with the square
                
                else: 
                    spectrum_f[n] = np.mean(temp_spec_f)
                    self.combined_errors[n] = 1000000000000

            self.combined_spectrum = spectrum_f

        if _output is True:
            # ie if called as a function rather than from ACID function
            return self.combined_wavelengths, self.combined_spectrum, self.combined_errors, self.combined_sn

    def residual_mask(self, k_max=None):
        ## iterative residual masking - mask continuous areas first - then possibly progress to masking the narrow lines

        if k_max:
            self.k_max = k_max

        forward = mcmc_utils.model_func(self.model_inputs, self.x, alpha=self.alpha, k_max=self.k_max)

        a, b = self._get_normalisation_coeffs(self.x)

        mdl1 = 0
        for i in range(self.k_max, len(self.model_inputs) - 1):
            mdl1 = mdl1 + (self.model_inputs[i] * ((self.x * a) + b) ** (i - self.k_max))

        mdl1 = mdl1 * self.model_inputs[-1]

        data_normalised = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))
        forward_normalised = (forward - np.min(forward)) / (np.max(forward) - np.min(forward))
        residuals = data_normalised - forward_normalised
        
        ### finds consectuative sections where at least pix_chunk points have residuals greater than 0.25 - these are masked
        idx = (abs(residuals) > self.dev_perc / 100)

        flag_min = 0
        flag_max = 0
        for value in range(len(idx)):
            if idx[value] == True and flag_min <= value:
                flag_min = value
                flag_max = value
            elif idx[value] == True and flag_max < value:
                flag_max = value
            elif idx[value] == False and flag_max - flag_min >= self.pix_chunk:
                self.yerr[flag_min:flag_max] = 10000000000000000000
                flag_min = value
                flag_max = value

        ##############################################
        #                  TELLURICS                 #   
        ##############################################

        # self.yerr_compare = self.yerr.copy()

        ## masking tellurics
        for line in self.telluric_lines:
            limit = (21/LSD.ckms)*line +3
            idx = np.logical_and((line-limit) <= self.x, self.x <= (limit+line))
            self.yerr[idx] = 10000000000000000000

        self.residual_masks = tuple([self.yerr >= 10000000000000000000])

        ###################################
        ###      sigma clip masking     ###
        ###################################

        m = np.median(residuals)
        sigma = np.std(residuals)

        # TODO! : check what the hell is gong on here with a_old
        a_old = 1

        upper_clip = m + a_old * sigma
        lower_clip = m - a_old * sigma

        rcopy = residuals.copy()

        idx1 = tuple([rcopy <= lower_clip])
        idx2 = tuple([rcopy >= upper_clip])

        self.yerr[idx1] = 10000000000000000000
        self.yerr[idx2] = 10000000000000000000

        poly_inputs, _bin, bye = self.continuumfit(self.y, (self.x*a_old)+b, self.yerr, self.poly_ord)
        # velocities1, profile, profile_err, self.alpha, continuum_waves, continuum_flux, no_line = LSD.LSD(
        #     self.x, _bin, bye, self.linelist_path, 'False', self.poly_ord, 100, 30, self.run_name, self.velocities)
        LSD_masking = LSD.LSD()
        LSD_masking.run_LSD(self.x, _bin, bye, self.linelist_path, 'False',
                        self.poly_ord, 100, 30, self.run_name, self.velocities)
        # profile = LSD_masking.profile
        self.alpha = LSD_masking.alpha

        # model_input_resids = np.concatenate((profile, poly_inputs))

        # ## comment if you would like to keep sigma clipping masking in for final LSD run 
        # self.residual_masks = tuple([self.yerr>=1000000000000000000])

        return

    def _get_profiles(self, all_frames, counter):
        flux = self.frame_flux[counter]
        error = self.frame_errors[counter]
        wavelengths = self.frame_wavelengths[counter]
        sn = self.frame_sns[counter]

        a, b = self._get_normalisation_coeffs(wavelengths)

        mdl1 =0
        for i in np.arange(0, len(self.poly_cos)-1):
            mdl1 = mdl1+self.poly_cos[i]*((a*wavelengths)+b)**(i)
        mdl1 = mdl1*self.poly_cos[-1]

        # Masking based off residuals interpolated onto new wavelength grid
        if len(self.frame_wavelengths)>1:
            reference_wave = self.frame_wavelengths[self.frame_sns==max(self.frame_sns)][0]
        else:
            reference_wave = self.frame_wavelengths[0]
        mask_pos = np.ones(reference_wave.shape)
        mask_pos[self.residual_masks]=10000000000000000000
        f2 = interp1d(reference_wave, mask_pos, bounds_error = False, fill_value = np.nan)
        interp_mask_pos = f2(wavelengths)
        interp_mask_idx = tuple([interp_mask_pos>=10000000000000000000])

        error[interp_mask_idx]=10000000000000000000

        # corrrecting continuum
        error = (error/flux) + (self.continuum_error/mdl1)
        flux = flux/mdl1
        error  = flux*error

        remove = tuple([flux<0])
        flux[remove]=1.
        error[remove]=10000000000000000000

        idx = tuple([flux>0])
        
        if len(flux[idx])==0:
            print('continuing... frame %s'%counter)
        
        else:
            LSD_profiles = LSD.LSD(self)
            LSD_profiles.run_LSD(wavelengths, flux, error, self.linelist_path, 'False',
                                 self.poly_ord, sn, 10, self.run_name, self.velocities)
            profile_OD = LSD_profiles.profile
            profile_errors = LSD_profiles.profile_errors
            # velocities1, profile1, profile_errors, alpha, continuum_waves, continuum_flux, no_line= LSD_profiles(
            #     wavelengths, flux, error, self.linelist_path, 'False', self.poly_ord, sn, 10, 'test', self.velocities)

            # Need to check whats going on here with the -1
            p = np.exp(profile_OD)-1
            profile_f = np.exp(profile_OD)
            profile_errors_f = np.sqrt(profile_errors**2/profile_f**2)
            profile_f = profile_f-1

            all_frames[counter, self.order]=[profile_f, profile_errors_f]
            
            return all_frames

    def combineprofiles(self, spectra, errors):
        spectra = np.array(spectra)
        idx = np.isnan(spectra)
        shape_og = spectra.shape
        if len(spectra[idx])>0:
            spectra = spectra.reshape((len(spectra)*len(spectra[0]), ))
            for n in range(len(spectra)):
                if spectra[n] == np.nan:
                    spectra[n] = (spectra[n+1]+spectra[n-1])/2
                    if spectra[n] == np.nan:
                        spectra[n] = 0.
        spectra = spectra.reshape(shape_og)
        errors = np.array(errors)

        
        spectra_to_combine = []
        weights=[]
        for n in range(0, len(spectra)):
            if np.sum(spectra[n])!=0:
                spectra_to_combine.append(list(spectra[n]))
                temp_err = np.array(errors[n, :])
                weight = (1/temp_err**2)
                weights.append(np.mean(weight))
        weights = np.array(weights/sum(weights))

        spectra_to_combine = np.array(spectra_to_combine)

        length, width = np.shape(spectra_to_combine)
        spectrum = np.zeros((1,width))
        spec_errors = np.zeros((1,width))

        for n in range(0, width):
            temp_spec = spectra_to_combine[:, n]
            spectrum[0,n]=sum(weights*temp_spec)/sum(weights)
            spec_errors[0,n] = (np.std(temp_spec, ddof=1)**2) * np.sqrt(sum(weights**2))

        spectrum = list(np.reshape(spectrum, (width,)))
        spec_errors = list(np.reshape(spec_errors, (width,)))

        return  spectrum, spec_errors

    def run_ACID(self, input_wavelengths, input_spectra, input_spectral_errors, frame_sns, linelist_path,
            velocities, all_frames=None, poly_ord=3, pix_chunk=20, dev_perc=25, name="test",
            n_sig=1, telluric_lines=None, order=0, verbose=True, parallel=True, cores=None,
            nsteps=5000, return_frames=True):
        """Accurate Continuum fItting and Deconvolution

        Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra, returning an LSD profile for each spectrum given. 
        Spectra must cover a similiar wavelength range.

        Parameters
        ----------
        input_wavelengths : list or array
            Wavelengths for each frame (in Angstroms).
        input_spectra : list or array
            Spectral frames (in flux).
        input_spectral_errors : list or array
            Errors for each frame (in flux).
        frame_sns : list or array
            Average signal-to-noise ratio for each frame (used to calculate minimum line depth to consider from line list).
        linelist_path : str
            Path to linelist. Takes VALD linelist in long or short format as input. Minimum line depth input into VALD must
            be less than 1/(3*SN) where SN is the highest signal-to-noise ratio of the spectra.  
        velocities : array
            Velocity grid for LSD profiles (in km/s). For example, use: np.arange(-25, 25, 0.82) to create
        all_frames : str or array, optional
            Output array for resulting profiles. Only neccessary if looping ACID function over many wavelength
            regions or order (in the case of echelle spectra). General shape needs to be (no. of frames, 1, 2, no. of velocity pixels)., by default None
        poly_ord : int, optional
            Order of polynomial to fit as the continuum, by default 3
        pix_chunk : int, optional
            Size of 'bad' regions in pixels. 'bad' areas are identified by the residuals between an inital model
            and the data. If a residual deviates by a specified percentage (dev_perc) for a specified number of pixels, by default 20
        dev_perc : int, optional
            Allowed deviation percentage. 'bad' areas are identified by the residuals between an inital model
            and the data. If a residual deviates by a specified percentage (dev_perc) for a specified number of pixels, by default 25
        n_sig : int, optional
            Number of sigma to clip in sigma clipping. Ill fitting lines are identified by sigma-clipping the
            residuals between an inital model and the data. The regions that are clipped from the residuals will
            be masked in the spectra. This masking is only applied to find the continuum fit and is removed when
            LSD is applied to obtain the final profiles, by default 1
        telluric_lines : list, optional
            List of wavelengths (in Angstroms) of telluric lines to be masked. This can also include problematic
            lines/features that should be masked also. For each wavelengths in the list ~3Å eith side of the line is masked., by default None
        order : int, optional
            Only applicable if an all_frames output array has been provided as this is the order position in that
            array where the result should be input. i.e. if order = 5 the output profile and errors would be inserted in all_frames[:, 5]., by default 0
        verbose : bool, optional
            If True prints out time taken for each section of the code. Defaults to True., by default True
        parallel : bool, optional
            If True uses multiprocessing to calculate the profiles for each frame in parallel, by default True
        cores : int, optional
            Number of cores to use if parallel=True. If None (default) all available cores will be used, by default None
        nsteps : int, optional
            nsteps (int, optional): Number of steps for the MCMC to run, try increasing if it doesn't converge, by default 5000
        return_frames : bool, optional
            If True, returns the all_frames array with the resulting profiles, by default True
        Returns
        -------
        all_frames : array
            Resulting profiles and errors for spectra, if specified with return_frames=True.

        Raises
        ------
        TypeError
            If the input types are not as expected.
        """
        ### Setup

        # Ensure inputs are lists, np.arrays are converted to lists, sn treated separately
        input_wavelengths, input_spectra, input_spectral_errors = [
            utils.ensure_list(v) for v in (input_wavelengths, input_spectra, input_spectral_errors)]
        frame_sns = utils.ensure_list(frame_sns, sn=True)

        # Define telluric_lines if not input, check type if it is
        if telluric_lines:
            self.telluric_lines = telluric_lines
        if not isinstance(self.telluric_lines, list):
            raise TypeError("telluric_lines must be a list of telluric lines to mask (could be empty or single-valued)")

        self.frame_wavelengths = np.array(input_wavelengths)
        self.frame_flux = np.array(input_spectra)
        self.frame_errors = np.array(input_spectral_errors)
        self.frame_sns = np.array(frame_sns)

        self.velocities = velocities
        self.linelist_path = linelist_path
        self.poly_ord = poly_ord
        self.run_name = name
        self.verbose = verbose
        self.nsteps = nsteps
        self.order = order

        self.pix_chunk = pix_chunk
        self.dev_perc = dev_perc
        self.n_sig = n_sig

        if all_frames is None:
            if self.all_frames is None:
                # By default order_range is [1], so len(self.order_range) = 1, which is same as original
                # code behaviour. This change allows self.order_range to be used in ACID_HARPS.
                self.all_frames = np.zeros((len(self.frame_flux), len(self.order_range), 2, len(self.velocities)))
        else:
            self.all_frames = all_frames
        if not isinstance(self.all_frames, np.ndarray):
            raise TypeError("'all_frames' must be a numpy array")

        if verbose:
            t0 = time.time()
            print('Initialising...')

        # Combines spectra from each frame (weighted based of S/N), returns to S/N of combined spectra
        # This function uses as inputs:
        # self.frame_wavelengths, self.frame_flux, self.frame_errors, self.frame_sns
        # To generate:
        # self.combined_wavelengths, self.combined_spectrum, self.combined_errors, self.combined_sn
        self.combine_spec(_output=False)

        ### getting the initial polynomial coefficents
        a, b = self._get_normalisation_coeffs(self.combined_wavelengths)
        self.poly_inputs, self.fluxes_order1, self.flux_error_order1 = self.continuumfit(
            self.combined_spectrum, (self.combined_wavelengths*a)+b, self.combined_errors, self.poly_ord)

        #### getting the initial profile
        self.adjust_continuum = False
        LSD_initial_profile = LSD.LSD(self)
        LSD_initial_profile.run_LSD(order=30)
        self.velocities = LSD_initial_profile.velocities
        self.initial_profile = LSD_initial_profile.profile
        self.initial_profile_errors = LSD_initial_profile.profile_errors
        self.alpha = LSD_initial_profile.alpha

        ## Setting the number of points in vgrid (k_max)
        self.k_max = len(self.initial_profile)
        self.model_inputs = np.concatenate((self.initial_profile, self.poly_inputs))

        ## Setting x, y, yerr for emcee
        self.x = self.combined_wavelengths
        self.y = self.combined_spectrum
        self.yerr = self.combined_errors

        ## Setting these normalisation factors as global variables - used in the figures below
        a, b = self._get_normalisation_coeffs(self.x)

        # Masking based off residuals
        if verbose:
            print('Residual masking...')
        # yerr, model_inputs_resi, self.residual_masks = self.residual_mask(
            # x, y, yerr, model_inputs, self.poly_ord, pix_chunk=pix_chunk, dev_perc=dev_perc, tell_lines=self.tell_lines, n_sig=n_sig, alpha=self.alpha)
        # Inputs:
        # self.x, self.y, self.yerr, self.model_inputs, self.poly
        # Sets:
        # self.model_inputs_resi
        # Modifies:
        # self.alpha, self.yerr
        self.residual_mask()

        ## Setting number of walkers and their start values(pos)
        self.ndim = len(self.model_inputs)
        self.nwalkers = self.ndim * 3
        rng = np.random.default_rng()

        ### starting values of walkers with independent variation
        sigma = 0.8 * 0.005
        self.initial_state = []
        for i in range(0, self.ndim):
            if i < self.ndim - self.poly_ord - 2:
                pos = rng.normal(self.model_inputs[i], sigma, (self.nwalkers, ))
            else:
                sigma = abs(utils.round_sig(self.model_inputs[i], 1)) / 10
                pos = rng.normal(self.model_inputs[i], sigma, (self.nwalkers, ))
            self.initial_state.append(pos)

        self.initial_state = np.array(self.initial_state)
        self.initial_state = np.transpose(self.initial_state)

        if verbose:
            t5 = time.time()
            print('Initialised in %ss'%round((t5-t0), 2))

        if self.verbose:
            print('Fitting the continuum using emcee...')

        # Determine if running in SLURM environment
        try:
            if "bash" not in os.environ.get("SLURM_JOB_NAME").lower():
                self.slurm = True
            else:
                self.slurm = False
        except:
            self.slurm = False

        # Choose the number of cores
        self.cores = cores
        if self.cores is None:
            if self.slurm:
                self.cores = int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
            else:
                self.cores = os.cpu_count()
        else:
            self.cores = cores

        if parallel:

            if verbose:
                print(f"Using {self.cores} out of {os.cpu_count()} cores for MCMC")

            # For some reason, unspecified pooling as was before (as in case of windows in the else statement)
            # leds to a hung computer. So specify mp.get_context required, default is spawn, but spawn
            # causes multiple instances of this script to rerun, causing alpha matrix calculation to be redone
            # in each child process. Therefore, fork, which is legacy mp behavior on unix, is used.
            if sys.platform != "win32":
                global_data = {"x": self.x, "y": self.y, "yerr": self.yerr, "alpha": self.alpha,
                               "k_max": self.k_max, "velocities": self.velocities}
                ctx = mp.get_context("fork")
                with ctx.Pool(processes=self.cores, initializer=mcmc_utils._init_worker, initargs=(global_data,)) as pool:
                    self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, mcmc_utils._log_probability, pool=pool)
                    self.sampler.run_mcmc(self.initial_state, self.nsteps, progress=self.verbose, store=True)

            else: # Untested. Now tested, this doesn't work, needs serious modifications to make work
                raise NotImplementedError("Parallel MCMC on Windows is not currently supported.")
                with Pool() as pool:
                    self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, mcmc_utils._log_probability, pool=pool)
                    self.sampler.run_mcmc(pos, nsteps, progress=self.verbose)

        else:
            self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.log_probability)
            self.sampler.run_mcmc(self.initial_state, self.nsteps, progress=self.verbose)

        print('MCMC run takes: %s'%(time.time()-t5))

        ## discarding all vales except the last 1000 steps.
        dis_no = int(np.floor(self.nsteps-1000))

        ## combining all walkers together
        self.flat_samples = self.sampler.get_chain(discard=dis_no, flat=True)

        ## getting the final profile and continuum values - median of last 1000 steps
        self.profile = []
        self.poly_cos = []
        self.profile_err = []
        self.poly_cos_err = []

        for i in range(self.ndim):
            mcmc = np.median(self.flat_samples[:, i])
            error = np.std(self.flat_samples[:, i])
            mcmc = np.percentile(self.flat_samples[:, i], [16, 50, 84])
            error = np.diff(mcmc)
            if i<self.k_max:
                self.profile.append(mcmc[1])
                self.profile_err.append(np.max(error))
            else:
                self.poly_cos.append(mcmc[1])
                self.poly_cos_err.append(np.max(error))

        self.profile = np.array(self.profile)
        self.profile_err = np.array(self.profile_err)

        print('Getting the final profiles...')

        # finding error for the continuuum fit
        inds = np.random.randint(len(self.flat_samples), size=50)
        conts = []
        for ind in inds:
            sample = self.flat_samples[ind]
            mdl = mcmc_utils.model_func(sample, self.combined_wavelengths, alpha=self.alpha, k_max=self.k_max)
            mdl1_temp = 0
            for i in np.arange(self.k_max, len(sample)-1):
                mdl1_temp = mdl1_temp+sample[i]*((a*self.combined_wavelengths)+b)**(i-self.k_max)
            mdl1_temp = mdl1_temp*sample[-1]
            conts.append(mdl1_temp)

        self.continuum_error = np.std(np.array(conts), axis = 0)

        # TODO! : frame_flux is always >1 for frames
        if len(self.frame_flux)>1:
            for counter in range(len(self.frame_flux)):
                self.all_frames = self._get_profiles(self.all_frames, counter)  
        else:
            self.all_frames = self._get_profiles(self.all_frames, 0)

        if return_frames:
            return self.all_frames

    def run_ACID_HARPS(self, filelist, linelist_path, velocities, order_range=None, save_path = './',
                       file_type = 'e2ds', name="test", **kwargs):
        """Accurate Continuum fItting and Deconvolution for HARPS e2ds and s1d spectra (DRS pipeline 3.5)

        Fits the continuum of the given spectra and performs LSD on the continuum corrected spectra,
        returning an LSD profile for each file given. Files must all be kept in the same folder as well
        as their corresponding blaze files. If 's1d' are being used their e2ds equivalents must also be
        in this folder. Result files containing profiles and associated errors for each order (or
        corresponding wavelength range in the case of 's1d' files) will be created and saved to a
        specified folder. It is recommended that this folder is seperate to the input files.

        Parameters
        ----------
        filelist : list of strings
            List of files. Files must come from the same observation night as continuum is fit for a combined
            spectrum of all frames. A profile and associated errors will be produced for each file specified.
        linelist_path : str
            Path to linelist. Takes VALD linelist in long or short format as input. Minimum line depth input into VALD must
            be less than 1/(3*SN) where SN is the highest signal-to-noise ratio of the spectra. 
        velocities : array
            Velocity grid for LSD profiles (in km/s). For example, use: np.arange(-25, 25, 0.82) to create
        order_range : array, optional
            Orders to be included in the final profiles. If s1d files are input, the corresponding wavelengths 
            will be considered, by default None.
        save_path : str, optional
            Path to the directory where output files will be saved, by default './'
        file_type : str, optional
            Type of the input files, either "e2ds" or "s1d", by default 'e2ds'
        **kwargs
            Additional arguments to be passed to the ACID function. See ACID function for details.

        Returns
        -------
        list
            Barycentric Julian Date for files
        list
            Profiles (in normalised flux)
        list
            Errors on profiles (in normalised flux)
        """

        self.linelist_path = linelist_path
        self.order_range = order_range
        self.velocities = velocities
        # global frame_wavelengths
        # global frame_errors
        # global sns
        # global run_name
        self.run_name = name

        if self.order_range is None:
            # Be default, class is initialised with order_range = [1] for HARPS, this part forces
            # order range to np.arange(10, 70) if not specified for the ACID HARPS function.
            # I think this is way too high though
            self.order_range = np.arange(10, 70)

        for order in self.order_range:

            print('Running for order %s/%s...'%(order-min(self.order_range)+1, max(self.order_range)-min(self.order_range)+1))

            frame_wavelengths, frames, frame_errors, sns = self.read_in_frames(order, filelist, file_type)

            self.run_ACID(frame_wavelengths, frames, frame_errors, sns, self.linelist_path, self.velocities,
                          order=order-min(self.order_range), return_frames=False, **kwargs)

        # adding into fits files for each frame
        BJDs = []
        profiles = []
        errors = []
        for frame_no in range(0, len(frames)):
            file = filelist[frame_no]
            fits_file = fits.open(file)
            hdu = fits.HDUList()
            hdr = fits.Header()
            
            for order in self.order_range:
                hdr['ORDER'] = order
                hdr['BJD'] = fits_file[0].header['ESO DRS BJD']
                if order == self.order_range[0]:
                    BJDs.append(fits_file[0].header['ESO DRS BJD'])
                hdr['CRVAL1'] = np.min(self.velocities)
                hdr['CDELT1'] = self.velocities[1] - self.velocities[0]

                profile = self.all_frames[frame_no, order-min(self.order_range), 0]
                profile_err = self.all_frames[frame_no, order-min(self.order_range), 1]

                hdu.append(fits.PrimaryHDU(data = [profile, profile_err], header = hdr))
                if save_path != 'no save':
                    month = 'August2007'
                    hdu.writeto('%s%s_%s_%s.fits'%(save_path, month, frame_no, self.run_name), output_verify='fix', overwrite='True')

            result1, result2 = self.combineprofiles(self.all_frames[frame_no, :, 0], self.all_frames[frame_no, :, 1])
            profiles.append(result1)
            errors.append(result2)

        return BJDs, profiles, errors

    @staticmethod
    def _run_legacy_ACID(*args, **kwargs):
        """Runs the legacy ACID code

        This static function runs the legacy ACID code within the ACID class.
        This is provided for backwards compatibility with previous versions of ACID.
        It is recommended to use the ACID class and its methods for new code.

        Parameters
        ----------
        *kwargs
            Positional arguments to be passed to the ACID function.
        **kwargs
            Keyword arguments to be passed to the ACID function.

        Returns
        -------
        Any
            Returns the outputs of the ACID function.
        """
        return ACID(*args, **kwargs).run_ACID()
    
    @staticmethod
    def _run_legacy_ACID_HARPS(*args, **kwargs):
        """Runs the legacy ACID_HARPS code

        This static function runs the legacy ACID_HARPS code within the ACID class.
        This is provided for backwards compatibility with previous versions of ACID.
        It is recommended to use the ACID class and its methods for new code.

        Parameters
        ----------
        *kwargs
            Positional arguments to be passed to the ACID_HARPS function.
        **kwargs
            Keyword arguments to be passed to the ACID_HARPS function.

        Returns
        -------
        Any
            Returns the outputs of the ACID_HARPS function.
        """
        return ACID(*args, **kwargs).run_ACID_HARPS()

def run_ACID(*args, **kwargs):
    """Legacy ACID function

    This function runs the legacy ACID code. This is provided for backwards compatibility with previous versions of ACID.
    It is recommended to use the ACID class and its methods for new code.

    Parameters
    ----------
    *kwargs
        Positional arguments to be passed to the ACID function.
    **kwargs
        Keyword arguments to be passed to the ACID function.

    Returns
    -------
    Any
        Returns the outputs of the ACID function.
    """
    init_params = inspect.signature(ACID.__init__).parameters
    init_keys = set(init_params.keys()) - {"self"}

    # split kwargs
    init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
    run_kwargs = {k: v for k, v in kwargs.items() if k not in init_keys}

    acid = ACID(**init_kwargs)
    return acid.run_ACID(*args, **run_kwargs)

def run_ACID_HARPS(*args, **kwargs):
    """Legacy ACID_HARPS function

    This function runs the legacy ACID_HARPS code. This is provided for backwards compatibility with previous versions of ACID.
    It is recommended to use the ACID class and its methods for new code.

    Parameters
    ----------
    *kwargs
        Positional arguments to be passed to the ACID_HARPS function.
    **kwargs
        Keyword arguments to be passed to the ACID_HARPS function.

    Returns
    -------
    Any
        Returns the outputs of the ACID_HARPS function.
    """
    init_params = inspect.signature(ACID.__init__).parameters
    init_keys = set(init_params.keys()) - {"self"}

    # split kwargs
    init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
    run_kwargs = {k: v for k, v in kwargs.items() if k not in init_keys}

    acid = ACID(**init_kwargs)
    return acid.run_ACID_HARPS(*args, **run_kwargs)

def calc_deltav(*args):
    """Legacy calc_deltav function

    This function runs the legacy calc_deltav code. This is provided for backwards compatibility with previous versions of ACID.
    It is recommended to use the ACID class and its methods for new code.

    Parameters
    ----------
    *args
        Positional arguments to be passed to the calc_deltav function.

    Returns
    -------
    Any
        Returns the outputs of the calc_deltav function.
    """
    return ACID().calc_deltav(*args)