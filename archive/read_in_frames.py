# Old read in frames code as part of ACID.py, basically never touched except for modifications to make work with ACID_HARPS and classes

def read_in_frames(
        self,
        order,
        filelist,
        file_type,
        directory=None,
        ):
        # read in first frame
        fluxes, wavelengths, flux_error_order, sn = LSD().blaze_correct(
            file_type, 'order', order, filelist[0], directory, 'unmasked', self.config.name, 'y')
        # fluxes, wavelengths, flux_error_order, sn, mid_wave_order, telluric_spec, overlap = LSD.blaze_correct(
        #     file_type, 'order', order, filelist[0], directory, 'unmasked', self.config.name, 'y')

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
            frames[i], frame_wavelengths[i], errors[i], sns[i] = LSD().blaze_correct(
                file_type, 'order', order, file, directory, 'unmasked', self.config.name, 'y')

            return frames, frame_wavelengths, errors, sns
        
        ### reads in each frame and corrects for the blaze function, adds the spec, errors and sn to their subsequent lists
        for i in range(len(filelist[1:])+1):

            frames, frame_wavelengths, errors, sns = task_frames(frames, errors, frame_wavelengths, sns, i)
            
        ### finding highest S/N frame, saves this as reference frame

        idx = (sns==np.max(sns))
        # global reference_wave
        reference_wave = frame_wavelengths[idx][0]
        reference_frame = frames[idx][0]
        reference_frame[reference_frame == 0] = 0.001
        reference_error = errors[idx][0]
        reference_error[reference_frame == 0] = 1e12

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
            errors[n][idx] = 1e12

        return frame_wavelengths, frames, errors, sns