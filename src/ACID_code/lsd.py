from __future__ import annotations
import numpy as np
from astropy.io import  fits
import glob, psutil, os, traceback
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tqdm import tqdm
from scipy.linalg import cho_factor, cho_solve
from beartype import beartype
from . import utils
from .errors import LineListRangeError, SNCutError
from .data import Config, Data, LineList
from .utils import c_kms, IntLike, Scalar, Array1D, Array2D, Array3D


@beartype
class LSD:
    """
    Class containing all useful functions for performing least-squares deconvolution.
    This does not simultaneously fit the continuum and perform LSD (which ACID does). It is used
    for the initial parameters of the ACID mcmc run and for obtaining final profiles. It 
    can also be used as a standalone LSD implementation and for trying to do LSD without OD.
    For more details and an example, see :ref:`LSD` in the documentation.
    """
    def __init__(
            self,
            data    : object|None           = None,
            od      : bool                  = None,
            verbose : IntLike|bool|str|None = None,
        ) -> None:
        """Initialises the LSD class, optionally with a Data instance to take parameters from.

        Parameters
        ----------
        data : object | None, optional
            A data instance to draw parameters and configs from, by default None
        od : bool, optional
            Whether to perform LSD in optical depth space (True) or flux space (False), by default None.
            If None, takes from Data instance if provided, else defaults to True.
            We generally recommend always using optical depth as ACID was always intended, but you can set
            this to False if you wish to do your own testing. See :ref:`LSD` in the documentation for more details.
        verbose : :py:type:`IntLike | bool | str | None`, optional
            Verbosity level, if None, uses the :py:class:`Config` class existing value (in Data), or default of 2.
            Should follow the same format as :py:class:`Acid` verbosity. 
            Will overwrite the verbosity level in the config if a Data instance is input, by default None.
        """
        # Set class variables, taking from input data if it exists, else setting to defaults
        self.slurm             = "SLURM_JOB_ID" in os.environ
        self.data              = data if data is not None else Data()
        self.linelist          = self.data.linelist if self.data is not None else None
        self.od                = od if od is not None else self.data.config.od
        try: # try access the provided config, otherwise just use all defaults
            self.config = self.data.config
        except:
            self.config = Config() # uses defaults
        self.config.update_hipri(verbose=verbose) # Update config with new values, if not None

    def run_LSD(
        self,
        wavelengths    : Array1D|None                   = None,
        flux           : Array1D|None                   = None,
        errors         : Array1D|None                   = None,
        sn             : Scalar|None                    = None,
        key            : str|None                       = None,
        linelist       : Array2D|str|LineList|dict|None = None,
        velocities     : Array1D|None                   = None,
        alpha          : Array2D|Array3D|None           = None,
        profile_groups : Array1D|None                   = None,
        mp_lsd         : bool|None                      = None,
        sparse         : bool|None                      = None,
        skip_warnings  : bool|None                      = None,
        ) -> None:
        """Runs the LSD algorithm to extract the average line profile from the observed spectrum.

        Parameters
        ----------
        wavelengths : :py:type:`Array1D`, optional
            Array of wavelengths of the observed spectrum in Angstroms. By default None.
            Must be provided here or pulled from the Data instance (input in initialisation) with the "key" argument.
        flux : :py:type:`Array1D`, optional
            Array of flux values corresponding to the wavelengths (in linear space, and should be continuum normalized).
            Must be provided here or pulled from the Data instance (input in initialisation) with the "key" argument.
        errors : :py:type:`Array1D`, optional
            Array of error values corresponding to the flux.
            Must be provided here or pulled from the Data instance (input in initialisation) with the "key" argument.
        sn : :py:type:`Scalar`, optional
            Signal-to-noise ratio of the observed spectrum.
            Must be provided here or pulled from the Data instance (input in initialisation) with the "key" argument.
        key : str, optional
            The dictionary key to pull wavelengths, flux, errors, and sn from the Data instance provided in initialisation.
            If provided, will ignore the 4 previous parameters. By default None.
        linelist : :py:type:`Array2D | str | LineList | dict | None`, optional
            Linelist to use for LSD, should follow the same format as :py:class:`Acid`. 
            If None, uses the linelist already stored in the class, if it exists, by default None.
        velocities : :py:type:`Array1D`, optional
            Array of velocities corresponding to the observed spectrum.
            If the class was not initialised with an Acid instance, this is required, by default None
        alpha : :py:type:`Array2D | Array3D | None`, optional
            Precomputed alpha matrix, if already calculated and you want to skip directly to the Cholesky 
            decomposition and solving for the profile, by default None
        profile_groups : :py:type:`Array1D | None`, optional
            A mask for the linelist elements indicating which group they belong to. Each group is fitted with their own profiles.
            If provided, the resulting profiles will be a 2D array in with the same order as the index of the group.
            The groups should be 0 indexed, e.g. [0,0,2,0,1,0,3,...]. The shape must match the inputted linelist, otherwise,
            we attempt to cut the linelist to the input wavelength range and apply the S/N cut, after which they must match.
            If None, then just one profile is generated (as if the mask was [0,0,0,0,0,...]). By default None.
        mp_lsd : bool, optional
            An override to the automatic multi-profile LSD detection. If profile_groups is input and this has been overriden
            to False, then profile_groups is ignored and you will only receive one profile. By default None.
        sparse : bool, optional
            Whether to use "sparse" matrix calculations for the alpha matrix, by default True. If you set it to false it will use 
            the legacy method for alpha calculation which is slower and more memory intensive to no real benefit.
            It is kept mainly for testing. Note also that we are not acutally calculating a sparse matrix, instead,
            we are only calculating the contributions of the nearest neighbour velocity bins and setting the rest to 0.
            For sparse=False, it calculates the entire alpha matrix with an efficient numpy method.
        skip_warnings : bool, optional
            Override with True/False, otherwise (if None) takes from the Data instance and checks the lsd_warnings_flag.
            If True, skips warnings about the inputs.
            This function will always set the flag to True at the end of the function, by default None

        Returns
        -------
        None : Extract all the results you need from the class attributes shown below.

        Attributes
        ----------
        profile : 

        """
        if skip_warnings is None:
            skip_warnings = self.data.lsd_warnings_flag

        if key is not None:
            wavelengths = self.data.wavelengths[key]
            flux        = self.data.fitted_flux[key]
            errors      = self.data.fitted_errors[key]
            sn          = self.data.sn[key]
        elif any((
            wavelengths is None,
            flux is None,
            errors is None,
            sn is None,
        )):
            raise ValueError(f"If key is not provided; wavelengths, flux, errors, and SN must be provided.")

        sparse         = sparse if sparse is not None else self.config.sparse
        profile_groups = profile_groups if profile_groups is not None else self.data.input_profile_groups

        # Ensure inputs are numpy arrays
        wavelengths = np.array(wavelengths)
        flux = np.array(flux)
        errors = np.array(errors)
        
        # Ensure dimensions match
        if not wavelengths.shape == flux.shape == errors.shape:
            raise ValueError("Input wavelengths, flux, and errors must have the same shape.")
        self.n_wavelengths = len(wavelengths)

        # Check the flux has been at least somewhat normalised:
        fluxpercentile = np.nanpercentile(flux, 95)
        if fluxpercentile > 1.5:
            raise ValueError(f"The top 95th percentile of fluxes inputted to LSD lie above 1.5 (95th% = {fluxpercentile}).\n" \
                             f"The fluxes should be normalised.")

        # Set velocities either from inputs or from Data class if initialised with Acid instance
        self.data.velocities = velocities if velocities is not None else self.data.velocities
        if self.data.velocities is None:
            raise ValueError("Velocities must be provided either as an argument to run_LSD or when initialising the class with an Acid instance.")
        self.n_velocities = len(self.data.velocities)
        
        # If alpha is input check its shape matches the input wavelengths and velocities
        if alpha is not None:
            alpha = np.asarray(alpha)
            if alpha.ndim == 2:
                if alpha.shape != (self.n_wavelengths, self.n_velocities):
                    raise ValueError(f"Input 2D alpha shape {alpha.shape} does not match expected ({self.n_wavelengths}, {self.n_velocities}).")
            elif alpha.ndim == 3:
                if alpha.shape[1:] != (self.n_wavelengths, self.n_velocities):
                    raise ValueError(f"Input 3D alpha shape {alpha.shape} does not match expected (n_profs, {self.n_wavelengths}, {self.n_velocities}).")
            else:
                raise ValueError("Input alpha must be either 2D or 3D.")

        # Unpack the linelist stored in data
        self.data.linelist = linelist # Raises if no linelist available, overwrites if input
        wavelengths_linelist, depths_linelist = self.data.linelist
        original_wavelengths = wavelengths_linelist

        # Clip linelist to wavelength range of spectrum
        wavelengths_linelist, depths_linelist, profile_groups = self.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist, profile_groups)
        if len(wavelengths_linelist) == 0:
            error = LineListRangeError(
                "No lines in linelist are within the wavelength range of the observed spectrum.\n"
                "You may have mismatched wavelength units between linelist and spectrum or an empty linelist.\n"
                "Please check your linelist and input spectrum."
            )
            self.data.exception = error
            self.data.traceback = traceback.format_stack()
            raise error
        
        # Apply S/N cut (of 1/(3*SN)) to linelist
        wavelengths_linelist, depths_linelist, profile_groups = self.sn_clip(wavelengths_linelist, depths_linelist, sn, profile_groups, skip_warnings)

        # Save the mask that constructs the clipped linelist from the original, wavelengths are sorted already
        self.ll_mask = np.searchsorted(original_wavelengths, wavelengths_linelist)

        # Handle multi-profile groups logic
        if profile_groups is not None:
            self.profile_groups = np.asarray(profile_groups)
            if len(self.profile_groups) != len(depths_linelist):
                raise ValueError(f"profile_groups has {len(self.profile_groups)} entries but the linelist "
                                f"has {len(depths_linelist)} lines (after S/N and wavelength clipping).\n"
                                f"Please check len(profile_groups) matches len(depths_linelist). "
                                f"Also see profile_groups parameter description.")
        elif self.config.depth_group_rules is not None:
            self.profile_groups = self.group_profs_by_depth(depths_linelist, prof_group_rules=self.config.depth_group_rules)
        else:
            self.profile_groups = None

        self.data.profile_groups = self.profile_groups

        mp_lsd_mode = self.profile_groups is not None or (alpha is not None and alpha.ndim == 3)
        # Finally, we allow the mp_lsd override, this override is only a False override
        if mp_lsd is False:
            mp_lsd_mode = False

        # Convert to optical depth space for the linelist and the spectrum if needed, and convert errors accordingly
        flux, errors, depths_linelist = utils.flux_to_od(flux, errors, depths_linelist, od=self.od)
        flux -= 1 if not self.od else 0 # If in flux space, we want to fit to flux-1 as per LSD convention
        
        # Calculates alpha in optical depth, selects lines greater than 1/(3*sn)
        self.n_velocities = len(self.data.velocities)
        self.n_profs = 1 # overwritten by mp_lsd_mode with n_profs if applicable
        if alpha is None:
            if mp_lsd_mode:
                self.alpha, self.unique_prof_groups = self.calc_mp_alpha(
                    wavelengths,
                    self.data.velocities,
                    wavelengths_linelist,
                    depths_linelist,
                    self.profile_groups,
                    verbose=self.config.verbose,
                    sparse=sparse
                )
                self.n_profs = len(self.unique_prof_groups)
                self.alpha_flat = self.flatten_alpha(self.alpha)
            else:
                self.alpha = self.calc_alpha(
                    wavelengths,
                    wavelengths_linelist,
                    depths_linelist,
                    self.data.velocities,
                    verbose=self.config.verbose,
                    sparse=sparse
                )
                self.alpha_flat = self.alpha
        else:
            self.alpha = np.asarray(alpha)

            if self.alpha.ndim == 3:
                self.n_profs = self.alpha.shape[0]
                self.n_velocities = self.alpha.shape[2]
                self.unique_prof_groups = np.arange(self.n_profs)
                self.alpha_flat = self.flatten_alpha(self.alpha)

            elif self.alpha.ndim == 2:
                self.alpha_flat = self.alpha

        # Now solve for profile using Cholesky decomposition, independent of mp_lsd mode since alpha is flattened in both cases
        self.c_factor = self.calc_cholesky(self.alpha_flat, errors)

        # Solve for profile and profile errors using Cholesky factors
        self.profile_flat, self.profile_errors_flat, self.cov_z = self.solve_z(self.alpha_flat, flux, errors, self.c_factor, return_error=True, return_cov=True)

        # Profile in LSD fitting space
        if mp_lsd_mode:
            self.profile = self.profile_flat.reshape(self.n_profs, self.n_velocities)
            self.profile_errors = self.profile_errors_flat.reshape(self.n_profs, self.n_velocities)
        else:
            self.profile = self.profile_flat
            self.profile_errors = self.profile_errors_flat

        self.forward_model = self.alpha_flat @ self.profile_flat
        # self.forward_model_errors = np.sqrt((self.alpha**2) @ (self.profile_errors_flat**2))
        self.forward_model_errors = np.sqrt(np.sum((self.alpha_flat @ self.cov_z) * self.alpha_flat, axis=1))

        # Convert profile and forward model to flux space
        self.profile_F_flat, self.profile_errors_F_flat, self.cov_z_F = utils.od_to_flux(
            self.profile_flat,
            self.profile_errors_flat,
            cov_matrix=self.cov_z,
            od=self.od,
        )
        self.forward_model, self.forward_model_errors = utils.od_to_flux(
            self.forward_model,
            self.forward_model_errors,
            od=self.od,
        )

        # Flux-space LSD is fitted to flux - 1
        if not self.od:
            self.profile_F_flat += 1
            self.forward_model += 1

        if mp_lsd_mode:
            self.profile_F = self.profile_F_flat.reshape(self.n_profs, self.n_velocities)
            self.profile_errors_F = self.profile_errors_F_flat.reshape(self.n_profs, self.n_velocities)
        else:
            self.profile_F = self.profile_F_flat
            self.profile_errors_F = self.profile_errors_F_flat

        self.data.lsd_warnings_flag = True # Future calls will now skip warnings

        return

    def sn_clip(
            self,
            wavelengths_linelist : Array1D,
            depths_linelist      : Array1D,
            sn                   : Scalar,
            profile_groups       : Array1D | None = None,
            skip_warnings        : bool = False,
        ) -> tuple[Array1D, Array1D, Array1D | None]:
        """
        Applies a signal-to-noise cut to the linelist, removing lines shallower than 1/(3*sn) as per Dolan et al (2024).

        Parameters
        ----------
        wavelengths_linelist : :py:type:`Array1D`
            Wavelengths from the linelist
        depths_linelist : :py:type:`Array1D`
            Depths from the linelist
        sn : :py:type:`Scalar`
            Signal-to-noise ratio threshold
        profile_groups : :py:type:`Array1D` | None, optional
            The profile group mask, if provided. If None, no profile grouping is applied.
        skip_warnings : bool, optional
            Whether to skip warnings about the number of lines remaining after the S/N cut,
            by default False

        Returns
        -------
        tuple[:py:type:`Array1D`, :py:type:`Array1D`, :py:type:`Array1D`]
            Clipped wavelengths, depths, and profile groups from the linelist
        """
        # Selecting lines deeper than 1/(3*sn)
        idx = (depths_linelist >= 1/(3*sn))
        wavelengths_linelist = wavelengths_linelist[idx]
        depths_linelist = depths_linelist[idx]

        # Analyse remaining lines
        ncut = np.sum(~idx)
        nrest = np.sum(idx)
        perc = 100 * nrest / (nrest + ncut)
        if nrest == 0:
            error = SNCutError(f"No lines remain in the linelist after S/N cut. Please check your linelist and S/N value.")
            self.data.exception = error
            self.data.traceback = traceback.format_stack()
            raise error
        if not skip_warnings:
            if self.config.verbose > 0 and not skip_warnings:
                if perc < 5:
                    print("Warning: Less than 5% of lines remain after S/N cut. Check your linelist and S/N value.")
                if self.config.verbose > 2:
                    print(f"{perc:.2f}% of lines used in LSD: {nrest} out of {nrest + ncut} remain from S/N cut.")
        return wavelengths_linelist, depths_linelist, profile_groups[idx] if profile_groups is not None else None

    @staticmethod
    def clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist, profile_groups=None, pad=5):
        """
        Clips the linelist to only include lines within the wavelength range of the observed spectrum.
        Includes a pad either side of the wavelength range so that the wings of lines outside
        the range can also contribute to the fit.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelengths of the observed spectrum
        wavelengths_linelist : np.ndarray
            Wavelengths from the linelist
        depths_linelist : np.ndarray
            Depths from the linelist
        profile_groups : np.ndarray | None, optional
            The profile group mask.
        pad : float, optional
            Number of angstroms to pad on either side of the wavelength range. By default, 5.

        Returns
        -------
        wavelengths_linelist : np.ndarray
            Clipped wavelengths from the linelist
        depths_linelist : np.ndarray
            Clipped depths from the linelist
        profile_groups : np.ndarray | None
            Clipped profile group mask, if provided
        """
        lower, upper = np.nanmin(wavelengths)-pad, np.nanmax(wavelengths)+pad
        idx = (wavelengths_linelist >= lower) & (wavelengths_linelist <= upper)
        return wavelengths_linelist[idx], depths_linelist[idx], profile_groups[idx] if profile_groups is not None else None

    @staticmethod
    def calc_alpha(
        wavelengths          : Array1D,
        wavelengths_linelist : Array1D,
        depths_linelist      : Array1D,
        velocities           : Array1D|None,
        verbose              : IntLike|bool|str|None = None,
        sparse               : bool = True,
        ) -> Array2D:
        """
        Calculates the alpha matrix given flux and errors and a linelist.
        Note that if this function is called without using run_LSD, there is no selection of lines deeper than 1/(3*sn).
        If you still wish to do this, it needs to be done in linear space with the sn_clip function before converting to (if desired) OD space.
        The units of the alpha matrix will match the units of the input linelist.

        Parameters
        ----------
        wavelengths : :py:type:`Array1D`
            Array of wavelengths of the observed spectrum in optical depth space
        wavelengths_linelist : :py:type:`Array1D`
            Array of wavelengths from the linelist in optical depth space
        depths_linelist : :py:type:`Array1D`
            Array of depths from the linelist in optical depth space
        velocities : :py:type:`Array1D`, optional
            Array of velocities, needs to either be initialised by class with Acid instance, or input here, by default None
        verbose : :py:type:`IntLike`, :py:type:`bool`, :py:type:`str`, optional
            Verbosity level, by default None, see :py:class:`Acid` for more details on verbosity levels.
        sparse : bool, optional
            Optionally override the sparse setting with a boolean True/False, by default True.
            
        Returns
        -------
        :py:type:`Array2D`
            The alpha matrix, to be used in the Cholesky decomposition and solving for the profile. 
            The units will match the units of the input linelist (OD or flux).
        """
        verbose = Config(verbose=verbose).verbose

        # Check velocity spacing is constant
        if not np.allclose(np.diff(velocities), velocities[1] - velocities[0]):
            raise ValueError("Velocity spacing must be constant for the alpha matrix calculation.")

        # Calculate velocity pixel size
        deltav = velocities[1] - velocities[0]

        # Clip linelist to wavelength range of spectrum (again just in case this is called without run_LSD, saves memory by reducing lines)
        wavelengths_linelist, depths_linelist, _ = LSD.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist)

        # Find differences and velocities
        blankwaves = wavelengths

        n_blank = len(blankwaves)
        n_lines = len(wavelengths_linelist)
        n_vel = len(velocities)

        if n_vel < 2:
            raise ValueError("At least two velocity bins are required.")

        v0 = velocities[0]

        if sparse:

            # Sparse triangular interpolation.
            # This still returns a dense alpha matrix, so alpha itself may be large.
            available_memory = utils.get_available_memory()

            alpha_bytes = n_blank * n_vel * np.dtype(np.float64).itemsize

            if alpha_bytes > 0.8 * available_memory:
                raise MemoryError(
                    f"Output alpha matrix alone requires {alpha_bytes/1024**3:.2f} GB, "
                    f"but only {available_memory/1024**3:.2f} GB appears to be available.\n"
                    f"This exception should only be tripped if you have extremely low memory "
                    f"or if you are calculating an enormous alpha matrix."
                )

            alpha = np.zeros((n_blank, n_vel), dtype=np.float64)

            # Re-check after allocating alpha.
            available_memory = utils.get_available_memory()
            work_memory = max(1, available_memory // 2)

            # Conservative estimate for temporary arrays per wavelength-line pair:
            # u/frac, k0, k1, rows, weights, masks, plus indexing overhead.
            bytes_per_pair = 96
            max_pairs = max(1, work_memory // bytes_per_pair)

            # Prefer blocking over lines while keeping all wavelengths together.
            # If even one line over all wavelengths is too big, also block wavelengths.
            if n_blank <= max_pairs:
                wave_block = n_blank
                line_block = max(1, min(n_lines, max_pairs // n_blank))
            else:
                line_block = 1
                wave_block = max(1, min(n_blank, max_pairs))

            line_range = range(0, n_lines, line_block)

            if verbose > 1 and len(line_range) > 1:
                line_range = tqdm(line_range, desc="Calculating sparse alpha matrix")

            for line_start in line_range:
                line_end = min(line_start + line_block, n_lines)

                wl = wavelengths_linelist[line_start:line_end]
                dep = depths_linelist[line_start:line_end]

                for wave_start in range(0, n_blank, wave_block):
                    wave_end = min(wave_start + wave_block, n_blank)

                    waves = blankwaves[wave_start:wave_end]
                    n_wave_block = len(waves)
                    n_line_block = len(wl)

                    # Compute u = (vel - v0) / deltav without forming separate diff/vel arrays.
                    # u is the velocity bin position
                    u = np.empty((n_wave_block, n_line_block), dtype=np.float64)
                    np.subtract(waves[:, None], wl[None, :], out=u)
                    u *= c_kms
                    u /= wl[None, :]
                    u -= v0
                    u /= deltav

                    k0 = np.floor(u).astype(np.intp)

                    # Reuse u as frac for more temporary memory savings.
                    frac = u
                    frac -= k0
                    k1 = k0 + 1

                    rows = np.repeat(np.arange(wave_start, wave_end, dtype=np.intp), n_line_block)

                    k0_flat = k0.ravel()
                    k1_flat = k1.ravel()

                    # Contribution to lower neighbour.
                    w0 = 1.0 - frac
                    w0 *= dep[None, :]
                    w0 = w0.ravel()
                    mask0 = (k0_flat >= 0) & (k0_flat < n_vel)
                    np.add.at(alpha,(rows[mask0], k0_flat[mask0]), w0[mask0])

                    # Contribution to upper neighbour.
                    w1 = frac * dep[None, :]
                    w1 = w1.ravel()
                    mask1 = (k1_flat >= 0) & (k1_flat < n_vel)
                    np.add.at(alpha, (rows[mask1], k1_flat[mask1]), w1[mask1])
    
        else:

            # Get memory available depnding on whether were on slurm or not
            available_memory = utils.get_available_memory() # in bytes
            mat_size = len(wavelengths_linelist) * len(velocities) * len(blankwaves) * 8 # Matrix size in bytes
            m_available = available_memory / 2  # Available memory in bytes (divided by 2 to be safe)

            # Calculate alpha matrix in one go if it fits in memory
            if mat_size < m_available:
                diff = blankwaves[:, None] - wavelengths_linelist
                vel = c_kms * (diff / wavelengths_linelist)
                # Calculating entire alpha matrix at once, broadcasts into a 3D array of shape (n_wl, n_lines, n_vel)
                x = (vel[:, :, np.newaxis] - velocities) / deltav
                delta = np.clip(1.0 - np.abs(x), 0.0, 1.0)
                alpha = (depths_linelist[:, None] * delta).sum(axis=1)  # (n_wl, n_vel)

            # Else, calculate in blocks to save memory
            else:
                n_blank = len(blankwaves)
                n_vel   = len(velocities)
                mem_size = available_memory // 2
                bytes_per_row = n_blank * n_vel * 8 * 3 # *8 for float64, *3 for vel, x, delta in a row
                max_block = max(1, mem_size // bytes_per_row)
                block = int(min(max_block, len(wavelengths_linelist)))
                # Set initial alpha matrix to np.zeros
                alpha  = np.zeros((len(blankwaves), len(velocities)), dtype=np.float64)

                # Use tqdm progress bar if verbose
                if verbose > 1 and len(wavelengths_linelist) > 1:
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
                    x_blk   = (vel_blk[:, :, None] - velocities) / deltav
                    delta   = np.clip(1.0 - np.abs(x_blk), 0.0, 1.0)                    

                    alpha += (dep[:, None] * delta).sum(axis=1)
        return alpha

    @staticmethod
    def calc_cholesky(
        alpha : Array2D,
        error : Array1D,
        **kwargs,
        ) -> tuple:
        """
        Calculates the LHS Cholesky factorisation matrix given the alpha matrix and flux errors.
        The units of alpha and error should match (ie OD or flux), the output will be in the same units.

        Parameters
        ----------
        alpha : :py:type:`Array2D`
            The precomputed alpha matrix
        error : :py:type:`Array1D`
            Flux errors
        **kwargs : dict, optional
            Additional keyword arguments to pass to scipy.linalg.cho_factor. 
            Overwrite_a=False must be set by us, do not pass this as a kwarg.

        Returns
        -------
        c_factor : tuple
            Cholesky factorisation matrix and lower/upper flag, to be put straight into solve_z as `c_factor` parameter
        """
        V = 1.0 / (error ** 2) # variance vector in log space, error already in log space

        # M = αT V α,  b = αT V R
        AVA = alpha.T @ (V[:, None] * alpha)

        # Diangostics for common 1-th leading order linalg error
        # M = alpha.T @ (V[:, None] * alpha)
        # print("finite M:", np.all(np.isfinite(M)))
        # print("min diag:", np.min(np.diag(M)))
        # print("rank:", np.linalg.matrix_rank(M), " / ", M.shape[0])
        # col_norm = np.linalg.norm(np.sqrt(V)[:, None] * alpha, axis=0)
        # print("zero columns:", np.sum(col_norm == 0))

        # Cholesky factorisation of M
        # print(AVA.shape)
        # plt.imshow(AVA)
        # plt.colorbar()
        # plt.show()
        # plt.imshow(alpha)
        # plt.colorbar()
        # plt.show()
        # import sys
        # sys.exit()
        c_factor = cho_factor(AVA, overwrite_a=False, **kwargs)
        return c_factor

    @staticmethod
    def solve_z(
        alpha        : Array2D,
        flux         : Array1D,
        error        : Array1D,
        c_factor     : tuple,
        return_error : bool = True,
        return_cov   : bool = False,
        **kwargs,
        ) -> tuple:
        """
        Solves for the LSD profile and its errors using the Cholesky factors. 
        All units should match between alpha, flux, and error (ie all in OD or all in flux).
        Returns the profile in the same units.

        Parameters
        ----------
        alpha : :py:type:`Array2D`
            The precomputed alpha matrix
        flux : :py:type:`Array1D`
            The observed flux values in optical depth space
        error : :py:type:`Array1D`
            The flux errors in optical depth space
        c_factor : tuple
            Cholesky factorisation matrix and lower/upper flag, to be put straight into 
            scipy.linalg.cho_solve as `c_factor` parameter
        return_error : bool, optional
            Whether to calculate and return the profile errors along with the
            profile, by default True
        return_cov : bool, optional
            Whether to return the full covariance matrix instead of just the errors, by default False
        **kwargs : dict, optional
            Additional keyword arguments to pass to both scipy.linalg.cho_solve calls
            (one for the profile, one for the covariance matrix)
            Overwrite_b=False is already set by default, do not pass this as a kwarg.

        Returns
        -------
        profile, profile_errors, cov_z : tuple
            LSD profile and its errors (if return_error is True) and covariance matrix (if return_cov is True)
        """
        V = 1.0 / (error ** 2) # variance vector in log space, error already in log space
        R = flux         # R matrix in log space

        # M = αT V α,  b = αT V R
        AVR = alpha.T @ (V * R)

        # z = M⁻¹ b
        profile = cho_solve(c_factor, AVR, overwrite_b=False, **kwargs)

        # Find error, cov(z) = M⁻¹, take diagonal
        if return_error or return_cov:
            AVA = alpha.T @ (V[:, None] * alpha)
            cov_z = cho_solve(c_factor, np.eye(AVA.shape[0]), overwrite_b=False, **kwargs)
            profile_errors = np.sqrt(np.diag(cov_z))
            if return_cov:
                return profile, profile_errors, cov_z
            else:
                return profile, profile_errors
        else:
            return profile

    @classmethod
    def convolve_profile(
        cls,
        profile              : Array1D|Array2D,
        alpha                : Array2D|Array3D|None = None,
        profile_groups       : Array1D|None = None,
        velocities           : Array1D|None = None,
        wavelengths          : Array1D|None = None,
        linelist_wavelengths : Array1D|None = None,
        linelist_depths      : Array1D|None = None,
        return_alpha         : bool = False,
        ) -> Array1D|Array2D|tuple:
        """
        Convolve your profile either using an inputted alpha matrix or by calculating one using :py:meth:`calc_alpha` 
        with the inputted wavelengths and linelist. The units of the output convolved model spectrum will match the 
        units of the input profile (ie OD or flux) and alpha matrix/linelist depths. If alpha is not input, the wavelengths 
        and linelist inputs are required to calculate the alpha matrix.
        See :py:func:`utils.flux_to_od` and :py:func:`utils.od_to_flux` for conversions.

        Parameters
        ----------
        profile : :py:type:`Array1D` | :py:type:`Array2D`
            1D or 2D array of the LSD profile to be convolved. If 2D, the first dimension should correspond to the profile groups and the second to velocities.
            Should be in the same units as the alpha matrix (OD or flux)
        alpha : :py:type:`Array2D` | :py:type:`Array3D` | None, optional
            Precomputed alpha matrix, if already calculated and you want to skip directly to the convolution, by default None.
            Can be 2D (n_wavelengths, n_velocities) or 3D (n_profs, n_wavelengths, n_velocities). If 3D, the first dimension should correspond to profile groups.
        profile_groups : :py:type:`Array1D` | None, optional
            Profile groups to calculate the alpha matrix if not input. If None, the alpha matrix to be calculated (if needed) will not be grouped.
        velocities : :py:type:`Array1D` | None, optional
            Array of velocities corresponding to the observed spectrum, required if alpha is not input, by default None
        wavelengths : :py:type:`Array1D` | None, optional
            Array of wavelengths of the observed spectrum, required if alpha is not input, by default None
        linelist_wavelengths : :py:type:`Array1D` | None, optional
            Array of wavelengths from the linelist, required if alpha is not input, by default None
        linelist_depths : :py:type:`Array1D` | None, optional
            Array of depths from the linelist, required if alpha is not input. Must be in the same units
            as the alpha matrix (OD or flux), by default None
        return_alpha : bool, optional
            Whether to return the calculated alpha matrix along with the convolved model spectrum, by default False
        
        Returns
        -------
        model_spectrum : :py:type:`Array1D` | :py:type:`Array2D`
            The convolved model spectrum, in the same units as the input profile and alpha matrix/linelist depths.
            If the input profile is 1D, the output will be 1D. If the input profile is 2D, the output will be 2D with shape (n_profs, n_wavelengths).
        alpha : :py:type:`Array2D` | :py:type:`Array3D`
            The calculated alpha matrix, only returned if return_alpha is True. Will be 2D 
            (n_wavelengths, n_velocities) or 3D (n_profs, n_wavelengths, n_velocities) 
            depending on whether profile_groups is provided.
        """

        if alpha is None:
            if (
                velocities is None
                or wavelengths is None
                or linelist_wavelengths is None
                or linelist_depths is None
            ):
                raise ValueError(
                    "If alpha is not input, velocities, wavelengths, "
                    "linelist_wavelengths, and linelist_depths are required."
                )

            if profile_groups is not None:
                alpha, _ = cls.calc_mp_alpha(
                    wavelengths,
                    velocities,
                    linelist_wavelengths,
                    linelist_depths,
                    profile_groups,
                )

            alpha = cls.calc_alpha(
                wavelengths=wavelengths,
                wavelengths_linelist=linelist_wavelengths,
                depths_linelist=linelist_depths,
                velocities=velocities,
            )

        model_spectrum = cls.dot_alpha_and_profile(alpha, profile)

        if return_alpha:
            return model_spectrum, alpha
        else:
            return model_spectrum

    @staticmethod
    def dot_alpha_and_profile(alpha, profile):
        alpha = np.asarray(alpha)
        profile = np.asarray(profile)

        if alpha.ndim == 2:
            profile_flat = profile.reshape(-1)

            if alpha.shape[1] != profile_flat.size:
                raise ValueError(
                    f"2D alpha shape {alpha.shape} is incompatible with "
                    f"profile shape {profile.shape}."
                )

            return alpha @ profile_flat

        if alpha.ndim == 3:
            n_profs, _, n_velocities = alpha.shape

            if profile.ndim == 1:
                if profile.size != n_profs * n_velocities:
                    raise ValueError(
                        f"Flat profile length {profile.size} does not match "
                        f"n_profs*n_velocities={n_profs * n_velocities}."
                    )
                profile = profile.reshape(n_profs, n_velocities)

            if profile.shape != (n_profs, n_velocities):
                raise ValueError(
                    f"3D alpha shape {alpha.shape} requires profile shape "
                    f"{(n_profs, n_velocities)}, got {profile.shape}."
                )

            return np.einsum("inw,iw->n", alpha, profile)

        raise ValueError("Alpha matrix must be either 2D or 3D.")

    @classmethod
    def calc_mp_alpha(
        cls,
        wavelengths: Array1D,
        velocities: Array1D,
        linelist_wavelengths: Array1D,
        linelist_depths: Array1D,
        profile_groups: Array1D,
        sparse: bool = True,
        verbose: IntLike|bool|str|None = None,
        ) -> tuple[Array3D, Array1D]:
        """
        Build one alpha block per line-depth profile group.

        Returns
        -------
        mp_alpha : Array3D
            Shape: (n_profs, n_wavelengths, n_velocities)

        unique_groups : Array1D
            Profile group labels in the same order as mp_alpha.
        """
        profile_groups = np.asarray(profile_groups)

        if len(profile_groups) != len(linelist_depths):
            raise ValueError("profile_groups must have the same length as the linelist.")

        unique_profile_groups = np.unique(profile_groups)
        alpha_blocks = []

        verbose = Config(verbose=verbose).verbose
        if verbose > 1:
            iterator = tqdm(unique_profile_groups, desc="Calculating alpha blocks for each profile group")
        else:
            iterator = unique_profile_groups
    
        for group in iterator:
            idx = profile_groups == group
            alpha_i = cls.calc_alpha(wavelengths, linelist_wavelengths[idx], linelist_depths[idx], velocities, sparse=sparse, verbose=0)
            alpha_blocks.append(alpha_i)

        mp_alpha = np.stack(alpha_blocks, axis=0)
        return mp_alpha, unique_profile_groups

    @staticmethod
    def flatten_alpha(alpha:Array3D) -> Array2D:
        if alpha.ndim != 3:
            raise ValueError(f"Expected 3D alpha matrix, got shape {alpha.shape}")
        return np.concatenate(alpha, axis=1)

    @staticmethod
    def group_profs_by_depth(depths_linelist:Array1D, prof_group_rules:dict|None=None) -> Array1D:
        """
        Assign each line to a profile group based on its linear line depth.

        Explicit group rules specify the minimum depth for that group. Each
        group takes all available lines down to that depth, or continues
        shallower until min_lines is reached. Groups without an explicit depth
        rule split the remaining lines equally.
        """
        depths_linelist = np.asarray(depths_linelist)

        rules = prof_group_rules

        n_groups = int(rules["n_groups"])
        min_lines = int(rules["min_lines"])

        if len(depths_linelist) < n_groups*min_lines:
            raise ValueError(f"Cannot make {n_groups} profile groups with at least {min_lines} lines each from {len(depths_linelist)} lines.")
        if n_groups < 1:
            raise ValueError("n_groups must be at least 1.")
        if min_lines < 1:
            raise ValueError("min_lines must be at least 1.")

        depth_rules = {}
        for key, value in rules.items():
            if key in ["n_groups", "min_lines"]:
                continue

            group = int(key)

            if group < 0 or group >= n_groups:
                raise ValueError(f"Profile group rule {group} is outside n_groups={n_groups}.")

            depth_rules[group] = float(value)

        depth_groups = sorted(depth_rules)

        if depth_groups != list(range(len(depth_groups))):
            raise ValueError("Explicit depth rules must be consecutive starting from group 0.")

        depth_values = [depth_rules[group] for group in depth_groups]
        if not all(depth_values[i] > depth_values[i+1] for i in range(len(depth_values)-1)):
            raise ValueError("Profile group minimum depths must decrease from deepest to shallowest group.")

        # Sort lines deepest to shallowest
        sorted_idx = np.argsort(depths_linelist)[::-1]
        sorted_depths = depths_linelist[sorted_idx]

        prof_groups = np.full(len(depths_linelist), -1, dtype=int)
        start = 0

        # Fill groups with explicit depth rules
        for group in depth_groups:

            remaining_depths = sorted_depths[start:]

            # Take all remaining lines down to this group's minimum depth
            n_take = np.sum(remaining_depths >= depth_rules[group])

            # If that is not enough, continue shallower until min_lines
            n_take = max(n_take, min_lines)

            if start + n_take > len(sorted_idx):
                raise ValueError(f"Not enough lines remaining to give profile group {group} at least {min_lines} lines.")

            end = start + n_take
            prof_groups[sorted_idx[start:end]] = group
            start = end

        # Fill groups without explicit depth rules equally
        remaining_groups = list(range(len(depth_groups), n_groups))
        remaining_idx = sorted_idx[start:]

        if len(remaining_groups) > 0:

            if len(remaining_idx) < len(remaining_groups)*min_lines:
                raise ValueError(
                    f"Only {len(remaining_idx)} lines remain for {len(remaining_groups)} profile groups with min_lines={min_lines}.\n"
                    f"Try reducing n_groups or reducing min_lines (we don't recommend reducing min_lines below 10)."
                )

            for group, idx in zip(remaining_groups, np.array_split(remaining_idx, len(remaining_groups))):
                prof_groups[idx] = group

        # If depths were specified for every group, the shallowest group
        # contains any lines remaining after its minimum requirements
        elif len(remaining_idx) > 0:
            prof_groups[remaining_idx] = n_groups - 1

        return prof_groups

    @classmethod
    def runlsd_and_store(cls, data:Data, key:str, return_cls:bool=False, **kwargs) -> None|LSD:
        """
        Extracts the wavelength, flux, etc. from the Data instance using they key.
        Runs LSD with the Config in the Data instance and stores the result.
        Used both in Acid and Result. Can also be used by the user if Data has been preconfigured.
        """
        # First check if alpha can be reused, they need to specify in the Data class if so.
        alpha = data.alpha[key] if key in data.alpha else None

        lsd = cls(data)
        lsd.run_LSD(key=key, alpha=alpha, **kwargs)
        data.c_factor[key]  = lsd.c_factor
        data.alpha[key]     = lsd.alpha
        data.forward_x[key] = data.wavelengths[key]
        data.forward_y[key] = lsd.forward_model * data.continuum[key]
        data.profile[key]   = [lsd.profile_F, lsd.profile_errors_F, lsd.cov_z_F]
        data.residuals[key] = (data.flux[key] - data.forward_y[key]) / data.forward_y[key]
        data.ll_mask[key] = lsd.ll_mask
        if return_cls:
            return lsd