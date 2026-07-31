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
            sparse  : bool|None             = None,
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
        sparse : bool, optional
            Whether to use sparse matrix calculations for the alpha matrix, by default True. If you use false it will use 
            the legacy method for alpha calculation which is significantly longer and more memory intensive to no real benefit.
            It is just kept for testing/backwards compatibility.
        """
        # Set class variables, taking from input data if it exists, else setting to defaults
        # TODO: Add tests for new sparse option
        self.slurm    = "SLURM_JOB_ID" in os.environ
        self.data     = data if data is not None else Data()
        self.linelist = self.data.linelist if self.data is not None else None
        self.od       = od if od is not None else self.data.config.od
        self.sparse   = sparse if sparse is not None else self.data.config.sparse
        try:
            self.config = self.data.config
        except:
            self.config = Config() # uses defaults
        self.config.update_hipri(verbose=verbose) # Update config with new values, if not None

    def run_LSD(
        self,
        wavelengths    : Array1D,
        flux           : Array1D,
        errors         : Array1D,
        sn             : Scalar,
        linelist       : Array2D|str|LineList|dict|None = None,
        velocities     : Array1D|None                   = None,
        alpha          : Array2D|Array3D|None                   = None,
        use_ions       : bool|None = None,
        skip_warnings  : bool = False,
        ) -> None:
        """Runs the LSD algorithm to extract the average line profile from the observed spectrum.

        Parameters
        ----------
        wavelengths : :py:type:`Array1D`
            Array of wavelengths of the observed spectrum in Angstroms
        flux : :py:type:`Array1D`
            Array of flux values corresponding to the wavelengths (in linear space, and should be continuum normalized)
        errors : :py:type:`Array1D`
            Array of error values corresponding to the flux
        sn : :py:type:`Scalar`
            Signal-to-noise ratio of the observed spectrum
        linelist : :py:type:`Array2D | str | LineList | dict | None`, optional
            Linelist to use for LSD, should follow the same format as :py:class:`Acid`. 
            If None, uses the linelist already stored in the class, if it exists, by default None.
        velocities : :py:type:`Array1D`, optional
            Array of velocities corresponding to the observed spectrum.
            If the class was not initialised with an Acid instance, this is required, by default None
        alpha : :py:type:`Array2D | Array3D | None`, optional
            Precomputed alpha matrix, if already calculated and you want to skip directly to the Cholesky 
            decomposition and solving for the profile, by default None
        """

        # Ensure inputs are numpy arrays
        wavelengths = np.array(wavelengths)
        flux = np.array(flux)
        errors = np.array(errors)
        
        # Ensure dimensions match
        if not wavelengths.shape == flux.shape == errors.shape:
            raise ValueError("Input wavelengths, flux, and errors must have the same shape.")
        self.n_wavelengths = len(wavelengths)

        # Set velocities either from inputs or from Data class if initialised with Acid instance
        self.data.velocities = velocities if velocities is not None else self.data.velocities
        if self.data.velocities is None:
            raise ValueError("Velocities must be provided either as an argument to run_LSD or when initialising the class with an Acid instance.")
        self.n_velocities = len(self.data.velocities)
        
        # If alpha is input check its shape matches the input wavelengths and velocities
        if alpha is not None:
            alpha = np.asarray(alpha)
            if alpha.ndim == 2:
                if alpha.shape[0] != self.n_wavelengths:
                    raise ValueError(f"Input alpha first dimension {alpha.shape[0]} must match n_wavelengths={self.n_wavelengths}.")
                if alpha.shape[1] % self.n_velocities != 0:
                    raise ValueError(f"Input 2D alpha second dimension {alpha.shape[1]} must be a multiple of n_velocities={self.n_velocities}.")
            elif alpha.ndim == 3:
                if alpha.shape[1:] != (self.n_wavelengths, self.n_velocities):
                    raise ValueError(f"Input 3D alpha shape {alpha.shape} does not match expected (n_ions, {self.n_wavelengths}, {self.n_velocities}).")
            else:
                raise ValueError("Input alpha must be either 2D or 3D.")

        # Unpack the linelist stored in data
        self.data.linelist = linelist # Raises if no linelist available, overwrites if input
        wavelengths_linelist, depths_linelist, ions_linelist = self.data.linelist

        # Clip linelist to wavelength range of spectrum
        wavelengths_linelist, depths_linelist, ions_linelist = utils.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist, ions_linelist)
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
        wavelengths_linelist, depths_linelist, ions_linelist = self.sn_clip(wavelengths_linelist, depths_linelist, ions_linelist, sn, skip_warnings)

        # At this point we our mask for points with negative fluxes and large masked errors and nans
        self.mask = (flux > 0) & (errors < 1e11) & (errors > 0) & ~np.isnan(flux) & ~np.isnan(errors)

        # Convert to optical depth space for the linelist and the spectrum if needed, and convert errors accordingly
        if self.od:
            flux, errors, depths_linelist = utils.flux_to_od(flux, errors, depths_linelist)
        else:
            flux -= 1

        # Calculates alpha in optical depth, selects lines greater than 1/(3*sn)
        self.alpha_ion = None
        self.unique_ions = None
        self.n_ions = 1
        self.n_velocities = len(self.data.velocities)
        if alpha is None:
            if ions_linelist is None or use_ions is False:
                self.alpha = self.calc_alpha(wavelengths, wavelengths_linelist, depths_linelist, self.data.velocities, verbose=self.config.verbose)
            else:
                self.ions_grouped = self.group_sparse_ions(
                    ions_linelist,
                    min_lines_per_ion=20,
                    other_label="other",
                )
                self.alpha_ion, self.unique_ions = self.calc_alpha_ion(
                    wavelengths,
                    self.data.velocities,
                    wavelengths_linelist,
                    depths_linelist,
                    self.ions_grouped,
                    verbose=self.config.verbose,
                )
                self.n_ions = len(self.unique_ions)
                self.n_velocities = len(self.data.velocities)
                self.alpha = self.flatten_alpha(self.alpha_ion)
        else:
            self.alpha = np.asarray(alpha)

            if self.alpha.ndim == 3:
                self.n_ions = self.alpha.shape[0]
                self.n_velocities = self.alpha.shape[2]
                self.unique_ions = np.arange(self.n_ions)  # Placeholder if ions are not provided
                self.alpha_ion = self.alpha
                self.alpha = self.flatten_alpha(self.alpha_ion)
            
            elif alpha.ndim == 2:
                self.alpha = alpha
                self.n_velocities = len(self.data.velocities)

                if self.alpha.shape[1] % self.n_velocities != 0:
                    raise ValueError(
                        f"2D alpha shape {self.alpha.shape} is incompatible with "
                        f"n_velocities={self.n_velocities}."
                    )

                self.n_ions = self.alpha.shape[1] // self.n_velocities

        ion_mode = (self.alpha_ion is not None) or (self.n_ions > 1)
        ion_mode = ion_mode and (use_ions is not False)  # Allow override

        print(self.alpha.shape)
        print(self.n_ions, self.n_velocities, self.alpha.shape, ion_mode, self.unique_ions)

        # Now solve for profile using Cholesky decomposition, independent of ion mode since alpha is flattened in both cases
        self.c_factor = self.calc_cholesky(self.alpha, errors)

        # Solve for profile and profile errors using Cholesky factors
        self.profile_flat, self.profile_errors_flat, self.cov_z = self.solve_z(self.alpha, flux, errors, self.c_factor, return_error=True, return_cov=True)

        if ion_mode:
            self.profile = self.profile_flat.reshape(self.n_ions, self.n_velocities)
            self.profile_errors = self.profile_errors_flat.reshape(self.n_ions, self.n_velocities)
        else:
            self.profile = self.profile_flat
            self.profile_errors = self.profile_errors_flat

        self.forward_model = self.alpha @ self.profile_flat
        self.forward_model_errors = np.sqrt((self.alpha**2) @ (self.profile_errors_flat**2))
        # self.forward_model_errors = np.sqrt(np.sum((self.alpha @ self.cov_z) * self.alpha, axis=1))

        # Convert profile back to flux if needed
        if self.od:
            self.profile_F_flat, self.profile_errors_F_flat, self.cov_z_F = utils.od_to_flux(
                self.profile_flat,
                self.profile_errors_flat,
                cov_matrix=self.cov_z,
            )

            if ion_mode:
                self.profile_F = self.profile_F_flat.reshape(self.n_ions, self.n_velocities)
                self.profile_errors_F = self.profile_errors_F_flat.reshape(self.n_ions, self.n_velocities)
            else:
                self.profile_F = self.profile_F_flat
                self.profile_errors_F = self.profile_errors_F_flat

            self.forward_model, self.forward_model_errors = utils.od_to_flux(
                self.forward_model,
                self.forward_model_errors,
            )

        else:
            self.profile_flat += 1

            if ion_mode:
                self.profile = self.profile_flat.reshape(self.n_ions, self.n_velocities)
                self.profile_errors = self.profile_errors_flat.reshape(self.n_ions, self.n_velocities)
            else:
                self.profile = self.profile_flat
                self.profile_errors = self.profile_errors_flat

            self.profile_F = self.profile
            self.profile_errors_F = self.profile_errors
            self.cov_z_F = self.cov_z
            self.forward_model += 1
            self.profile_F_flat = self.profile_flat
            self.profile_errors_F_flat = self.profile_errors_flat

        return

    def sn_clip(
            self,
            wavelengths_linelist : Array1D,
            depths_linelist      : Array1D,
            ions_linelist        : Array1D,
            sn                   : Scalar,
            skip_warnings       : bool = False,
            ) -> tuple[Array1D, Array1D, Array1D]:
        """
        Applies a signal-to-noise cut to the linelist, removing lines shallower than 1/(3*sn) as per Dolan et al (2024).

        Parameters
        ----------
        wavelengths_linelist : :py:type:`Array1D`
            Wavelengths from the linelist
        depths_linelist : :py:type:`Array1D`
            Depths from the linelist
        ions_linelist : :py:type:`Array1D`
            Ions from the linelist
        sn : :py:type:`Scalar`
            Signal-to-noise ratio threshold
        skip_warnings : bool, optional
            Whether to skip warnings about the number of lines remaining after the S/N cut,
            by default False

        Returns
        -------
        tuple[:py:type:`Array1D`, :py:type:`Array1D`, :py:type:`Array1D`]
            Clipped wavelengths, depths, and ions from the linelist
        """
        # Selecting lines deeper than 1/(3*sn)
        idx = (depths_linelist >= 1/(3*sn))
        wavelengths_linelist = wavelengths_linelist[idx]
        depths_linelist = depths_linelist[idx]
        if ions_linelist is not None:
            ions_linelist = ions_linelist[idx]

        # Analyse remaining lines
        if not skip_warnings:
            ncut = np.sum(~idx)
            nrest = np.sum(idx)
            perc = 100 * nrest / (nrest + ncut)
            if nrest == 0:
                error = SNCutError(f"No lines remain in the linelist after S/N cut. Please check your linelist and S/N value.")
                self.data.exception = error
                self.data.traceback = traceback.format_stack()
                raise error
            if self.config.verbose > 0 and not skip_warnings:
                if perc < 5:
                    print("Warning: Less than 5% of lines remain after S/N cut. Check your linelist and S/N value.")
                if self.config.verbose > 2:
                    print(f"{perc:.2f}% of lines used in LSD: {nrest} out of {nrest + ncut} remain from S/N cut.")
        return wavelengths_linelist, depths_linelist, ions_linelist

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
        wavelengths_linelist, depths_linelist = utils.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist)

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

            if verbose > 1:
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
                    u = np.empty((n_wave_block, n_line_block), dtype=np.float64)
                    np.subtract(waves[:, None], wl[None, :], out=u)
                    u *= c_kms
                    u /= wl[None, :]
                    u -= v0
                    u /= deltav

                    k0 = np.floor(u).astype(np.intp)

                    # Reuse u as frac to avoid another full temporary.
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
                if verbose > 1:
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
        velocities           : Array1D|None = None,
        wavelengths          : Array1D|None = None,
        linelist_wavelengths : Array1D|None = None,
        linelist_depths      : Array1D|None = None,
        linelist_ions        : Array1D|None = None,
        return_alpha         : bool = False,
        ): # TODO: put back return hint
        """
        Convolve your profile either using an inputted alpha matrix or by calculating one using :py:meth:`calc_alpha` 
        with the inputted wavelengths and linelist. The units of the output convolved model spectrum will match the 
        units of the input profile (ie OD or flux) and alpha matrix/linelist depths. If alpha is not input, the wavelengths 
        and linelist inputs are required to calculate the alpha matrix.
        See :py:func:`utils.flux_to_od` and :py:func:`utils.od_to_flux` for conversions.

        Parameters
        ----------
        profile : :py:type:`Array1D` | :py:type:`Array2D`
            1D or 2D array of the LSD profile to be convolved. If 2D, the first dimension should correspond to ions and the second to velocities.
            Should be in the same units as the alpha matrix (OD or flux)
        alpha : :py:type:`Array2D` | :py:type:`Array3D` | None, optional
            Precomputed alpha matrix, if already calculated and you want to skip directly to the convolution, by default None.
            Can be 2D (n_wavelengths, n_velocities) or 3D (n_ions, n_wavelengths, n_velocities). If 3D, the first dimension should correspond to ions.
        velocities : :py:type:`Array1D` | None, optional
            Array of velocities corresponding to the observed spectrum, required if alpha is not input, by default None
        wavelengths : :py:type:`Array1D` | None, optional
            Array of wavelengths of the observed spectrum, required if alpha is not input, by default None
        linelist_wavelengths : :py:type:`Array1D` | None, optional
            Array of wavelengths from the linelist, required if alpha is not input, by default None
        linelist_depths : :py:type:`Array1D` | None, optional
            Array of depths from the linelist, required if alpha is not input. Must be in the same units
            as the alpha matrix (OD or flux), by default None
        linelist_ions : :py:type:`Array1D` | None, optional
            Array of ions from the linelist, required if alpha is not input, by default None
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

            if linelist_ions is None:
                alpha = cls.calc_alpha(
                    wavelengths=wavelengths,
                    wavelengths_linelist=linelist_wavelengths,
                    depths_linelist=linelist_depths,
                    velocities=velocities,
                )
            else:
                alpha, _ = cls.calc_alpha_ion(
                    wavelengths=wavelengths,
                    velocities=velocities,
                    linelist_wavelengths=linelist_wavelengths,
                    linelist_depths=linelist_depths,
                    linelist_ions=linelist_ions,
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
            n_ions, _, n_velocities = alpha.shape

            if profile.ndim == 1:
                if profile.size != n_ions * n_velocities:
                    raise ValueError(
                        f"Flat profile length {profile.size} does not match "
                        f"n_ions*n_velocities={n_ions * n_velocities}."
                    )
                profile = profile.reshape(n_ions, n_velocities)

            if profile.shape != (n_ions, n_velocities):
                raise ValueError(
                    f"3D alpha shape {alpha.shape} requires profile shape "
                    f"{(n_ions, n_velocities)}, got {profile.shape}."
                )

            return np.einsum("inw,iw->n", alpha, profile)

        raise ValueError("Alpha matrix must be either 2D or 3D.")

    @classmethod
    def calc_alpha_ion(
        cls,
        wavelengths: Array1D,
        velocities: Array1D,
        linelist_wavelengths: Array1D,
        linelist_depths: Array1D,
        linelist_ions: Array1D,
        verbose: IntLike | bool | str | None = None,
    ) -> tuple[Array3D, Array1D]:
        """
        Build one alpha block per ion.

        Returns
        -------
        alpha_ion : Array3D
            Shape: (n_ions, n_wavelengths, n_velocities)

        unique_ions : Array1D
            Ion labels in the same order as alpha_ion.
        """
        linelist_ions = np.asarray(linelist_ions)
        unique_ions = np.unique(linelist_ions)

        alpha_blocks = []

        verbose = Config(verbose=verbose).verbose
        if verbose:
            iterator = tqdm(unique_ions, desc="Calculating alpha blocks for each ion")
        else:
            iterator = unique_ions

        for ion_label in iterator:
            idx = linelist_ions == ion_label

            alpha_i = cls.calc_alpha(
                wavelengths=wavelengths,
                wavelengths_linelist=linelist_wavelengths[idx],
                depths_linelist=linelist_depths[idx],
                velocities=velocities,
                verbose=0,
            )

            alpha_blocks.append(alpha_i)

        alpha_ion = np.stack(alpha_blocks, axis=0)
        return alpha_ion, unique_ions
    
    @staticmethod
    def flatten_alpha(alpha:Array3D) -> Array2D:
        if alpha.ndim != 3:
            raise ValueError(f"Expected 3D alpha_ion, got shape {alpha.shape}")
        return np.concatenate(alpha, axis=1)

    @staticmethod
    def group_sparse_ions(
        ions_linelist,
        min_lines_per_ion=10,
        other_label="other",
    ):
        ions_linelist = np.asarray(ions_linelist).astype(object)

        unique_ions, counts = np.unique(ions_linelist, return_counts=True)
        good_ions = unique_ions[counts >= min_lines_per_ion]

        grouped_ions = ions_linelist.copy()
        sparse = ~np.isin(grouped_ions, good_ions)
        grouped_ions[sparse] = other_label

        return grouped_ions