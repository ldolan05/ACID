from __future__ import annotations
import numpy as np
from astropy.io import  fits
import glob, psutil, os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import LSQUnivariateSpline
from tqdm import tqdm
from scipy.linalg import cho_factor, cho_solve
from beartype import beartype
from . import utils
from .errors import LineListRangeError, SNCutError
from .data import Config, Data, LineList
from .utils import c_kms, IntLike, Scalar, Array1D, Array2D

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
            OD      : bool                  = True,
            verbose : IntLike|bool|str|None = None,
            ) -> None:
        """Initialises the LSD class, optionally with a Data instance to take parameters from.

        Parameters
        ----------
        data : object | None, optional
            A data instance to draw parameters and configs from, by default None
        OD : bool, optional
            Whether to perform LSD in optical depth space (True) or flux space (False), by default True.
            We generally recommend always using optical depth as ACID was always intended, but you can set
            this to False if you wish to do your own testing. See :ref:`LSD` in the documentation for more details.
        verbose : :py:type:`IntLike | bool | str | None`, optional
            Verbosity level, if None, uses the :py:class:`Config` class existing value (in Data), or default of 2.
            Should follow the same format as :py:class:`Acid` verbosity. 
            Will overwrite the verbosity level in the config if a Data instance is input, by default None.
        """
        # TODO: test putting non-OD inputs
        # Set class variables, taking from input data if it exists, else setting to defaults
        self.slurm    = "SLURM_JOB_ID" in os.environ
        self.data     = data if data is not None else Data()
        self.linelist = data.linelist if data is not None else None
        self.OD       = OD
        try:
            self.config = data.config
        except:
            self.config = Config() # uses defaults
        self.config.update_hipri(verbose=verbose) # Update config with new values, if not None

    def run_LSD(
        self,
        wavelengths : Array1D,
        flux        : Array1D,
        errors      : Array1D,
        sn          : Scalar,
        linelist    : Array2D|str|LineList|dict|None = None,
        velocities  : Array1D|None                   = None,
        alpha       : Array2D|None                   = None,
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
        alpha : :py:type:`Array2D` | None, optional
            Precomputed alpha matrix, if already calculated and you want to skip directly to the Cholesky 
            decomposition and solving for the profile, by default None
        """
        # Ensure inputs are numpy arrays
        wavelengths = np.asarray(wavelengths)
        flux = np.asarray(flux)
        errors = np.asarray(errors)

        # Ensure dimensions match
        if not wavelengths.shape == flux.shape == errors.shape:
            raise ValueError("Input wavelengths, flux, and errors must have the same shape.")       

        # Set linelist and velocities either from inputs or from Data class if initialised with Acid instance
        self.data.velocities = velocities if velocities is not None else self.data.velocities
        if self.data.velocities is None:
            raise ValueError("Velocities must be provided either as an argument to run_LSD or when initialising the class with an Acid instance.")
        self.data.set_linelist(linelist) # Raises if no linelist available, overwrites if input

        # Unpack the linelist stored in data
        wavelengths_linelist, depths_linelist = self.data.linelist

        # Clip linelist to wavelength range of spectrum
        wavelengths_linelist, depths_linelist = utils.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist)
        if len(wavelengths_linelist) == 0:
            raise LineListRangeError(f"No lines in linelist are within the wavelength range of the observed spectrum. \n"\
                                     f"You may have mismatched wavelengths units between linelist and spectrum or an empty linelist.\n"\
                                     f"Please check your linelist and input spectrum.")

        # Apply S/N cut (of 1/(3*SN)) to linelist
        wavelengths_linelist, depths_linelist = self.sn_clip(wavelengths_linelist, depths_linelist, sn)

        # Convert to optical depth space for the linelist and the spectrum if needed, and convert errors accordingly
        if self.OD:
            flux, errors, depths_linelist = utils.flux_to_od(flux, errors, depths_linelist)

        # Calculates alpha in optical depth, selects lines greater than 1/(3*sn)
        if alpha is None:
            self.alpha = self.calc_alpha(wavelengths, wavelengths_linelist, depths_linelist)
        else:
            self.alpha = alpha

        # Now solve for profile using Cholesky decomposition
        self.c_factor = self.calc_cholesky(self.alpha, errors)

        # Solve for profile and profile errors using Cholesky factors
        self.profile, self.profile_errors, self.cov_z = self.solve_z(self.alpha, flux, errors, self.c_factor, return_cov=True)

        # Convert profile back to flux if needed
        if self.OD:
            self.profile_F, self.profile_errors_F, self.cov_z_F = utils.od_to_flux(self.profile, self.profile_errors, cov_matrix=self.cov_z)
        else:
            self.profile_F, self.profile_errors_F, self.cov_z_F = self.profile, self.profile_errors, self.cov_z

        return

    def sn_clip(
            self,
            wavelengths_linelist : Array1D,
            depths_linelist      : Array1D,
            sn                   : Scalar,
            ) -> tuple[Array1D, Array1D]:
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

        Returns
        -------
        tuple[:py:type:`Array1D`, :py:type:`Array1D`]
            Clipped wavelengths and depths from the linelist
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
            raise SNCutError(f"No lines remain in the linelist after S/N cut. Please check your linelist and S/N value.")
        if self.config.verbose > 0:
            if perc < 5:
                print("Warning: Less than 5% of lines remain after S/N cut. Check your linelist and S/N value.")
            if self.config.verbose > 2:
                print(f"{perc:.2f}% of lines used in LSD: {nrest} out of {nrest + ncut} remain from S/N cut.")
        return wavelengths_linelist, depths_linelist

    def calc_alpha(
        self,
        wavelengths          : Array1D,
        wavelengths_linelist : Array1D,
        depths_linelist      : Array1D,
        velocities           : Array1D|None = None,
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
        
        Returns
        -------
        :py:type:`Array2D`
            The alpha matrix, to be used in the Cholesky decomposition and solving for the profile. 
            The units will match the units of the input linelist (OD or flux).
        """

        # Set velocities from input or from class variable if initialised with Acid instance, raise error if not provided in either way
        if velocities is not None:
            self.data.velocities = velocities
        elif self.data.velocities is None:
            raise ValueError("Velocities must be provided either as input or initialized in the Acid instance.")

        # Check velocity spacing is constant
        if not np.allclose(np.diff(self.data.velocities), self.data.velocities[1] - self.data.velocities[0]):
            raise ValueError("Velocity spacing must be constant for the alpha matrix calculation.")

        # Calculate velocity pixel size
        deltav = self.data.velocities[1] - self.data.velocities[0]

        # Clip linelist to wavelength range of spectrum (again just in case this is called without run_LSD, saves memory by reducing lines)
        wavelengths_linelist, depths_linelist = utils.clip_wavelengths(wavelengths, wavelengths_linelist, depths_linelist)

        # Find differences and velocities
        blankwaves = wavelengths
        diff = blankwaves[:, np.newaxis] - wavelengths_linelist
        vel = c_kms * (diff / wavelengths_linelist)

        # Get memory available depnding on whether were on slurm or not
        if self.slurm:
            available_memory = int(os.environ.get('SLURM_MEM_PER_NODE')) # in MB
            available_memory *= 1e6  # Convert to bytes as in the else statement below
        else:
            available_memory = psutil.virtual_memory().available
        mat_size = len(wavelengths_linelist) * len(self.data.velocities) * len(blankwaves) * 8 * 1e-9 # Matrix size in GB
        m_available = available_memory * 1e-9 / 2  # Available memory in GB (divided by 2 to be safe)

        # Calculate alpha matrix in one go if it fits in memory
        if mat_size < m_available:
            # Calculating entire alpha matrix at once, broadcasts into a 3D array of shape (n_wl, n_lines, n_vel)
            x = (vel[:, :, np.newaxis] - self.data.velocities) / deltav
            delta = np.clip(1.0 - np.abs(x), 0.0, 1.0)
            alpha = (depths_linelist[:, None] * delta).sum(axis=1)  # (n_wl, n_vel)

        # Else, calculate in blocks to save memory
        else:
            n_blank = len(blankwaves)
            n_vel   = len(self.data.velocities)
            mem_size = available_memory // 2
            bytes_per_row = n_blank * n_vel * 8 * 3 # *8 for float64, *3 for vel, x, delta in a row
            max_block = max(1, mem_size // bytes_per_row)
            block = int(min(max_block, len(wavelengths_linelist)))
            # Set initial alpha matrix to np.zeros
            alpha  = np.zeros((len(blankwaves), len(self.data.velocities)), dtype=np.float64)

            # Use tqdm progress bar if verbose
            if self.config.verbose>1:
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
                x_blk   = (vel_blk[:, :, None] - self.data.velocities) / deltav
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
        profile              : Array1D,
        profile_errors       : Array1D,
        alpha                : Array2D|None = None,
        velocities           : Array1D|None = None,
        wavelengths          : Array1D|None = None,
        linelist_wavelengths : Array1D|None = None,
        linelist_depths      : Array1D|None = None,
        ) -> tuple[Array1D, Array1D]:
        """
        Convolve your profile either using an inputted alpha matrix or by calculating one using :py:meth:`calc_alpha` 
        with the inputted wavelengths and linelist. The units of the output convolved model spectrum will match the 
        units of the input profile (ie OD or flux) and alpha matrix/linelist depths. If alpha is not input, the wavelengths 
        and linelist inputs are required to calculate the alpha matrix.
        See :py:func:`utils.flux_to_od` and :py:func:`utils.od_to_flux` for conversions.

        Parameters
        ----------
        profile : :py:type:`Array1D`
            The LSD profile to be convolved, in the same units as the alpha matrix (OD or flux)
        profile_errors : :py:type:`Array1D`
            The errors on the LSD profile, in the same units as the alpha matrix (OD or flux)
        alpha : :py:type:`Array2D` | None, optional
            Precomputed alpha matrix, if already calculated and you want to skip directly to the convolution, by default None
        velocities : :py:type:`Array1D` | None, optional
            Array of velocities corresponding to the observed spectrum, required if alpha is not input, by default None
        wavelengths : :py:type:`Array1D` | None, optional
            Array of wavelengths of the observed spectrum, required if alpha is not input, by default None
        linelist_wavelengths : :py:type:`Array1D` | None, optional
            Array of wavelengths from the linelist, required if alpha is not input, by default None
        linelist_depths : :py:type:`Array1D` | None, optional
            Array of depths from the linelist, required if alpha is not input. Must be in the same units
            as the alpha matrix (OD or flux), by default None
        """

        if alpha is None:
            if velocities is None or wavelengths is None or linelist_wavelengths is None or linelist_depths is None:
                raise ValueError("If alpha matrix is not input, velocities, wavelengths, linelist_wavelengths, and " \
                "linelist_depths must all be provided to calculate the alpha matrix.")
            cls.__init__(cls)
            alpha = cls.calc_alpha(cls, wavelengths, linelist_wavelengths, linelist_depths, velocities, verbose=2)

        model_spectrum = alpha @ profile
        model_errors = np.sqrt((alpha**2) @ (profile_errors**2))

        return model_spectrum, model_errors
