from __future__ import annotations
import sys
from time import time
import numpy as np
import os, functools, inspect, warnings
from collections.abc import Callable, Iterator
from typing import Any, TYPE_CHECKING
from emcee import EnsembleSampler
from beartype import beartype
from scipy.interpolate import interp1d
from numpy.polynomial import polynomial as P
from .lsd import LSD
from . import utils
from .diagnostics.errors import *
from .data import Data
from .plot import Plot, _require_sampler
from .utils import IntLike, Scalar, Array1D, Array2D, Array3D
from .diagnostics.warnings import *
from .diagnostics.logging import get_logger
from .diagnostics.output import with_output_settings

if TYPE_CHECKING:
    import ACID_code

try:
    from dynesty.sampler import Sampler
except ImportError:
    Sampler = None

logger = get_logger(__name__)

def _require_profiles(method:Callable[..., Any]) -> Callable[..., Any]:
    # Check the original parameters before the state guard wraps them in *args/**kwargs.
    method = beartype(method)
    # Final products require processed results; intermediate keys can be plotted directly.
    sig = inspect.signature(method)
    @functools.wraps(method)
    def wrapper(self, *args:Any, **kwargs:Any) -> Any:
        key = sig.bind_partial(self, *args, **kwargs).arguments.get("key", "final")
        if key == "final" and not self.data.complete: # complete is flag for if profiles have been made
            name = method.__qualname__
            if self.sampler is not None:
                # This should very rarely be triggered.
                warnings.warn(f"Note: The Result object was created without the profiles processed. " \
                    f"Running {name} requires all results to be processed, " \
                    "so process_results() will now be called...", ACIDStateWarning, stacklevel=3)
                self.process_results()
            else:
                error = f"Cannot call {name}. The profiles attribute is not available, and no " \
                "sampler object is available to process results. Please pass an Acid/Data " \
                "instance after running ACID to the results init."
                raise ACIDResultError(error)
        return method(self, *args, **kwargs)
    return wrapper

@beartype
class Result(Plot):
    """
    Handles stored results and MCMC result processing. Completed Data can be used without a sampler.
    Inherits Plot, with plot_* aliases retained for compatibility.
    Methods decorated with _require_sampler need one to process or inspect samples.
    """

    def __init__(
        self,
        data            : "Data | ACID_code.acid.Acid | ACID_code.result.Result",
        sampler         : EnsembleSampler|Sampler|None                           = None,  # type:ignore
        process_results : bool                                                   = True,
        verbose         : IntLike|bool|str|None                                  = None,
        ) -> None:
        """
        Initialize the Result class

        Parameters
        ----------
        data : :py:class:`~ACID_code.Data` | :py:class:`~ACID_code.Acid` | :py:class:`~ACID_code.Result`
            An Acid, Result or Data object (contained in Acid class). If an Acid
            object is provided, all other arguments are taken from there. If a Data object is 
            provided, a sampler can be provided in the second argument. If a sampler object 
            is provided, it will be used as the sampler, but all other attributes will need 
            to be set manually for the Result object to be fully functional.
        sampler : :py:class:`emcee.EnsembleSampler` | :py:class:`dynesty.sampler.Sampler`, optional
            Sets and overwrites the sampler in the Data object with this if provided, by default None. 
        process_results : bool, optional
            Whether to process the results from the Acid object upon initialisation, by default True.
            If False, the profiles attribute will not be available until :py:meth:`~ACID_code.Result.process_results` is called.
            The process_results functions does a LSD call, which can be skipped to save time and use
            the Result object for methods that do not require the profiles attribute, such as 
            continue_sampling() or plot_walkers(). This requires a Data object with the necessary attributes, 
            and a sampler object in the initialisation, or an Acid object with the necessary attributes already set.
            By default, True.
        verbose : :py:type:`~ACID_code.IntLike` | :py:class:`bool` | :py:class:`str`, optional
            Verbosity level, works exactly the same as :py:class:`~ACID_code.Acid`, if not provided
            defaults to provided :py:class:`~ACID_code.Acid`/:py:class:`~ACID_code.Data` class verbosity (which itself defaults to 2).
            Overwrites any value passed trough the Data object.
        """
        super().__init__(data)

        # Handle config and verbose options
        self.config.verbose = verbose # property overwrites or handles if verbose input was None

        # Handle the sampler if input, initiate if one exists
        self.sampler = sampler if sampler is not None else self.sampler # update sampler if provided, otherwise keep the same
        if self.sampler is not None:
            self.initiate_sampler(self.sampler) # set internal variables based on sampler, sets sampler_initialiated to True

        if not self.data.complete:
            if process_results:
                if self.sampler is None:
                    raise ACIDStateError("Cannot process results without a sampler. Please provide a sampler in the initialisation or set process_results=False.")
                else:
                    self.process_results()
            else:
                warnings.warn("Warning: Results not processed. Profiles attribute will not be available until " \
                "Result.process_results() is called or passed through a method.", ACIDStateWarning, stacklevel=2)
        elif self.sampler is None:
            warnings.warn(f"Warning: No sampler provided or found in Data object. \n" \
            f"Some methods will not work unless a sampler is provided as a parameter or if Result.initiate_sampler(sampler) is called.",
            ACIDStateWarning, stacklevel=2)

    @_require_sampler
    @with_output_settings
    def process_results(self) -> None:
        """
        Processes the MCMC sampler results to obtain the final LSD profiles and continuum fit, and errors on both.
        This is effectively the final step in the ACID pipeline, and must be run before the profiles attribute is available. 
        This is automatically called if process_results is True during :py:class:`~ACID_code.Acid` initialization.
        This function is stored here instead of the Acid class because it is not necessary to have the final profiles to use some of the
        methods contained within this class.
        """
        t0 = time()

        # Obtain flattened samples
        if self.dynesty:
            flat_samples = self.sampler.results.samples_equal()
        else:
            flat_samples = self.sampler.get_chain(discard=self.burnin, thin=self.thin, flat=True)
        if len(flat_samples) == 0:
            raise ACIDStateError("No posterior samples remain after ACID's automatic sample selection. "
                                 "Check the convergence warnings and try continuing the MCMC run for longer. "
                                 "If this persists, report it as an ACID issue.")

        # May be used in a future update for more accurate error estimation
        # Get approximate 1 sigma errors on the parameters using the 16th and 84th percentiles
        # quartiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
        # param_errors_upanddown = np.diff(quartiles, axis=0)
        # param_errors = np.max(param_errors_upanddown, axis=0)
        # TODO: Is this the right treatment of errors? What about the intuition of LSD errors being more like an absolute_sigma=True
        # rather than false?

        # Getting the final profile and continuum values
        # Slice the samples to get the poly_coeffs, as the profile params are included in the non-determinstic model
        n_profile_params = 0
        if not self.config.deterministic_profile:
            n_profile_params = self.data.alpha["mcmc"].shape[1]

        # med_poly_coeffs = quartiles[1, n_profile_params:]
        # med_poly_coeffs_err = param_errors[n_profile_params:] # may be used in a future update
        all_poly_coeffs = flat_samples[:, n_profile_params:]

        logger.info('Getting the final profiles...')

        # We first run LSD on the final state of the sampler (for debugging, but this can be useful to see how masking is affecting the sampler)
        if self.config.output.debug: # debugging mode
            self.data.alpha["mcmc"] = self._get_order_alpha(self.data.wavelengths["mcmc"], "mcmc")
            _lsd = self._continuum_correct_and_runlsd("mcmc", all_poly_coeffs, return_cls=True) # alpha is recalculated
            self.data.debug["lsd_mcmc"] = _lsd.__dict__.copy()

        # Set the inputs for a final LSD call on continuum corrected, and unmasked (except line mask) spectrum
        self.data.wavelengths["final"] = self.data.wavelengths["initial"]
        self.data.flux["final"]        = self.data.flux["initial"]
        self.data.errors["final"]      = self.data.errors["initial"] # includes the line_mask set to 1e12 (and hence masked for LSD)
        self.data.sn["final"]          = self.data.sn["initial"]
        self.data.alpha["final"]       = self.data.alpha["initial"]
        if self.data.continuum_slices:
            self.data.continuum_slices["final"] = self.data.continuum_slices["initial"]

        # Final continuum correction and LSD run on the unmasked spectrum, generates the final profiles
        _lsd = self._continuum_correct_and_runlsd("final", all_poly_coeffs, return_cls=True)

        if self.config.output.debug: # save if debugging is on
            self.data.debug["lsd_final"] = _lsd.__dict__.copy()
        _lsd = None

        # We also want to have a final single profile where we force LSD to run on non-mp mode to get a "combined_profile"
        if self.data.profile_groups is not None:
            # Set the inputs, reset alpha, force LSD to use non multi-profile mode
            self.data.alpha["single_profile"] = None
            if self.data.continuum_slices:
                self.data.continuum_slices["single_profile"] = self.data.continuum_slices["initial"]
                self.data.alpha["single_profile"] = self.data.alpha["initial"].sum(axis=0)
            self.data.wavelengths["single_profile"] = self.data.wavelengths["initial"]
            self.data.flux["single_profile"] = self.data.flux["initial"]
            self.data.errors["single_profile"] = self.data.errors["initial"]
            self.data.sn["single_profile"] = self.data.sn["initial"]
            _lsd = self._continuum_correct_and_runlsd("single_profile", all_poly_coeffs, mp_lsd=False, return_cls=True)

            if self.config.output.debug: # save if debugging is on
                self.data.debug["lsd_single_profile"] = _lsd.__dict__.copy()
            _lsd = None

        profiles = []
        # Iterate through each frame to get per-frame profiles
        for i in range(len(self.data.wavelengths["input"])):

            # Get the wavelengths, flux, error, and sn for this frame
            wavelengths = self.data.wavelengths["input"][i]
            flux = self.data.flux["input"][i]
            errors = self.data.errors["input"][i]
            sn = self.data.sn["input"][i]

            # Clean NaNs
            wavelengths, flux, errors = utils.drop_invalid(wavelengths, flux, errors)

            # Reapply the line_mask to the errors
            if not self.data.continuum_slices: # joint input frames already carry each order's line mask
                line_mask = self.config.masking_lines.get_1d_mask_on_grid(wavelengths)
                errors = np.where(line_mask, 1e12, errors) # set masked errors to 1e12 (so that those points are ignored in LSD)

            # Avoid alpha recalculation if the input wavelengths are the same as the combined wavelengths, always true for single frame LSD
            if np.array_equal(wavelengths, self.data.wavelengths["final"]):
                if np.array_equal(flux, self.data.flux["final"]) and np.array_equal(errors, self.data.errors["final"]) and np.array_equal(sn, self.data.sn["final"]):
                    # in this case, the input frame is the same as the combined frame, so we can just directly use the final profile
                    profiles.append((self.data.profile["final"][0], self.data.profile["final"][1], self.data.profile["final"][2]))
                    continue
                else: # here they only share a common wavelength grid, so alpha can be reused, otherwise alpha is None and needs to be recalculated
                    alpha = self.data.alpha["final"]
            else:
                alpha = self._get_order_alpha(wavelengths, f"input_{i}")

            # The process below is almost identical to self._continuum_correct_and_runlsd, see that for more details.
            norm_wl, continuum, continuum_error = self._get_continuum(
                wavelengths, all_poly_coeffs, key=f"input_{i}")
            fitted_flux = flux / continuum
            fitted_errors = np.sqrt((errors/continuum)**2 + (fitted_flux*continuum_error/continuum)**2)
            lsd = LSD(self.data)
            lsd.run_LSD(wavelengths, fitted_flux, fitted_errors, sn, alpha=alpha)
            profiles.append((lsd.profile_F, lsd.profile_errors_F, lsd.cov_z_F))

        self.data.profiles = profiles # store per-frame flux, errors and covariance
        self.data.results_time = time() - t0
        self.data.total_time = self.data.setup_time + self.data.mcmc_time + self.data.results_time
        self.data.complete = True

        # Now that results are complete, save the data instance if specified
        if self.config.save_path is not None:
            self.save() # the sampler is already saved if specified

        logger.info(f"Done! Results processed in {self.data.results_time:.2f} seconds. Total time: {self.data.total_time:.2f} seconds.")
        return

    def _continuum_correct_and_runlsd(self, key:str, all_poly_coeffs:Array2D, **kwargs:Any) -> LSD|None:

        data = self.data

        # Get the poly coeffs as the median of all samples
        poly_coeffs = np.median(all_poly_coeffs, axis=0)

        # Normalise the wavelengths and evaluate the continuum and error from the median
        norm_wavelengths, continuum, continuum_error = self._get_continuum(
            data.wavelengths[key], all_poly_coeffs, key)

        # Fit the flux and error with the continuum
        fitted_flux = data.flux[key] / continuum
        fitted_errors = np.sqrt((data.errors[key]/continuum)**2 + (fitted_flux*continuum_error/continuum)**2)

        # Store all the results, which are then used in LSD.runlsd_and_store
        data.poly_coeffs[key]      = poly_coeffs
        data.norm_wavelengths[key] = norm_wavelengths
        data.continuum[key]        = continuum
        data.continuum_error[key]  = continuum_error
        data.fitted_flux[key]      = fitted_flux
        data.fitted_errors[key]    = fitted_errors

        # Handles running LSD logic and storing all results in Data
        _lsd = LSD.runlsd_and_store(data, key, **kwargs)

        return _lsd

    def _get_order_alpha(self, wavelengths:Array1D, key:str) -> Array2D|Array3D|None:
        """Recalculate each order's triangular interpolation on its own grid."""
        slices = self.data.continuum_slices.get(key)
        if slices is None:
            return None # let the usual single-order LSD path calculate alpha
        if not self.data.order_ll_masks:
            raise ACIDStateError("Recalculating joint alpha requires per-order line selections. Rerun run_multi_order_ACID.")
        matrices = []
        groups = self.data.profile_groups
        for (i, j, _, _), mask in zip(slices, self.data.order_ll_masks):
            lines = self.data.linelist["wavelengths"][mask]
            depths = self.data.linelist["depths"][mask]
            depths = utils.flux_to_od(linelist=depths, od=self.config.od)
            if groups is None:
                alpha = LSD.calc_alpha(wavelengths[i:j], lines, depths,
                                       self.data.velocities, sparse=self.config.sparse)
            else:
                order_groups = groups[np.searchsorted(self.data.ll_mask, mask)]
                alpha = np.stack([LSD.calc_alpha(wavelengths[i:j], lines[order_groups == group],
                                 depths[order_groups == group], self.data.velocities,
                                 sparse=self.config.sparse) for group in np.unique(groups)])
                if key == "mcmc":
                    alpha = LSD.flatten_alpha(alpha)
            matrices.append(alpha)
        return np.concatenate(matrices, axis=1 if groups is not None and key != "mcmc" else 0)

    def _get_continuum(self, wavelengths:Array1D, samples:Array2D, key:str) -> tuple[Array1D, Array1D, Array1D]:
        slices = self.data.continuum_slices.get(key)
        if slices is None:
            slices = [(0, len(wavelengths), 0, samples.shape[1])]
        norm_wl, continua, errors = [], [], []
        for i, j, a, b in slices:
            # MCMC wavelengths were normalised before residual masking.
            x = (self.data.norm_wavelengths[key][i:j] if key == "mcmc"
                 else utils.normalize_wavelengths(wavelengths[i:j]))
            norm_wl.append(x)
            continua.append(utils.eval_continuum(x, np.median(samples[:, a:b], axis=0),
                                                method=self.config.continuum_method))
            errors.append(self._get_continuum_error(x, samples[:, a:b]))
        return tuple(np.concatenate(values) for values in (norm_wl, continua, errors))

    def _get_continuum_error(self, norm_wl:Array1D, all_poly_coeffs:Array2D) -> Array1D:
        ncoeffs = all_poly_coeffs.shape[1]

        if self.data.config.continuum_method == "polyval":
            powers = np.vander(norm_wl, N=ncoeffs, increasing=True)
        else:
            powers = np.polynomial.chebyshev.chebvander(norm_wl, ncoeffs-1)

        # First check memory to see if all samples can be used
        available_memory = utils.get_available_memory() # in bytes

        # MacOS (darwin based) uses 'memory pressure' and is much better at handling memory, so we can allow a less strict safety gap
        safety_gap = 1 if sys.platform == "darwin" else 0.8
        m_available = available_memory * safety_gap / (1024**3) # in GB, with 0.8 factor safety gap

        n_samples, ncoeffs = all_poly_coeffs.shape
        npix = powers.shape[0]
        matrix_size_gb = (2 * n_samples * npix + n_samples * ncoeffs + npix * ncoeffs) * 8 / (1024**3)
        # If memory exceeded, fallback to using 1000 random samples
        if matrix_size_gb > m_available:
            warnings.warn(f"Calculating continuum error with all samples may exceed available memory "
                          f"({matrix_size_gb:.2f} GB required, {m_available:.2f} GB available). "
                          "Calculating with a max of 1000 random samples instead.", ACIDPerformanceWarning, stacklevel=3)
            indices_size = min(1000, n_samples)
            random_indices = self.data.get_result_rng().choice(n_samples, size=indices_size, replace=False)
            coeffs = all_poly_coeffs[random_indices, :]
        else:
            coeffs = all_poly_coeffs

        # Then get the error from the conts matrix above
        if self.data.config.continuum_method == "polyval":
            conts = (coeffs @ powers.T)
        else:
            conts = np.exp(coeffs @ powers.T)
        continuum_error = np.std(conts, axis=0)
        return continuum_error

    @_require_profiles
    def __getitem__(self, item:int|str|tuple) -> list|np.ndarray:
        """
        Allows indexing into the profiles array directly from the Result object.
        """
        if isinstance(item, tuple):
            # Tuples allow for array-like indexing of the list
            if len(item) == 3:
                _order, frame, velocity = item
                return self.data.profiles[frame][velocity]
            elif len(item) == 2:
                frame, velocity = item
                return self.data.profiles[frame][velocity]
            elif len(item) == 1:
                return self.data.profile["final"][item[0]]
            else:
                raise ACIDInputError(f"Tuple indexing must be of length 1, 2, or 3. Got {len(item)} instead.")
        elif isinstance(item, int):
            # Return just the profile or error (or cov_mat) for single int input
            if item < 0 or item > 2:
                raise ACIDInputError(f"Integer index must be 0, 1, or 2 to specify whether to return the profile, error, or covariance matrix. Got {item} instead.")
            return self.data.profile["final"][item]
        else:
            # Various different options for string inputs, why not
            if "error" in item.lower():
                return self.data.profile["final"][1]
            elif "cov" in item.lower():
                return self.data.profile["final"][2]
            elif "profile" in item.lower():
                return self.data.profile["final"][0]
            else:
                raise ACIDInputError(f"String index must contain either 'error', 'cov', or 'profile' to specify which to return. Got {item} instead.")

    @_require_profiles
    def __iter__(self) -> Iterator[tuple|list]:
        """Allows iterating over the profiles array directly from the Result object."""
        return iter(self.data.profiles)

    @property
    @_require_profiles
    def profiles(self) -> "ACID_code.profiles.Profiles":
        """Return a the fitter for the final profiles; use plot_profiles for raw plots."""
        from .profiles import Profiles
        return Profiles(data=self.data)

    def __repr__(self) -> str:
        # Only print out the sampler and data attributes, and whether profiles is available, to avoid printing large arrays
        return f"Result object with sampler={self.sampler}, data={self.data}, profiles={'available' if self.data.profiles is not None else 'not available'}"

    def __str__(self) -> str:
        return self.__repr__()

    @_require_sampler
    @with_output_settings
    def continue_sampling(self, process_results:bool=True, sampler:EnsembleSampler|None=None, **kwargs) -> None:
        """
        Continue MCMC sampling for additional steps. Passes the stored sampler into a Acid instance with the saved data. See
        :py:meth:`~ACID_code.Acid.continue_sampling` for more details on the parameters that can be passed.

        Parameters
        ----------
        process_results : bool, optional
            Whether to process the results after continuing sampling, by default True.
            If False, the profiles attribute will not be updated until :py:meth:`~ACID_code.Result.process_results` is called.
        sampler : emcee.EnsembleSampler | None, optional
            Optionally provide a different sampler to continue sampling from, otherwise,
            takes the sampler from the Result object, by default None
        nsteps : :py:type:`~ACID_code.IntLike`, optional
            Number of additional MCMC steps to run. Passed to :py:meth:`~ACID_code.Acid.continue_sampling` through ``**kwargs``.
        max_steps : :py:type:`~ACID_code.IntLike`, optional
            Maximum number of MCMC steps to run, by default None. Passed to :py:meth:`~ACID_code.Acid.continue_sampling` through ``**kwargs``.
        max_steps_kwargs : dict, optional
            Additional keyword arguments to be passed to the run_mcmc_until_converged function if max_steps is specified, by default None.
            The kwargs description can be found in :py:meth:`~ACID_code.Acid.ACID`, they are the 4 kwargs appearing after max_steps. Typos for kwargs are silently
            ignored. Passed to :py:meth:`~ACID_code.Acid.continue_sampling` through ``**kwargs``.
        parallel : bool, optional
            Overwrites config with whether to run the MCMC in parallel. If None, uses already existing configuration. Default is None.
            Passed to :py:meth:`~ACID_code.Acid.continue_sampling` through ``**kwargs``.
        cores : int, optional
            Overwrites config with the number of cores to use for parallel MCMC. If None, uses already existing configuration. Default is None.
            Passed to :py:meth:`~ACID_code.Acid.continue_sampling` through ``**kwargs``.
        moves : dict, optional
            Overwrites config with the dictionary specifying the moves to use for MCMC sampling. If None, uses already existing configuration. 
            Default is None. See :py:meth:`~ACID_code.Acid.ACID` for format. Passed to :py:meth:`~ACID_code.Acid.continue_sampling` through ``**kwargs``.
        """
        # Note that sampler input updates the sampler stored using the @_require_sampler decorator

        # Continue sampling using the Acid method
        from .acid import Acid
        acid = Acid(data=self.data) # includes config data
        acid.continue_sampling(**kwargs) # updates self.data.sampler (or self.sampler)

        self.initiate_sampler(self.sampler) # update internal variables to match new sampler

        if process_results:
            self.process_results() # update profiles
        else:
            warnings.warn(
                "Warning: Results not processed. profiles attribute will not be available until " \
                "Result.process_results() is called.", ACIDStateWarning, stacklevel=3)

    # Forward model plots to Plot class, allowing for hints to carry accross too
    plot_walkers = Plot.walkers
    plot_traceplot = Plot.traceplot
    plot_corner = Plot.corner
    plot_profiles = _require_profiles(Plot.profiles)
    plot_forward_model = _require_profiles(Plot.forward_model)
    plot_autocorrelation = Plot.autocorrelation
    plot_acf = Plot.acf

    def save(self, *args, **kwargs) -> None:
        """
        Saves the Data instance which the Result class inherits from Acid.
        See :py:meth:`~ACID_code.Data.save` for more details on the parameters that can be passed.
        """
        self.data.save(*args, **kwargs)

    @classmethod
    def load(cls, data:str|Result|Data="result.pkl") -> Result:
        """
        Loads a Result object from a pickle file or from a Data/Result object.

        Parameters
        ----------
        data : :py:class:`str` | :py:class:`~ACID_code.Result` | :py:class:`~ACID_code.Data`, optional
            A pickle file name or an object with the same attributes as a saved Result object, by default "result.pkl"

        Returns
        ----------
        :py:class:`~ACID_code.Result`
            A Result object loaded from the pickle file or from the provided object.
        """
        if isinstance(data, str):
            if os.path.isdir(data):
                try:
                    result = cls(Data.load(os.path.join(data, "data.pkl")))
                except FileNotFoundError:
                    raise FileNotFoundError(f"No 'data.pkl' found in provided directory '{data}'")
            else:
                result = cls(Data.load(data))
            return result
        elif isinstance(data, Result):
            return cls(data.data)
        elif isinstance(data, Data):
            return cls(data)

    @property
    @_require_profiles
    def chi2(self) -> float:
        """Calculates and returns the chi-squared value for the final model."""
        data = self.data
        key = "final"
        return utils.calc_chi2(data.forward_y[key], data.flux[key], data.errors[key])
