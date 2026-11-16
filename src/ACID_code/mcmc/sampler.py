"""Sampler selection, worker pools, and continuation of prepared ACID runs."""
from __future__ import annotations
import multiprocessing as mp
import os, time
from contextlib import nullcontext
from dataclasses import fields
from multiprocessing.pool import ThreadPool
from typing import TYPE_CHECKING
import numpy as np
from .. import utils
from ..data import Config, Data
from ..diagnostics.errors import ACIDInputError, ACIDInitialStateError, ACIDStateError
from ..diagnostics.logging import get_logger
from ..diagnostics.output import with_output_settings
from ..utils import IntLike
from .mcmc import MCMC

if TYPE_CHECKING:
    import ACID_code
    import emcee

logger = get_logger(__name__)


# Process workers initialise their model once, avoiding serialization per sample.
def _mp_init_worker(data):
    global _model
    _model = MCMC(data)


def _mp_log_probability(theta):
    return _model.log_probability(theta)


def _mp_log_likelihood(theta):
    return _model.dynesty_logprob(theta)


def _mp_ptform(u):
    return _model.ptform(u)


class Sampler:
    """Run the configured sampler against a shared Data instance."""

    def __init__(self, data:Data):
        self.data = data

    @property
    def config(self):
        return self.data.config

    @property
    def backend(self) -> "Sampler":
        if self.config.sampler_type == "emcee":
            from .emcee import EmceeBackend
            return EmceeBackend(self.data)
        if self.config.sampler_type == "pymc":
            from .pymc import PyMCBackend
            return PyMCBackend(self.data)
        from .dynesty import DynestyBackend
        return DynestyBackend(self.data)

    @with_output_settings
    def run_mcmc(self, nsteps:IntLike, state=None) -> None:
        """Run the selected sampler for the requested steps (live points for dynesty)."""
        self.backend.run_mcmc(nsteps, state)

    @with_output_settings
    def run_mcmc_until_converged(self, max_steps:IntLike, state=None) -> None:
        """Run the selected sampler until convergence or the step limit."""
        self.backend.run_mcmc_until_converged(max_steps, state)

    def get_initial_state(self, reuse:bool=False) -> np.ndarray|None:
        """Set sampler dimensions and prepare valid walkers for emcee."""
        # Preserve the initial RNG for results; only walkers and sampling advance this one.
        rng = self.data.get_rng()

        n_profile_params = self.data.alpha["mcmc"].shape[1]

        # Set the number of dimensions, add the no. of velocity points if also fitting the profile
        self.data.ndim = len(self.data.poly_coeffs["masked"])
        if not self.config.deterministic_profile:
            self.data.ndim += n_profile_params

        # emcee recommendation is 3 times the number of dimensions (we add a +3 buffer as well), but can be overridden by user input
        self.data.nwalkers = 3 + self.data.ndim * 3 if self.config.nwalkers is None else self.config.nwalkers

        if self.config.sampler_type == "pymc":
            return self.backend.get_initial_state(reuse)

        if self.config.sampler_type == "emcee":
            # Preparation with run_mcmc=False already drew these walkers.
            if (reuse and self.data.initial_state is not None
                    and self.data.initial_state.shape == (self.data.nwalkers, self.data.ndim)):
                return self.data.initial_state
            theta0 = self.data.poly_coeffs["masked"]

            if not self.config.deterministic_profile:
                profile0 = np.asarray(self.data.profile["masked"][0]).reshape(-1)
                profile0 = -np.log(profile0) if self.config.od else profile0
                theta0 = np.concatenate((profile0, theta0))

            # Test to see if the starting position is valid, up to a max of 1000 attempts
            test_mcmc = MCMC(self.data)
            width = np.maximum(0.5 * np.abs(theta0), 1e-3)
            walkers = []
            max_attempts = max(1000, 100 * self.data.nwalkers)
            n_attempt = 0
            while len(walkers) < self.data.nwalkers:
                theta = rng.normal(theta0, width)
                if np.isfinite(test_mcmc.log_probability(theta)):
                    walkers.append(theta)

                n_attempt += 1
                if n_attempt == max_attempts:
                    raise ACIDInitialStateError("Reached the max number of attempts for finding an initial state for MCMC walkers.")

            initial_state = np.array(walkers)
        else:
            initial_state = None

        return initial_state

    def _get_sampler_pool(self):
        """Use threads for JAX and the original process pool for NumPy."""
        if self.config.cores is None:
            self.config.cores = (int(os.environ.get("SLURM_CPUS_ON_NODE", 1))
                                 if "SLURM_JOB_ID" in os.environ else os.cpu_count())
        use_threads = self.config.use_jax
        if self.config.parallel:
            utils.configure_mp_environ(os)
            workers = "threads" if use_threads else "processes"
            logger.info(f"Using {self.config.cores} {workers} for MCMC")

            if not use_threads:
                pool = mp.get_context("fork").Pool(
                    processes=self.config.cores,
                    initializer=_mp_init_worker,
                    initargs=(self.data,),
                )
                log_prob = _mp_log_probability if self.config.sampler_type == "emcee" else _mp_log_likelihood
                return pool, log_prob, _mp_ptform

        # Each pool owns its model. Do not use the process-global worker model
        # for threads: simultaneous ACID runs would overwrite each other's data.
        model = MCMC(self.data)
        log_prob = model if self.config.sampler_type == "emcee" else model.dynesty_logprob
        pool = ThreadPool(processes=self.config.cores) if self.config.parallel else nullcontext(None)
        return pool, log_prob, model.ptform

    @with_output_settings
    def continue_sampling(
        self,
        nsteps           : IntLike|None = None,
        max_steps        : IntLike|None = None,
        max_steps_kwargs : dict|None    = None,
        parallel         : bool|None    = None,
        cores            : int|None     = None,
        moves            : dict|None    = None,
        return_sampler   : bool         = False,
        ) -> "emcee.EnsembleSampler | None":
        """
        Continue a prepared emcee run for additional steps, sharing its Data and RNG.

        Parameters
        ----------
        nsteps : :py:type:`~ACID_code.IntLike`, optional
            Number of additional steps to run the MCMC for.
        max_steps : :py:type:`~ACID_code.IntLike`, optional
            Maximum number of steps to run the MCMC for in total (including previous steps).
            If specified, the MCMC will stop if this number of steps is reached even if convergence has not been reached, by default None.
            If input, nsteps is ignored.
        max_steps_kwargs : dict, optional
            Additional keyword arguments to be passed to the run_mcmc_until_converged function if max_steps is specified, by default None.
            The kwargs description can be found in Acid.ACID(), they are the 4 kwargs appearing after max_steps. Typos for kwargs are silently
            ignored.
        parallel : bool, optional
            Overwrites config with whether to run the MCMC in parallel. If None, uses already existing configuration. Default is None.
        cores : int, optional
            Overwrites config with the number of cores to use for parallel MCMC. If None, uses already existing configuration. Default is None.
        moves : dict, optional
            Overwrites config with the dictionary specifying the moves to use for MCMC sampling. If None, uses already existing configuration. 
            Default is None. See :py:meth:`~ACID_code.Acid.ACID` for format.
        return_sampler : bool, optional
            Whether to return the sampler after continuing sampling. Default is False.

        Returns
        -------
        emcee.EnsembleSampler | None
            The MCMC sampler after running for the additional steps, or None if return_sampler is False.
        """
        # Update config with any new parallel, cores, or moves settings for the continued sampling
        self.config.update_hipri(parallel=parallel, cores=cores, moves=moves)

        if max_steps is not None:
            if max_steps_kwargs is not None:
                self.config.update_hipri(**max_steps_kwargs)
            remaining_steps = max_steps - self.data.sampler.backend.iteration
            if remaining_steps > 0:
                self.run_mcmc_until_converged(remaining_steps, state=None)
                self.data.complete = False

        else:
            if nsteps is None:
                raise ACIDInputError("Either nsteps or max_steps must be provided.")

            self.run_mcmc(nsteps, state=None)
            self.data.complete = False

        self.data.nsteps = self.data.sampler.backend.iteration

        if return_sampler:
            return self.data.sampler

    def has_saved_run(self, config_kwargs:dict) -> bool:
        """Check both paths so an incomplete saved run still raises on resume."""
        paths = ("dir", "save_path", "sampler_path")
        stored = self.config.to_dict()
        # Preserve disabling strings, which the public properties resolve to None.
        config = Config(**{name: stored["_" + name] for name in paths
                           if "_" + name in stored})
        config.update_hipri(**{name: config_kwargs[name] for name in paths if name in config_kwargs})
        return any(path is not None and os.path.exists(path)
                   for path in (config.save_path, config.sampler_path))

    def resume(self, config_kwargs:dict) -> "ACID_code.result.Result | None":
        """Restore the existing save format and use the normal continuation path."""

        # Define the editable runtime settings for the continuation of the MCMC run.
        runtime_settings = (
            "verbose", "sampler_progress", "dir", "save_path", "sampler_path", "figure_dir",
            "parallel", "cores", "use_jax", "_use_jax_vmap", "moves", "nsteps", "max_steps", "check_interval",
            "min_checks", "min_tau_factor", "tau_tol", "run_mcmc", "resume",
        )
        # Include settings supplied through Acid.__init__ or an existing Config.
        overrides = self.config.to_dict()
        overrides = {name: getattr(self.config, name) for name in runtime_settings
                     if name in overrides or "_" + name in overrides}
        overrides.update({name: value for name, value in config_kwargs.items()
                          if name in runtime_settings and value is not None})
        config = Config(**overrides)
        if config.save_path is None or not os.path.isfile(config.save_path):
            raise ACIDStateError("resume=True requires the saved Data .pkl file as well as the .h5 chain. "
                                 "Set dir or save_path to the saved run.")

        data = Data.load(config.save_path, sampler_path=config.sampler_path)
        data.config.update_hipri(**overrides)
        # Point relocated runs at the paths used to load them.
        data.config.update_hipri(save_path=config.save_path, sampler_path=config.sampler_path,
                                figure_dir=config.figure_dir)
        if data.config.sampler_type != "emcee" or data.sampler is None:
            raise ACIDStateError("resume=True requires a saved emcee HDF5 sampler.")
        backend = data.sampler.backend
        if backend.iteration == 0:
            raise ACIDStateError("The saved sampler has no samples to resume. Start a new run with resume=False.")
        if backend.shape != (data.nwalkers, data.ndim):
            raise ACIDStateError("The saved sampler shape does not match the saved Data.")

        # Restore in place so DataList and Result retain their reference to this Data.
        for field in fields(data):
            setattr(self.data, field.name, getattr(data, field.name))
        self.data.exception = self.data.traceback = None
        self.data.nsteps = backend.iteration
        self.data.complete = False
        self.data.get_rng().set_state(self.data.sampler.random_state)
        if not self.config.run_mcmc:
            return None

        logger.info("Resuming MCMC from step %d", self.data.nsteps)
        started = time.time()
        remaining_steps = max(0, self.config.nsteps - self.data.nsteps)
        if self.config.max_steps is not None or remaining_steps > 0:
            self.continue_sampling(nsteps=remaining_steps, max_steps=self.config.max_steps)
        self.data.mcmc_time += time.time() - started
        from ..result import Result
        return Result(self.data)

