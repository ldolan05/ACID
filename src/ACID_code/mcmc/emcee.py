"""Ensemble sampling and autocorrelation-based convergence with emcee."""
from __future__ import annotations
import os, emcee
import numpy as np
from emcee import EnsembleSampler
from tqdm import tqdm
from .. import utils
from ..diagnostics.errors import ACIDInputError, ACIDStateError
from ..diagnostics.logging import get_logger
from ..diagnostics.output import with_output_settings
from ..utils import IntLike
from .sampler import Sampler

logger = get_logger(__name__)


class EmceeBackend(Sampler):
    """emcee-specific sampling, persistence, and convergence checks."""

    @with_output_settings
    def run_mcmc(self, nsteps:IntLike, state=None) -> None:
        """Sample for a fixed number of additional steps."""
        sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps, state)
        pool_context, log_prob, _ = self._get_sampler_pool()
        with pool_context as pool:
            self.data.sampler = EnsembleSampler(log_prob_fn=log_prob, pool=pool, **sampler_kwargs)
            try:
                self.data.sampler.run_mcmc(**mcmc_kwargs)
            finally:
                self.data.get_rng().set_state(self.data.sampler.random_state)

    @with_output_settings
    def run_mcmc_until_converged(self, max_steps:IntLike, state=None) -> None:
        """Sample in chunks until the enabled autocorrelation criteria pass or max_steps is reached."""
        if self.config.min_tau_factor == 0 and self.config.tau_tol == 0:
            raise ACIDInputError("min_tau_factor and tau_tol cannot both be 0; at least one convergence criterion must be enabled.")

        # Get sampler and stopping criterion kwargs for the first run based on initial state, then update nsteps in mcmc_kwargs for subsequent runs
        sampler_kwargs, mcmc_kwargs = self._get_sampler_kwargs(nsteps=self.config.check_interval, state=state)
        show_progress = mcmc_kwargs.get("progress", True)
        mcmc_kwargs["progress"] = False
        stopping_criterion_args = (self.config.min_checks, self.config.min_tau_factor, self.config.tau_tol)

        # Set variables to be updated within the convergence loop
        step_number = 0
        tau_list = []
        max_samples = int(np.ceil(max_steps / self.config.check_interval))
        last_tolerance = np.inf
        last_neff = 0
        condition = False
        pool_context, log_prob_fn, _ = self._get_sampler_pool()

        with pool_context as pool, tqdm(
                 total=max_samples, desc="MCMC convergence", unit="check", disable=not show_progress,
             ) as progress:

            self.data.sampler = EnsembleSampler(**sampler_kwargs, pool=pool, log_prob_fn=log_prob_fn)

            # Count completed checks explicitly so early convergence is counted too.
            for _ in range(max_samples):
                steps_this_run = min(self.config.check_interval, max_steps-step_number)
                mcmc_kwargs["nsteps"] = steps_this_run

                try:
                    self.data.sampler.run_mcmc(**mcmc_kwargs, skip_initial_state_check=True)
                finally:
                    self.data.get_rng().set_state(self.data.sampler.random_state)
                mcmc_kwargs["initial_state"] = None

                step_number += steps_this_run

                # We want to keep the time for get_autocorr_time to run constant, so thin accordingly
                # It scales with the number of steps, so thin by the number of steps taken divided by 
                # the check interval to keep the same number of samples for get_autocorr_time to process.
                try:
                    tau = self.data.sampler.get_autocorr_time(tol=0, thin=max(1, self.data.sampler.backend.iteration//self.config.check_interval))
                except emcee.autocorr.AutocorrError:
                    progress.set_postfix_str("autocorrelation unavailable", refresh=False)
                    progress.update(1)
                    continue
                tau_list.append(tau)

                # The stopping criterion function below handles the logic for determining stopping condition
                total_step_number = self.data.sampler.backend.iteration
                condition, last_tolerance, last_neff = self.stopping_criterion(tau_list, total_step_number, *stopping_criterion_args)
                if condition is True and self.config.min_tau_factor != 0:
                    # Confirm only the effective sample size using the full chain.
                    full_tau = self.data.sampler.get_autocorr_time(tol=0)
                    tau_max = np.max(full_tau)
                    last_neff = total_step_number / tau_max if np.all(np.isfinite(full_tau)) and tau_max > 0 else 0
                    condition = bool(last_neff > self.config.min_tau_factor)
                tol_str, neff_str = self.progress_description(last_tolerance, last_neff, self.config)
                progress.set_postfix_str(f"tolerance: {tol_str}, neff: {neff_str}", refresh=False)
                if condition is True:
                    progress.set_description_str("MCMC converged", refresh=False)
                progress.update(1)

                if condition is True:
                    break

        if condition is True:
            logger.info(
                "Converged at step %d (tolerance %.4f, effective sample size %.2f)",
                total_step_number,
                last_tolerance,
                last_neff,
            )

    def _get_sampler_kwargs(self, nsteps, state=None):
        # Gets sampler kwargs for the emcee EnsembleSampler and run_mcmc functions based on the current state of the
        # ACID instance and the inputted nsteps and state.

        # Set verbosity of the sampler with sampler_progress override if specified
        sampler_verbosity = self.config.output.sampling
        
        backend = None
        if state is None:
            if self.data.sampler is None:
                raise ACIDStateError(f"Either a state or an existing sampler must be provided to initiate the sampler. \n" \
                                      "This has most likely happened because you ran continue_sampling without first running ACID or using run_mcmc=False.")
            backend = self.data.sampler.backend # This includes previous seed
        
        # Now that the backend has been set depending on an existing state, if the backend is stil None, we choose depending on sampler_path
        if backend is None and self.config.sampler_path is not None:
            utils.ensure_directory(os.path.dirname(self.config.sampler_path), "sampler directory")
            backend = emcee.backends.HDFBackend(self.config.sampler_path)
            if state is not None:
                backend.reset(self.data.nwalkers, self.data.ndim)

        elif self.data.sampler is not None:
            # And, if it is still none, we check if self.data.sampler already exists in memory, 
            # and if so, we reuse its backend. We started with the path backend as that supersedes
            backend = self.data.sampler.backend
        # else: leave none and a normal in-memory sampler backend is used

        # Configure moves based on config, this function converts a Config moves dictionary format to a class format
        # accepted for emcee moves.
        moves = utils.convert_moves_to_emcee(self.config.moves)

        # A fresh chain starts immediately after walker generation. For continuation,
        # leave state=None so emcee restores both walkers and RNG from its backend.
        if state is not None:
            state = emcee.State(state, copy=True)
            if state.random_state is None:
                state.random_state = self.data.get_rng().get_state()

        sampler_kwargs = {
            "nwalkers": self.data.nwalkers,
            "ndim"    : self.data.ndim,
            "moves"   : moves,
            "backend" : backend,
        }
        mcmc_kwargs = {
            "initial_state"  : state,
            "nsteps"         : nsteps,
            "progress"       : sampler_verbosity,
            "store"          : True,
            "tune"           : True,
            "progress_kwargs": {"mininterval":0.5}
        }
        # Quietly save the prepared likelihood before sampling can be interrupted. The chain
        # stays in HDF5 and Data.load reconnects it through config.sampler_path.
        if state is not None and isinstance(backend, emcee.backends.HDFBackend) and self.config.save_path is not None:
            self.data.complete = False
            self.data.save(quiet=True)
        return sampler_kwargs, mcmc_kwargs

    @staticmethod
    def stopping_criterion(tau_list, step_number, min_checks, min_tau_factor, tau_rel_tol):
        """Determines whether the MCMC sampling has converged based on the list of tau estimates and the current step number."""
        # --- always try to compute a tolerance metric as early as possible ---
        tol_metric = np.nan  # show NaN until computable
        n_eff = 0

        if len(tau_list) >= 2:
            # compute a "most recent" change metric from the last two tau estimates
            a, b = tau_list[-2], tau_list[-1]
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                den = np.maximum(b, 1e-12)
                tol_metric = float(np.percentile(np.abs(b - a) / den, 90))
            
            n_eff = step_number / np.max(tau_list)
            try:
                n_eff = int(n_eff)
            except:
                pass
            if tau_rel_tol != 0 and np.isnan(tol_metric):
                return False, np.inf, n_eff

        # Need enough tau estimates to do min_checks consecutive deltas
        if len(tau_list) < (min_checks + 1):
            return False, np.inf, n_eff

        tau = tau_list[-1]
        if not np.all(np.isfinite(tau)):
            return False, tol_metric, n_eff
        if np.any(tau <= 0):
            return False, tol_metric, n_eff

        tau_ref = np.max(tau)
        if min_tau_factor != 0 and not (step_number > min_tau_factor * tau_ref):
            return False, tol_metric, n_eff

        # Stability over last min_checks deltas
        recent_tau = tau_list[-(min_checks + 1):]
        rel_changes = []
        for a, b in zip(recent_tau[:-1], recent_tau[1:]):
            den = np.maximum(b, 1e-12)
            rel_changes.append(np.percentile(np.abs(b - a) / den, 90))

        tol_metric = float(np.max(rel_changes))  # overwrite with the stricter metric
        return bool(tau_rel_tol == 0 or tol_metric < tau_rel_tol), tol_metric, n_eff

    @staticmethod
    def progress_description(last_tolerance, last_neff, config):
        """Formats the tqdm description string with the current tolerance and effective sample size metrics."""
        tol_str = "<" if last_tolerance < config.tau_tol else ">"
        tol_str = f"{last_tolerance:.4f}{tol_str}{config.tau_tol}"
        neff_str = ">" if last_neff > config.min_tau_factor else "<"
        neff_str = f"{last_neff:.2f}{neff_str}{config.min_tau_factor}"
        if config.tau_tol == 0:
            tol_str = "disabled"
        if config.min_tau_factor == 0:
            neff_str = "disabled"
        return tol_str, neff_str

