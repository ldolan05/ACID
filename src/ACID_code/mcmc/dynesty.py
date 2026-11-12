"""Nested sampling with the optional dynesty sampler."""
from __future__ import annotations
from ..diagnostics.errors import ACIDInputError
from ..diagnostics.output import with_output_settings
from ..utils import IntLike
from .sampler import Sampler


class DynestyBackend(Sampler):
    """dynesty sampling using the shared likelihood and prior transform."""

    @with_output_settings
    def run_mcmc(self, nsteps:IntLike, state=None) -> None:
        import dynesty
        pool_context, log_prob, ptform = self._get_sampler_pool()
        queue_size = self.config.cores if self.config.parallel else None
        with pool_context as pool:
            if pool is not None:
                pool.size = self.config.cores
            self.data.sampler = dynesty.NestedSampler(
                log_prob, ptform, self.data.ndim, nsteps,
                pool=pool, queue_size=queue_size, rstate=self.data.get_rng())
            self.data.sampler.run_nested(print_progress=self.config.output.sampling)

    def run_mcmc_until_converged(self, max_steps:IntLike, state=None) -> None:
        raise ACIDInputError("dynesty handles convergence internally; use run_mcmc with nsteps=nlive.")
