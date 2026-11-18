"""Optional PyMC/NumPyro NUTS sampling, with emcee-compatible output."""
from __future__ import annotations
from functools import lru_cache
import os
import numpy as np
from emcee import EnsembleSampler, State
from tqdm import tqdm
from ..diagnostics.logging import get_logger
from emcee.backends import Backend
from .. import utils
from ..diagnostics.errors import ACIDInputError, ACIDInitialStateError
from ..diagnostics.output import with_output_settings
from ..utils import IntLike
from .jax import _import_jax
from .mcmc import MCMC
from .sampler import Sampler


@lru_cache(maxsize=1)
def _import_pymc():
    """Create the small PyTensor bridge only when PyMC sampling is requested."""
    try:
        import pymc as pm
        import numpyro
        import pytensor.tensor as pt
        from pymc.sampling.jax import sample_numpyro_nuts
        from pytensor.graph.op import Op
        from pytensor.link.jax.dispatch import jax_funcify
    except ImportError as exc:
        raise ImportError("The pymc sampler requires PyMC, NumPyro, and JAX. "
                          "Install them with: pip install 'ACID_code[pymc]'") from exc

    # Expose the shared posterior as a PyTensor operation for PyMC's model graph.
    class LogProbability(Op):
        itypes = [pt.dvector]
        otypes = [pt.dscalar]

        def __init__(self, log_probability):
            self.log_probability = log_probability

        def perform(self, node, inputs, outputs):
            outputs[0][0] = np.asarray(self.log_probability(inputs[0]), dtype=float)

    @jax_funcify.register(LogProbability)
    def log_probability_dispatch(op, **kwargs):
        # NumPyro differentiates the shared JAX model, including its Cholesky solve.
        return op.log_probability

    return pm, sample_numpyro_nuts, LogProbability


logger = get_logger(__name__)


class PyMCBackend(Sampler):
    """Sample the shared posterior with NUTS; use emcee only to store the output."""

    def get_initial_state(self, reuse:bool=False):
        """Start from the fitted continuum and, for the full model, its LSD profile."""
        # Here nwalkers counts independent NUTS chains.
        self.data.nwalkers = 4 if self.config.nwalkers is None else self.config.nwalkers
        if not isinstance(self.data.nwalkers, (int, np.integer)) or self.data.nwalkers < 1:
            raise ACIDInputError("nwalkers must be a positive integer (the number of NUTS chains).")
        if (reuse and self.data.initial_state is not None and self.data.initial_state.shape
                in ((self.data.ndim,), (self.data.nwalkers, self.data.ndim))):
            return self.data.initial_state

        # Match the shared posterior's parameter order: profile first, then continuum.
        theta = np.asarray(self.data.poly_coeffs["masked"], dtype=float)
        if not self.config.deterministic_profile:
            profile = np.asarray(self.data.profile["masked"][0]).reshape(-1)
            profile = -np.log(profile) if self.config.od else profile
            theta = np.concatenate((profile, theta))
        return theta

    @with_output_settings
    def run_mcmc(self, nsteps:IntLike, state=None, *, _until_converged:bool=False) -> None:
        """Draw nsteps per chain after warmup, appending when state is None and a sampler exists.

        Continuation starts a new NUTS adaptation from the last saved positions.
        Warmup is discarded on every call; the NumPyro kernel itself is not persisted.
        _until_converged is used internally by run_mcmc_until_converged.
        """
        # Keep draw counts, seeds and model selection under ACID's control.
        if nsteps < 1:
            raise ACIDInputError("nsteps must be positive for PyMC sampling.")
        options = dict(self.config.pymc_kwargs or {})
        managed = {"draws", "chains", "random_seed", "model", "var_names", "initvals", "nuts_sampler"}
        if managed.intersection(options):
            raise ACIDInputError("Set draws, chains, seed, and initial state through nsteps, nwalkers, "
                                 "seed, and state; pymc_kwargs cannot override the model or sampler.")

        # Build the same posterior as emcee, using JAX for evaluation and gradients.
        modules = _import_jax(_disable_warnings=True)
        if modules is None:
            raise ImportError("The pymc sampler requires working 64-bit JAX; install 'ACID_code[pymc]'.")
        jax, jnp, linalg = modules
        pm, sample_numpyro_nuts, LogProbability = _import_pymc()
        probability = MCMC(self.data, numpy_module=jnp, linalg=linalg)
        log_probability = jax.jit(probability.log_probability)

        # Resume saved positions when no new state is supplied. Adaptation starts
        # afresh for each call; only batches within a convergence run share it.
        previous = self.data.sampler if state is None else None
        if previous is not None:
            state = previous.get_last_sample().coords
        elif state is None:
            state = self.get_initial_state()
        if isinstance(state, State):
            state = state.coords

        # Accept one shared starting point or a separate point for each chain.
        state = np.asarray(state, dtype=float)
        if state.shape == (self.data.ndim,):
            state = np.broadcast_to(state, (self.data.nwalkers, self.data.ndim))
        if state.shape != (self.data.nwalkers, self.data.ndim):
            raise ACIDInputError("The PyMC initial state must have shape (ndim,) or (nwalkers, ndim).")

        # Reject starting points with non-finite densities or gradients.
        for theta in state:
            value, gradient = jax.value_and_grad(log_probability)(theta)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise ACIDInitialStateError("PyMC requires a finite initial log probability and gradient.")

        options.setdefault("jitter", False)
        options.setdefault("chain_method", "vectorized" if self.config.parallel else "sequential")
        options.setdefault("progressbar", self.config.output.sampling)

        # The shared posterior already includes priors; Flat adds no extra penalty.
        with pm.Model() as model:
            theta = pm.Flat("theta", shape=self.data.ndim, dtype="float64")
            pm.Potential("log_probability", LogProbability(log_probability)(theta))
            if _until_converged:
                chain, stats = self.sample_until_converged(model, state, options, nsteps, previous)
            else:
                trace = sample_numpyro_nuts(
                    draws=int(nsteps), chains=self.data.nwalkers, model=model, var_names=["theta"],
                    initvals=[{"theta": theta.copy()} for theta in state],
                    random_seed=int(self.data.get_rng().integers(2**30)), **options)

                # Convert PyMC's output to emcee's (draw, chain, parameter) layout.
                chain = np.asarray(trace.posterior["theta"]).transpose(1, 0, 2)
                stats = {name: np.asarray(value.transpose("draw", "chain"))
                         for name, value in trace.sample_stats.data_vars.items() if value.ndim == 2}
                self.data.sampler_diagnostics = {}

        # Evaluate explicitly: NumPyro's `lp` statistic can use potential-energy signs.
        log_prob = np.asarray(jax.lax.map(log_probability, jnp.asarray(
            chain.reshape(-1, self.data.ndim)))).reshape(chain.shape[:2])

        # Store NUTS diagnostics as emcee blobs, preserving the schema on continuation.
        blobs = np.empty(chain.shape[:2], dtype=(previous.get_blobs().dtype if previous is not None else
            [(name, stats[name].dtype) for name in sorted(stats)]))
        for name, value in stats.items():
            blobs[name] = value
        accepted = stats["acceptance_rate"].sum(axis=0)

        # Append retained samples and acceptance totals to any existing run.
        if previous is not None:
            chain = np.concatenate((previous.get_chain(), chain))
            log_prob = np.concatenate((previous.get_log_prob(), log_prob))
            blobs = np.concatenate((previous.get_blobs(), blobs))
            accepted += previous.backend.accepted

        # Expose NUTS results through emcee's chain access, plotting and saving interfaces.
        backend = Backend()
        backend.reset(self.data.nwalkers, self.data.ndim)
        backend.chain, backend.log_prob, backend.blobs = chain, log_prob, blobs
        backend.accepted, backend.iteration = accepted, len(chain)
        if self.config.sampler_path is not None:
            utils.ensure_directory(os.path.dirname(self.config.sampler_path), "sampler directory")
            backend = utils.save_backend_to_hdf5(backend, self.config.sampler_path)
        self.data.sampler = EnsembleSampler(self.data.nwalkers, self.data.ndim,
                                           probability, backend=backend)
        self.data.nsteps = len(chain)
        self.data.complete = False

    @with_output_settings
    def run_mcmc_until_converged(self, max_steps:IntLike, state=None) -> None:
        """Warm up once, then check NUTS diagnostics every check_interval retained draws."""
        if self.data.nwalkers is None or self.data.nwalkers < 2:
            raise ACIDInputError("NUTS convergence checks require at least two chains (nwalkers).")
        if self.config.check_interval < 1 or self.config.min_checks < 1:
            raise ACIDInputError("check_interval and min_checks must be positive.")
        self.run_mcmc(max_steps, state, _until_converged=True)

    @staticmethod
    def stopping_criterion(chain, divergences):
        """Rank-normalized R-hat and bulk/tail ESS, using ArviZ's native diagnostics."""
        import arviz as az
        import xarray as xr
        diagnostics = dict(rhat_max=np.inf, ess_bulk_min=0., ess_tail_min=0.,
                           divergences=int(divergences), converged=False)
        if chain.shape[0] < 4:
            return diagnostics
        # ArviZ expects named chain/draw axes; take the worst diagnostic over all parameters.
        posterior = xr.Dataset({"theta": (("chain", "draw", "parameter"), chain.transpose(1, 0, 2))})
        diagnostics.update(
            rhat_max=float(az.rhat(posterior, method="rank")["theta"].max(skipna=False)),
            ess_bulk_min=float(az.ess(posterior, method="bulk")["theta"].min(skipna=False)),
            ess_tail_min=float(az.ess(posterior, method="tail")["theta"].min(skipna=False)))
        values = [diagnostics[name] for name in ("rhat_max", "ess_bulk_min", "ess_tail_min")]
        diagnostics["converged"] = bool(
            np.all(np.isfinite(values)) and diagnostics["rhat_max"] < 1.01
            and min(values[1:]) >= 100 * chain.shape[1] and divergences == 0)
        return diagnostics

    def sample_until_converged(self, model, state, options, max_steps, previous):
        """Use NumPyro's public continuation API to preserve adaptation and RNG between checks."""
        import jax
        import jax.numpy as jnp
        from numpyro.infer import MCMC as NUTSSampler, NUTS
        from pymc.sampling.jax import get_jaxified_logp

        # This path drives NumPyro directly, so PyMC postprocessing options do not apply.
        options = options.copy()
        allowed = {"tune", "target_accept", "nuts_kwargs", "chain_method", "progressbar",
                   "jitter", "compute_convergence_checks"}
        if set(options) - allowed or options.get("jitter", False):
            raise ACIDInputError("NUTS convergence runs support tune, target_accept, nuts_kwargs, "
                                 "chain_method, and progressbar; jitter must be False.")

        # Construct one sampler for the whole run, including its initial warmup.
        kernel = NUTS(potential_fn=get_jaxified_logp(model, negative_logp=False),
                      target_accept_prob=options.get("target_accept", 0.8),
                      **options.get("nuts_kwargs", {}))
        sampler = NUTSSampler(kernel, num_warmup=options.get("tune", 1000),
                              num_samples=int(min(max_steps, self.config.check_interval)),
                              num_chains=self.data.nwalkers, chain_method=options["chain_method"],
                              progress_bar=False)
        key = jax.random.PRNGKey(int(self.data.get_rng().integers(2**30)))
        chain_parts, stat_parts = [], []

        # Convergence checks include earlier retained samples and any earlier divergences.
        old_chain = previous.get_chain() if previous is not None else None
        divergences = int(previous.get_blobs()["diverging"].sum()) if previous is not None else 0
        step_number = checks = passed = 0
        self.data.sampler_diagnostics = {}
        with tqdm(total=max_steps, desc="NUTS convergence", unit="draw",
                  disable=not options["progressbar"]) as progress:
            while step_number < max_steps:
                sampler.num_samples = int(min(self.config.check_interval, max_steps - step_number))
                sampler.run(key, init_params=[jnp.asarray(state)], extra_fields=(
                    "accept_prob", "diverging", "energy", "num_steps", "potential_energy", "adapt_state.step_size"))
                # Collect this batch and translate NumPyro's statistics to the stored names.
                chain_parts.append(np.asarray(sampler.get_samples(group_by_chain=True)[0]).transpose(1, 0, 2))
                fields = sampler.get_extra_fields(group_by_chain=True)
                names = {"accept_prob": "acceptance_rate", "num_steps": "n_steps",
                         "adapt_state.step_size": "step_size", "potential_energy": "lp"}
                stats = {names.get(name, name): np.asarray(value).T for name, value in fields.items()}
                stats["tree_depth"] = np.floor(np.log2(np.maximum(stats["n_steps"], 1))).astype(int) + 1
                stat_parts.append(stats)
                divergences += int(stats["diverging"].sum())
                step_number += sampler.num_samples
                checks += 1

                # Check the accumulated history, requiring consecutive successful checks.
                chain = np.concatenate(chain_parts)
                history = chain if old_chain is None else np.concatenate((old_chain, chain))
                diagnostics = self.stopping_criterion(history, divergences)
                passed = passed + 1 if diagnostics["converged"] else 0
                diagnostics.update(converged=passed >= self.config.min_checks, checks=checks)
                self.data.sampler_diagnostics = diagnostics
                progress.update(sampler.num_samples)
                progress.set_postfix(rhat=diagnostics["rhat_max"], ess=min(
                    diagnostics["ess_bulk_min"], diagnostics["ess_tail_min"]), divergences=divergences)
                if diagnostics["converged"]:
                    logger.info("NUTS converged after %d retained draws", len(history))
                    break

                # Resume the adapted kernel and RNG instead of repeating warmup at each check.
                sampler.post_warmup_state = sampler.last_state
                key = sampler.last_state.rng_key
        return chain, {name: np.concatenate([part[name] for part in stat_parts]) for name in stats}

