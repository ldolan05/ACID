"""Compare unchanged emcee and PyMC NUTS samplers on HD189733.

python tests/mcmc/benchmark_nuts_convergence.py --output /private/tmp/acid-convergence

Time includes initial-state generation, compilation, warmup, and diagnostic checks.
Native checks are recorded, with an additional common precision floor: mean ESS
at least 400 and enough draws to estimate autocorrelation (50 times tau).
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def prepare(args):
    if all((args.output / f"{case}.pkl").exists() for case in args.cases):
        return
    from ACID_code import Acid, DataList
    with (args.data_dir / "transits.pkl").open("rb") as handle:
        observation = next(iter(pickle.load(handle).values()))[0]
    options = dict(velocities=np.arange(-25, 25, 1),
                   linelist=str(args.data_dir / "HD189733_new.txt"), verbose=0, seed=0,
                   sigma_lower=5, poly_ord=3, od=True, deterministic_profile=True,
                   masking_lines={}, use_jax=False, parallel=False)
    for case, indices in (("single", [40]), ("multi", list(range(33, 43)))):
        if case not in args.cases:
            continue
        if (args.output / f"{case}.pkl").exists():
            continue
        start = time.perf_counter()
        if case == "single":
            acid = Acid(**options)
            acid.ACID(observation["wave"][40], observation["data"][40],
                      observation["errors"][40], observation["sn"][40], run_mcmc=False)
            data = acid.data
        else:
            orders = DataList(wavelengths=observation["wave"][indices],
                              flux=observation["data"][indices],
                              errors=observation["errors"][indices],
                              sn=np.asarray(observation["sn"])[indices],
                              order_range=np.asarray(indices), **options)
            orders.run_multi_order_ACID(run_mcmc=False)
            data = orders.multi_order_data
        with (args.output / f"{case}.pkl").open("wb") as handle:
            pickle.dump(data, handle)
        print(json.dumps(dict(case=case, orders=indices, ndim=data.ndim,
                              pixels=len(data.flux["mcmc"]),
                              setup_seconds=time.perf_counter()-start)), flush=True)


def finite_json(value):
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def autocorrelation_diagnostics(chain):
    """Estimate mean ESS/MCSE from the ensemble-mean time series.

    Averaging walkers before estimating autocorrelation includes cross-walker
    dependence. The posterior variance divided by MCSE squared gives ESS.
    This measures precision of parameter means, not rank R-hat or tail ESS.
    """
    from emcee.autocorr import integrated_time
    unavailable = dict(ess_min=None, mcse_max=None, tau_max=None, mean_tau_max=None,
                       autocorrelation_reliable=False)
    if len(chain) < 4:
        return unavailable
    means = chain.mean(axis=1)
    variance = chain.var(axis=(0, 1), ddof=1)
    mean_variance = means.var(axis=0, ddof=1)
    if not np.all(variance > 0) or not np.all(mean_variance > 0):
        return unavailable
    tau = integrated_time(chain, tol=0)
    mean_tau = integrated_time(means, tol=0, has_walkers=False)
    mcse_squared = mean_variance * mean_tau / len(chain)
    valid = (np.all(np.isfinite(tau)) and np.all(tau > 0)
             and np.all(np.isfinite(mcse_squared)) and np.all(mcse_squared > 0)
             and np.all(np.isfinite(mean_tau)) and np.all(mean_tau > 0))
    if not valid:
        return unavailable
    return dict(ess_min=float(np.min(variance / mcse_squared)),
                mcse_max=float(np.max(np.sqrt(mcse_squared))),
                tau_max=float(tau.max()), mean_tau_max=float(mean_tau.max()),
                autocorrelation_reliable=bool(len(chain) > 50 * max(tau.max(), mean_tau.max())))


def precision_passed(common):
    return common["autocorrelation_reliable"] and common["ess_min"] >= 400


def sample(args):
    from ACID_code import Acid, MCMC
    from ACID_code.mcmc.emcee import EmceeBackend
    from ACID_code.mcmc.pymc import PyMCBackend
    with (args.output / f"{args.case}.pkl").open("rb") as handle:
        data = pickle.load(handle)
    nuts = args.worker == "pymc"
    data.config.update_hipri(sampler_type=args.worker, nwalkers=4 if nuts else None,
                            seed=args.seed, parallel=True, use_jax=True,
                            _use_jax_vmap=True,
                            check_interval=500, min_checks=1, sampler_progress=False,
                            pymc_kwargs={"tune":500}, verbose=0)
    # None means leave unchanged in Config; these snapshots use the automatic count.
    if not nuts:
        data.config.__dict__.pop("nwalkers", None)
    data._rng = data._initial_rng = None
    acid = Acid(data=data)
    history = []
    criterion = PyMCBackend.stopping_criterion if nuts else EmceeBackend.stopping_criterion
    start = time.perf_counter()

    def record(chain, native):
        common = autocorrelation_diagnostics(chain)
        item = finite_json(dict(steps=len(chain)+500, retained_draws=len(chain),
                                seconds=time.perf_counter()-start, common=common,
                                precision_passed=precision_passed(common), **native))
        history.append(item)
        print(json.dumps(item, allow_nan=False), flush=True)
        return precision_passed(common)

    def check_nuts(chain, divergences):
        diagnostics = criterion(chain, divergences)
        precision = record(chain, diagnostics)
        diagnostics["converged"] = diagnostics["converged"] and precision
        return diagnostics

    def check_emcee(tau_list, steps, min_checks, min_tau_factor, tau_tol):
        converged, tolerance, neff = criterion(tau_list, steps, min_checks, min_tau_factor, tau_tol)
        native_passed = converged
        if converged and min_tau_factor:
            full_tau = acid.sampler.get_autocorr_time(tol=0)
            native_passed = bool(np.all(np.isfinite(full_tau)) and np.max(full_tau) > 0
                                 and steps / np.max(full_tau) > min_tau_factor)
        precision = record(acid.sampler.get_chain(discard=500),
                           dict(converged=native_passed, tau_tolerance=tolerance, neff_per_walker=neff))
        return converged and precision, tolerance, neff

    if nuts:
        PyMCBackend.stopping_criterion = staticmethod(check_nuts)
    else:
        EmceeBackend.stopping_criterion = staticmethod(check_emcee)
    data.initial_state = acid.get_initial_state()
    initialization_seconds = time.perf_counter()-start
    acid.run_mcmc_until_converged(4500 if nuts else 5000, data.initial_state)
    seconds = time.perf_counter()-start
    chain = acid.sampler.get_chain()
    retained = chain[0 if nuts else 500:]
    selected = np.arange(min(4, data.nwalkers))

    # Verify the stored density against the common NumPy posterior outside timing.
    data.config.use_jax = False
    reference = MCMC(data)
    np.testing.assert_allclose(acid.sampler.get_log_prob()[-1, selected],
                               [reference(theta) for theta in chain[-1, selected]],
                               rtol=1e-10, atol=1e-7)
    common = autocorrelation_diagnostics(retained)
    result = finite_json(dict(
        case=args.case, sampler=args.worker, seed=args.seed, seconds=seconds,
        initialization_seconds=initialization_seconds,
        iterations=len(chain) + (500 if nuts else 0), stored_steps=len(chain),
        retained_draws=len(retained), walkers=data.nwalkers,
        ndim=data.ndim, pixels=len(data.flux["mcmc"]), history=history,
        native_converged=history[-1]["converged"],
        precision_passed=precision_passed(common), common=common,
        nuts_diagnostics=data.sampler_diagnostics if nuts else None,
        posterior_mean=retained.mean(axis=(0, 1)).tolist(),
        posterior_sd=retained.std(axis=(0, 1)).tolist(),
        versions={name:importlib.metadata.version(name)
                  for name in ("numpy", "emcee", "jax", "pymc", "numpyro", "arviz")},
        platform=platform.platform(), machine=platform.machine(), cores=os.cpu_count()))
    stem = args.output / f"{args.case}-{args.worker}-{args.seed}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    np.savez_compressed(stem.with_suffix(".npz"), chain=retained)
    print(json.dumps(result, allow_nan=False), flush=True)


def compare(args):
    prepare(args)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", MPLBACKEND="Agg",
               MPLCONFIGDIR="/private/tmp/acid-mpl")
    for case in args.cases:
        for seed in args.seeds:
            # Alternate execution order; never run timed samplers concurrently.
            samplers = ("pymc", "emcee") if seed % 2 == 0 else ("emcee", "pymc")
            for sampler in samplers:
                stem = args.output / f"{case}-{sampler}-{seed}"
                if stem.with_suffix(".json").exists():
                    previous = json.loads(stem.with_suffix(".json").read_text())
                    if precision_passed(previous["common"]) or previous["iterations"] >= 5000:
                        continue
                with stem.with_suffix(".log").open("w") as handle:
                    subprocess.run([sys.executable, __file__, "--worker", sampler,
                                    "--case", case, "--seed", str(seed),
                                    "--output", str(args.output)], env=env, stdout=handle,
                                   stderr=subprocess.STDOUT, check=True, timeout=args.timeout)
                print(stem.with_suffix(".json").read_text(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "science/large_storage/HD189733")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=("single", "multi"), default=["single", "multi"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--worker", choices=("pymc", "emcee"), help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=("single", "multi"), help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sample(args) if args.worker else compare(args)
