"""Time HD189733 order 40: python tests/mcmc/benchmark_jax_vmap.py [--full]."""
import argparse
from contextlib import nullcontext
import json
from multiprocessing.pool import ThreadPool
import os
from pathlib import Path
import pickle
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def prepare(data_dir, full):
    from ACID_code import Acid
    with (data_dir / 'transits.pkl').open('rb') as handle:
        observation = next(iter(pickle.load(handle).values()))[0]
    acid = Acid(velocities=np.arange(-25, 25, 1), verbose=0, seed=0,
                linelist=str(data_dir / 'HD189733_new.txt'))
    acid.ACID(observation['wave'][40], observation['data'][40],
              observation['errors'][40], observation['sn'][40], sigma_lower=5,
              poly_ord=3, od=True, deterministic_profile=not full, masking_lines={},
              use_jax=True, parallel=True, run_mcmc=False)
    return acid.data


def sample(prepared, mode, steps):
    import jax
    from ACID_code import Acid, MCMC
    with prepared.open('rb') as handle:
        acid = Acid(data=pickle.load(handle))
    acid.config._use_jax_vmap = mode == 'vmap'
    started = time.perf_counter()
    acid.run_mcmc(steps, acid.data.initial_state)
    cold = time.perf_counter() - started
    sampler = acid.sampler
    # The API closes its pool after sampling; reopen only the scalar pool, retaining the JIT.
    context = nullcontext(None) if sampler.vectorize else ThreadPool(acid.config.cores)
    with context as pool:
        sampler.pool = pool
        started = time.perf_counter()
        sampler.run_mcmc(None, steps, progress=False)
        warm = time.perf_counter() - started
    chain, log_prob = sampler.get_chain(), sampler.get_log_prob()
    assert sampler.vectorize == (mode == 'vmap')
    assert np.all(np.isfinite(chain)) and np.all(np.isfinite(log_prob))
    acid.config.use_jax = False
    reference = MCMC(acid.data)
    np.testing.assert_allclose(log_prob[-1], [reference(t) for t in chain[-1]], rtol=1e-10, atol=1e-8)
    np.savez(prepared.parent / f'{mode}.npz', chain=chain, log_prob=log_prob)
    return dict(mode=mode, cold_seconds=cold, warm_seconds=warm,
                pixels=len(reference.y), walkers=acid.data.nwalkers, ndim=acid.data.ndim,
                acceptance=float(np.mean(sampler.acceptance_fraction)),
                jax=jax.__version__, devices=str(jax.devices()))


def compare(args):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLBACKEND='Agg')
    runs = []
    data = prepare(args.data_dir, args.full)
    with TemporaryDirectory(prefix='acid-vmap-') as directory:
        prepared = Path(directory) / 'prepared.pkl'
        with prepared.open('wb') as handle:
            pickle.dump(data, handle)
        for _ in range(args.repeats):
            baseline_wall = None
            for mode in ('scalar', 'vmap'):
                # Bound compilation or execution stalls by the measured baseline plus 10 seconds.
                timeout = 180 if baseline_wall is None else baseline_wall + 10
                started = time.perf_counter()
                result = subprocess.run(
                    [sys.executable, __file__, '--worker', mode, '--prepared', str(prepared),
                     '--steps', str(args.steps)], env=env, capture_output=True, text=True, timeout=timeout)
                elapsed = time.perf_counter() - started
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                run = dict(json.loads(result.stdout), wall_seconds=elapsed, timeout_seconds=timeout)
                runs.append(run)
                print(json.dumps(run), flush=True)
                if mode == 'scalar':
                    baseline_wall = elapsed
        with np.load(prepared.parent / 'scalar.npz') as scalar, np.load(prepared.parent / 'vmap.npz') as batch:
            differences = {f'max_{name}_difference': float(np.max(np.abs(scalar[name] - batch[name])))
                           for name in ('chain', 'log_prob')}
    medians = {mode: {phase: float(np.median([r[f'{phase}_seconds'] for r in runs if r['mode'] == mode]))
                      for phase in ('cold', 'warm')} for mode in ('scalar', 'vmap')}
    summary = dict(steps=args.steps, full=args.full, medians=medians, runs=runs, **differences)
    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'science/large_storage/HD189733')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--full', action='store_true', help='fit profile points as well as the continuum')
    parser.add_argument('--output', type=Path, help='save timings and numerical differences as JSON')
    parser.add_argument('--worker', choices=('scalar', 'vmap'), help=argparse.SUPPRESS)
    parser.add_argument('--prepared', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(sample(args.prepared, args.worker, args.steps)))
    else:
        compare(args)
