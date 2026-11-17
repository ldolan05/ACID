"""Check the benchmark's ESS calculation against known dependence structures."""
import numpy as np
import pytest

from benchmark_nuts_convergence import autocorrelation_diagnostics


def test_independent_samples_have_ess_close_to_the_sample_count():
    chain = np.random.default_rng(3).normal(size=(20000, 8, 2))
    diagnostics = autocorrelation_diagnostics(chain)
    assert diagnostics["autocorrelation_reliable"]
    assert diagnostics["ess_min"] == pytest.approx(160000, rel=0.15)


def test_duplicating_dependent_walkers_does_not_multiply_ess():
    trajectory = np.random.default_rng(4).normal(size=(20000, 1, 2))
    single = autocorrelation_diagnostics(trajectory)
    duplicated = autocorrelation_diagnostics(np.repeat(trajectory, 16, axis=1))
    assert duplicated["ess_min"] == pytest.approx(single["ess_min"], rel=0.001)


def test_autocorrelation_reduces_ess():
    rng = np.random.default_rng(5)
    chain = rng.normal(size=(30000, 4, 1))
    for step in range(1, len(chain)):
        chain[step] += 0.9 * chain[step-1]
    diagnostics = autocorrelation_diagnostics(chain[1000:])
    assert diagnostics["autocorrelation_reliable"]
    # An AR(1) process has integrated autocorrelation time (1 + rho)/(1 - rho).
    assert diagnostics["ess_min"] == pytest.approx(29000 * 4 / 19, rel=0.25)


def test_short_or_constant_traces_do_not_claim_reliable_precision():
    for chain in (np.zeros((3, 4, 2)), np.zeros((100, 4, 2))):
        diagnostics = autocorrelation_diagnostics(chain)
        assert diagnostics["ess_min"] is None
        assert not diagnostics["autocorrelation_reliable"]
