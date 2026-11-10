"""Profile backend selection must also work without optional accelerators."""
import builtins
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import curve_fit

from ACID_code import Profiles
from ACID_code.diagnostics.warnings import ACIDRuntimeWarning
from ACID_code.mcmc import jax as jax_module


@pytest.fixture
def profile_inputs():
    x = np.linspace(-12, 12, 61)
    return x, Profiles.gaussian_func(x, -.25, 1.5, 2.2), np.full_like(x, .01)


@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt"])
@pytest.mark.parametrize("multiple", [False, True])
def test_numpy_profiles_do_not_import_jax(profile_inputs, monkeypatch, model, multiple):
    def unexpected_import(*args, **kwargs):
        pytest.fail("use_jax=False must not import JAX")

    monkeypatch.setattr(jax_module, "_import_jax", unexpected_import)
    x, flux, errors = profile_inputs
    if multiple:
        flux = np.array([flux, flux])
    parameters, covariance = getattr(Profiles(x, flux, errors, use_jax=False), f"fit_{model}")()
    np.testing.assert_allclose(parameters[..., 1], 1.5, atol=.2)
    assert np.isfinite(covariance).all()


@pytest.mark.parametrize("missing", ["jax", "jaxfit"])
@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt"])
def test_profiles_fall_back_without_optional_packages(profile_inputs, monkeypatch, missing, model):
    original_import = builtins.__import__

    def optional_import(name, *args, **kwargs):
        if name == missing or name.startswith(missing + "."):
            raise ModuleNotFoundError(f"No module named '{missing}'")
        # Exercise missing jaxfit even on machines without JAX installed.
        if missing == "jaxfit" and name in ("jax", "jax.numpy", "jax.scipy.linalg"):
            return SimpleNamespace(config=SimpleNamespace(update=lambda *args: None))
        return original_import(name, *args, **kwargs)

    expected = getattr(Profiles(*profile_inputs, use_jax=False), f"fit_{model}")()
    monkeypatch.setattr(builtins, "__import__", optional_import)
    with pytest.warns(ACIDRuntimeWarning, match="falling back"):
        actual = getattr(Profiles(*profile_inputs, use_jax=True), f"fit_{model}")()
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1])


@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt"])
@pytest.mark.parametrize("covariance", [False, True])
def test_profiles_select_jaxfit(profile_inputs, monkeypatch, model, covariance):
    # A test backend verifies dispatch even when the optional stack is absent.
    import scipy
    calls = []

    def fit(function, x, y, **kwargs):
        calls.append("fit")
        assert function.keywords == {"np": np, "scipy": scipy}
        np.testing.assert_array_equal(kwargs["sigma"], sigma)
        return curve_fit(function, x, y, **kwargs)

    def jit(function):
        calls.append("jit")
        return function

    modules = (SimpleNamespace(jit=jit, scipy=scipy), np, None,
               SimpleNamespace(CurveFit=lambda: SimpleNamespace(curve_fit=fit)))
    monkeypatch.setattr(jax_module, "_import_jax", lambda **kwargs: modules)
    x, flux, errors = profile_inputs
    sigma = np.diag(errors ** 2) if covariance else errors
    profiles = Profiles(x, flux, errors, sigma if covariance else None, use_jax=True)
    parameters, _ = getattr(profiles, f"fit_{model}")()
    assert calls == ["jit", "fit"]
    assert parameters[1] == pytest.approx(1.5, abs=.2)
    assert np.isfinite(profiles.fitted_yerr[model]).all()


@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt"])
def test_real_jaxfit_matches_scipy(profile_inputs, model):
    pytest.importorskip("jax")
    pytest.importorskip("jaxfit", exc_type=ImportError)
    expected = getattr(Profiles(*profile_inputs, use_jax=False), f"fit_{model}")()
    profiles = Profiles(*profile_inputs, use_jax=True)
    actual = getattr(profiles, f"fit_{model}")()
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-6, rtol=.05)
    assert np.isfinite(profiles.fitted_yerr[model]).all()


@pytest.mark.parametrize("model", ["gaussian", "lorentzian", "voigt"])
def test_jitted_profiles_match_scipy(profile_inputs, monkeypatch, model):
    # Use real JAX tracing with SciPy's optimiser to isolate model compatibility
    # from jaxfit's availability and version compatibility.
    pytest.importorskip("jax")
    modules = jax_module._import_jax()
    if modules is None:
        pytest.skip("JAX cannot enable double precision")
    jax, jnp, linalg = modules
    backend = SimpleNamespace(CurveFit=lambda: SimpleNamespace(curve_fit=curve_fit))
    monkeypatch.setattr(jax_module, "_import_jax", lambda **kwargs: (jax, jnp, linalg, backend))
    expected = getattr(Profiles(*profile_inputs, use_jax=False), f"fit_{model}")()
    actual = getattr(Profiles(*profile_inputs, use_jax=True), f"fit_{model}")()
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-3, rtol=1e-3)
