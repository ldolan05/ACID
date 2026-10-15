import warnings
from types import ModuleType

import numpy as np
import pytest


import ACID_code
from astropy.stats import sigma_clip
from ACID_code.diagnostics.warnings import ACIDDroppedDataWarning, ACIDWarning


def test_acid_warning_categories_are_public_and_filterable():
    assert issubclass(ACIDDroppedDataWarning, ACIDWarning)
    assert ACID_code.ACIDWarning is ACIDWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ACIDWarning)
        warnings.warn("problematic input", ACIDDroppedDataWarning)

    assert len(caught) == 1
    assert caught[0].category is ACIDDroppedDataWarning


def test_acid_warnings_use_default_filter():
    # Pytest deliberately replaces filters installed during package import, so
    # exercise ACID's configuration function in a locally scoped filter list.
    with warnings.catch_warnings():
        ACID_code.configure_warnings()
        assert any(
            action == "default" and category is ACIDWarning
            for action, _message, category, _module, _lineno in warnings.filters
        )


@pytest.mark.parametrize("action, expected", [("once", 1), ("always", 3)])
@pytest.mark.parametrize("operation", ["pixels", "linelist"])
def test_dropped_data_warning_filter_across_caller_modules(action, expected, operation):
    code = ('ACID_code.utils.drop_invalid(np.array([1., 2.]), np.array([1., np.nan]))'
            if operation == "pixels" else
            'ACID_code.LineList({"wavelengths": [5000., 5001.], "depths": [.2, np.nan]})')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action, ACIDDroppedDataWarning)
        for name in ("data", "data_list", "result"):
            caller = ModuleType(name)
            # Give each caller its own globals, as real modules and generated
            # beartype wrappers do. A helper here would share one registry.
            caller.__dict__.update(ACID_code=ACID_code, np=np)
            exec(code, caller.__dict__)

    assert len(caught) == expected
    assert all(warning.category is ACIDDroppedDataWarning for warning in caught)


@pytest.mark.parametrize("action, expected", [("once", 1), ("always", 3), ("ignore", 0)])
def test_dropped_data_warning_survives_sigma_clipping(action, expected):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(action, ACIDDroppedDataWarning)
        for _ in range(3):
            ACID_code.utils.drop_invalid(np.array([1., 2.]), np.array([1., np.nan]))
            sigma_clip(np.arange(5.), axis=0)
    assert len(caught) == expected


def test_explicit_error_filter_overrides_once_history():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once", ACIDDroppedDataWarning)
        ACID_code.utils.drop_invalid(np.array([1., 2.]), np.array([1., np.nan]))
        warnings.simplefilter("error", ACIDDroppedDataWarning)
        with pytest.raises(ACIDDroppedDataWarning):
            ACID_code.utils.drop_invalid(np.array([1., 2.]), np.array([1., np.nan]))
    assert len(caught) == 1


@pytest.mark.parametrize("size", [0, 1, 100, 10000])
@pytest.mark.parametrize("lower, upper", [(3, 3), (5, 3)])
def test_sigma_clipping_explicit_axis_preserves_masks_and_bounds(size, lower, upper):
    residuals = np.random.default_rng(1).normal(size=size)
    residuals[::23] = 10.
    original, lo, hi = sigma_clip(residuals, sigma_lower=lower, sigma_upper=upper, return_bounds=True)
    fast, fast_lo, fast_hi = sigma_clip(residuals, axis=0, sigma_lower=lower, sigma_upper=upper, return_bounds=True)
    np.testing.assert_array_equal(np.ma.getmaskarray(fast), np.ma.getmaskarray(original))
    np.testing.assert_allclose([fast_lo, fast_hi], [lo, hi], rtol=1e-12, atol=1e-12)

