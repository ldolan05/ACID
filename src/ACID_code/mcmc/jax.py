"""Lazy imports and double-precision setup for the optional JAX stack."""
from __future__ import annotations
from ..diagnostics.warnings import ACIDRuntimeWarning
import warnings


def _import_jax(_import_jaxfit=False, _disable_warnings=False):
    """Import the optional JAX stack only when the backend is requested."""
    # Attempt import of modules
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy.linalg as jsp_linalg
    except ImportError:
        if not _disable_warnings:
            warnings.warn(f"JAX was requested but is not importable; falling back to NumPy/SciPy.\n" \
                           "Install it with: pip install jax", ACIDRuntimeWarning, stacklevel=3)
        return None

    # ACID and its Cholesky solve use double precision, so enable the equivalent JAX mode
    try:
        jax.config.update("jax_enable_x64", True)
    except Exception as exc:
        if not _disable_warnings:
            warnings.warn(f"JAX could not enable 64-bit calculations ({exc}); falling back to NumPy/SciPy.", ACIDRuntimeWarning, stacklevel=3)
        return None

    # If requested, also import jaxfit
    if _import_jaxfit:
        try:
            import jaxfit
        except ImportError:
            if not _disable_warnings:
                warnings.warn(f"Profile fitting with JAX also requires jaxfit to be installed; falling back to NumPy/SciPy.\n" \
                               "Install it with: pip install jaxfit", ACIDRuntimeWarning, stacklevel=3)
            return None
        return jax, jnp, jsp_linalg, jaxfit

    return jax, jnp, jsp_linalg
