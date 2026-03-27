"""
Custom error classes for ACID.
"""

class LineListRangeError(Exception):
    """Custom error for when no lines in the linelist are within the wavelength range of the observed spectrum."""
    pass

class ContinuumError(ValueError):
    """Custom error for when the continuum fit results in negative fluxes or errors."""
    pass

class SNCutError(ValueError):
    """Custom error for when the S/N cut results in no valid pixels."""
    pass