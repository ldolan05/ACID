"""
Custom error classes for ACID. To be used mainly for handling specific errors in the DataList run_ACID method.
"""

class ACIDError(Exception):
    """Base class for all ACID-related errors."""

class LineListRangeError(ACIDError):
    """Custom error for when no lines in the linelist are within the wavelength range of the observed spectrum."""

class ContinuumError(ACIDError):
    """Custom error for when the continuum fit results in negative fluxes or errors."""

class SNCutError(ACIDError):
    """Custom error for when the S/N cut results in no valid pixels."""