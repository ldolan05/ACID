"""
Custom error classes for ACID. To be used mainly for handling specific errors in the DataList run_ACID method.
"""

class ACIDError(Exception):
    """Base class for all ACID-related errors."""

class ACIDInputError(ACIDError, ValueError):
    """Invalid input arguments to ACID."""

class ACIDLineListRangeError(ACIDError):
    """Custom error for when no lines in the linelist are within the wavelength range of the observed spectrum."""

class ACIDContinuumFitError(ACIDError):
    """Custom error for when the continuum fit results in negative fluxes or errors."""

class ACIDSNCutError(ACIDError):
    """Custom error for when the S/N cut results in no valid pixels."""

class ACIDInitialStateError(ACIDError):
    """Custom error for when the initial state for MCMC walkers is invalid, or the process of generating it fails."""

class ACIDResultError(ACIDError):
    """Custom error for when there is an issue with the results processing in the Result class."""

class ACIDStateError(ACIDError):
    """Custom error for when the state or attributes of the instances are invalid. Similar to ``ACIDStateWarning``, but the the issue cannot be continued from."""

class ACIDGroupingError(ACIDError):
    """Custom error for when there is an issue with profile group assignments."""