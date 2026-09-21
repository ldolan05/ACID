"""Warning categories and default warning behavior for ACID."""

import warnings


class ACIDWarning(UserWarning):
    """Base class for all warnings emitted by ACID."""


class ACIDInputWarning(ACIDWarning):
    """Input data are usable but contain potentially problematic values."""


class ACIDRuntimeWarning(ACIDWarning):
    """A runtime issue occurred during ACID execution."""


class ACIDConvergenceWarning(ACIDWarning):
    """A calculation completed without meeting a convergence criterion."""


class ACIDPerformanceWarning(ACIDWarning):
    """An operation may require unexpectedly large computational resources."""


class ACIDStateWarning(ACIDWarning):
    """Stored ACID state was discarded, changed, could not be restored, or requires additional processing."""


class ACIDDeprecationWarning(ACIDWarning, DeprecationWarning):
    """A deprecated ACID function/method was used."""


def configure_warnings(action="default", *, append=False):
    """Set the default action for ACID warnings only.

    Explicit calls take priority over existing filters by default. Set
    ``append=True`` to let existing filters retain priority instead.
    Valid actions are the standard values accepted by ``warnings.filterwarnings``.
    """
    warnings.filterwarnings(action, category=ACIDWarning, append=append)


# This is deliberately narrow: it affects only ACID warning categories.
configure_warnings(append=True)
