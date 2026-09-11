"""Warning categories and default warning behavior for ACID."""

import warnings


class ACIDWarning(UserWarning):
    """Base class for all warnings emitted by ACID."""


class ACIDInputWarning(ACIDWarning):
    """Input data are usable but contain potentially problematic values."""


class ACIDConvergenceWarning(ACIDWarning):
    """A calculation completed without meeting a convergence criterion."""


class ACIDPerformanceWarning(ACIDWarning):
    """An operation may require unexpectedly large computational resources."""


class ACIDStateWarning(ACIDWarning):
    """Stored ACID state was discarded, changed, or could not be restored."""


class ACIDDeprecationWarning(ACIDWarning):
    """A deprecated ACID function/method was used."""


def configure_warnings(action="once"):
    """Set the default action for ACID warnings only.

    User-installed filters retain priority because the ACID filter is appended.
    Valid actions are the standard values accepted by ``warnings.filterwarnings``.
    """
    warnings.filterwarnings(action, category=ACIDWarning, append=True)


# This is deliberately narrow: it affects only ACID warning categories.
configure_warnings()
