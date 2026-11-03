"""
Logging configuration helpers for ACID.

ACID shows normal progress messages by default.  The configuration is limited to
the ``ACID_code`` logger and never changes the application's root logger.
"""

from __future__ import annotations
from .errors import ACIDInputError

import logging
from typing import Optional, Union
from .output import Output, VERBOSITY_LOG_LEVELS


LOGGER_NAME = "ACID_code"
DEFAULT_LOG_LEVEL = logging.INFO
DISABLED_LOG_LEVEL = logging.CRITICAL + 1
DEFAULT_LOG_FORMAT = "%(levelname)s: %(message)s"

_HANDLER_MARKER = "_acid_default_handler"
_level_limit = None


class _OutputFilter(logging.Filter):
    """Filter records using the caller's output scope and an optional global cap."""
    def filter(self, record):
        if record.name != LOGGER_NAME and not record.name.startswith(LOGGER_NAME + "."):
            return True
        output = Output.current()
        level = DEFAULT_LOG_LEVEL if output is None else output.log_level
        if _level_limit is not None:
            level = _level_limit if output is None else max(level, _level_limit)
        return record.levelno >= level


_output_filter = _OutputFilter()


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the ACID package logger or one of its child loggers."""
    if name is None or name == LOGGER_NAME:
        logger = logging.getLogger(LOGGER_NAME)
    elif name.startswith(LOGGER_NAME + "."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"{LOGGER_NAME}.{name}")
    if _output_filter not in logger.filters:
        logger.addFilter(_output_filter)
    return logger


def _normalise_level(level: Union[int, str]) -> int:
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ACIDInputError(f"Unknown logging level: {level!r}")
        return numeric_level
    if isinstance(level, int):
        return level
    raise ACIDInputError("Logging level must be an integer, string, or None.")


def _default_handler(logger: logging.Logger) -> Optional[logging.Handler]:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler
    return None


def configure_logging(
    level  : Union[int, str, None] = DEFAULT_LOG_LEVEL,
    *,
    stream                         = None,
    format : str                   = DEFAULT_LOG_FORMAT,
    ) -> logging.Logger:
    """Configure ACID's own logging without affecting the root logger.

    Parameters
    ----------
    level
        A standard logging level name/value. ``None`` disables ACID logging.
    stream
        Optional stream for the package's default handler. The standard logging
        stream (``sys.stderr``) is used when omitted.
    format
        Format used by the package's default handler.
    """
    logger = get_logger()
    handler = _default_handler(logger)

    # Respect a handler installed by an application before ACID was imported.
    # Otherwise install exactly one package-owned terminal handler.
    if handler is None and not logger.handlers:
        handler = logging.StreamHandler(stream)
        setattr(handler, _HANDLER_MARKER, True)
        logger.addHandler(handler)

    if handler is not None:
        if stream is not None:
            handler.setStream(stream)
        handler.setFormatter(logging.Formatter(format))

    # Keep records available for per-operation filtering, including debug runs.
    logger.setLevel(logging.DEBUG)
    for installed in logger.handlers:
        if _output_filter not in installed.filters:
            installed.addFilter(_output_filter)
    set_log_level(level)

    # Avoid duplicate output through a root handler. Applications can opt back
    # into propagation if they deliberately manage the complete logging tree.
    logger.propagate = False
    return logger


def set_log_level(level: Union[int, str, None]) -> None:
    """Set a global minimum severity for ACID messages; ``None`` disables them.

    Runtime verbosity can further restrict output, but cannot bypass this limit.
    """
    global _level_limit
    _level_limit = DISABLED_LOG_LEVEL if level is None else _normalise_level(level)


def log_level_from_verbosity(verbose: int) -> int:
    """Translate a validated legacy verbosity value to a logging level."""
    try:
        return VERBOSITY_LOG_LEVELS[verbose]
    except (KeyError, TypeError) as exc:
        raise ACIDInputError("verbose must be an integer between 0 and 4") from exc


def set_log_level_from_verbosity(verbose: int) -> None:
    """Apply the logging equivalent of a validated legacy verbosity value."""
    set_log_level(log_level_from_verbosity(verbose))


def _configure_default_logging() -> None:
    """Install ACID's INFO-by-default behavior without changing root logging."""
    global _level_limit
    logger = get_logger()
    if logger.handlers:
        # An application configured ACID before importing it; do not replace
        # that configuration with the package defaults.
        return

    level = DEFAULT_LOG_LEVEL if logger.level == logging.NOTSET else logger.level
    configure_logging(level)
    # Import defaults allow verbose=4 to enable DEBUG; only explicit user logging
    # configuration imposes a global severity limit.
    _level_limit = None


_configure_default_logging()

# Public API, we dont want _ imports
__all__ = [
    "DEFAULT_LOG_LEVEL",
    "DISABLED_LOG_LEVEL",
    "LOGGER_NAME",
    "VERBOSITY_LOG_LEVELS",
    "configure_logging",
    "get_logger",
    "log_level_from_verbosity",
    "set_log_level",
    "set_log_level_from_verbosity",
]
