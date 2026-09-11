"""
Logging configuration helpers for ACID.

ACID shows normal progress messages by default.  The configuration is limited to
the ``ACID_code`` logger and never changes the application's root logger.
"""

from __future__ import annotations

import logging
from typing import Optional, Union


LOGGER_NAME = "ACID_code"
DEFAULT_LOG_LEVEL = logging.INFO
DISABLED_LOG_LEVEL = logging.CRITICAL + 1
DEFAULT_LOG_FORMAT = "%(levelname)s: %(message)s"

# This mapping preserves the textual-output meaning of the legacy verbosity
# setting.  Levels 3 and 4 may additionally control plots/debug-data storage.
VERBOSITY_LOG_LEVELS = {
    0: DISABLED_LOG_LEVEL,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.INFO,
    4: logging.DEBUG,
}

_HANDLER_MARKER = "_acid_default_handler"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return the ACID package logger or one of its child loggers."""
    if name is None or name == LOGGER_NAME:
        return logging.getLogger(LOGGER_NAME)
    if name.startswith(LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def _normalise_level(level: Union[int, str]) -> int:
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Unknown logging level: {level!r}")
        return numeric_level
    if isinstance(level, int):
        return level
    raise TypeError("Logging level must be an integer, string, or None.")


def _default_handler(logger: logging.Logger) -> Optional[logging.Handler]:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler
    return None


def configure_logging(
    level: Union[int, str, None] = DEFAULT_LOG_LEVEL,
    *,
    stream=None,
    format: str = DEFAULT_LOG_FORMAT,
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

    logger.setLevel(DISABLED_LOG_LEVEL if level is None else _normalise_level(level))

    # Avoid duplicate output through a root handler. Applications can opt back
    # into propagation if they deliberately manage the complete logging tree.
    logger.propagate = False
    return logger


def set_log_level(level: Union[int, str, None]) -> None:
    """Set the threshold for ACID log messages; ``None`` disables them."""
    get_logger().setLevel(DISABLED_LOG_LEVEL if level is None else _normalise_level(level))


def log_level_from_verbosity(verbose: int) -> int:
    """Translate a validated legacy verbosity value to a logging level."""
    try:
        return VERBOSITY_LOG_LEVELS[verbose]
    except (KeyError, TypeError) as exc:
        raise ValueError("verbose must be an integer between 0 and 4") from exc


def set_log_level_from_verbosity(verbose: int) -> None:
    """Apply the logging equivalent of a validated legacy verbosity value."""
    set_log_level(log_level_from_verbosity(verbose))


def _configure_default_logging() -> None:
    """Install ACID's INFO-by-default behavior without changing root logging."""
    logger = get_logger()
    if logger.handlers:
        # An application configured ACID before importing it; do not replace
        # that configuration with the package defaults.
        return

    level = DEFAULT_LOG_LEVEL if logger.level == logging.NOTSET else logger.level
    configure_logging(level)


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
