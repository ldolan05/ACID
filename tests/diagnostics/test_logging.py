import logging

import pytest

import ACID_code
from ACID_code.diagnostics.logging import log_level_from_verbosity


def test_acid_logger_is_info_by_default():
    logger = logging.getLogger("ACID_code")
    original_level = logger.level
    try:
        ACID_code.configure_logging()
        assert logger.getEffectiveLevel() == logging.INFO
        assert logger.handlers
        assert logger.propagate is False
    finally:
        logger.setLevel(original_level)


@pytest.mark.parametrize(
    "verbose, expected",
    [
        (0, logging.CRITICAL + 1),
        (1, logging.WARNING),
        (2, logging.INFO),
        (3, logging.INFO),
        (4, logging.DEBUG),
    ],
)
def test_legacy_verbosity_logging_translation(verbose, expected):
    assert log_level_from_verbosity(verbose) == expected


def test_public_log_level_configuration():
    logger = logging.getLogger("ACID_code")
    original_level = logger.level
    try:
        ACID_code.set_log_level("DEBUG")
        assert logger.level == logging.DEBUG
        ACID_code.set_log_level(None)
        assert logger.level == logging.CRITICAL + 1
    finally:
        logger.setLevel(original_level)

