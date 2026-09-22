import logging
import warnings

import pytest

import ACID_code
from ACID_code.diagnostics.logging import log_level_from_verbosity
from ACID_code.diagnostics.warnings import ACIDDroppedDataWarning, ACIDWarning


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
