import logging
from io import StringIO

import pytest

import ACID_code
from ACID_code.diagnostics.logging import log_level_from_verbosity, get_logger
import ACID_code.diagnostics.logging as log_config


@pytest.fixture(autouse=True)
def restore_logging(monkeypatch):
    monkeypatch.setattr(log_config, "_level_limit", log_config._level_limit)
    logger = get_logger()
    handler = next(h for h in logger.handlers if getattr(h, "_acid_default_handler", False))
    stream = handler.stream
    yield
    handler.setStream(stream)


def test_acid_logger_is_info_by_default():
    stream = StringIO()
    logger = ACID_code.configure_logging(stream=stream)
    logger.debug("hidden debug")
    logger.info("visible info")
    assert "visible info" in stream.getvalue()
    assert "hidden debug" not in stream.getvalue()
    assert logger.propagate is False


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
    stream = StringIO()
    logger = ACID_code.configure_logging("DEBUG", stream=stream)
    logger.debug("debug enabled")
    assert "debug enabled" in stream.getvalue()
    ACID_code.set_log_level(None)
    logger.error("disabled")
    assert "disabled" not in stream.getvalue()


@pytest.mark.parametrize("verbose", [0, 2, 3])
def test_alpha_progress_does_not_change_package_logging(synthetic_spectrum, verbose):
    import numpy as np
    from ACID_code import LSD

    wavelengths, _, _, _, velocities, lines = synthetic_spectrum
    logger = logging.getLogger("ACID_code")
    original_level = logger.level
    try:
        logger.setLevel(logging.DEBUG)
        LSD.calc_mp_alpha(wavelengths, velocities, lines["wavelengths"],
                          lines["depths"], np.array([0, 1]), verbose=verbose)
        assert logger.level == logging.DEBUG
        LSD.calc_alpha(wavelengths, lines["wavelengths"], lines["depths"],
                       velocities, verbose=verbose)
        assert logger.level == logging.DEBUG
    finally:
        logger.setLevel(original_level)
