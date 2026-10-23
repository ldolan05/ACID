"""Runtime output must compose without changing stored or process-wide state."""
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
import asyncio
import inspect
import logging
from threading import Barrier
import warnings

import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from ACID_code import Config, LSD, configure_logging, set_log_level
from ACID_code.diagnostics import logging as log_config
from ACID_code.diagnostics.logging import get_logger
from ACID_code.diagnostics.output import Output, normalize_verbose, output_settings, with_output_settings


@pytest.fixture
def messages(monkeypatch):
    monkeypatch.setattr(log_config, "_level_limit", log_config._level_limit)
    logger = get_logger()
    handler = next(h for h in logger.handlers if getattr(h, "_acid_default_handler", False))
    original = handler.stream
    stream = StringIO()
    configure_logging("DEBUG", stream=stream)
    yield stream
    handler.setStream(original)


def test_config_construction_and_assignment_have_no_output_side_effects(messages):
    logger = get_logger(__name__)
    root_level = get_logger().level
    with Output(4).scope():
        config = Config(verbose=0)
        config.verbose = "low"
        Config(**config.to_dict())
        logger.debug("still in the debug operation")
        assert config.verbose == 1
        assert config.output.debug
    assert "still in the debug operation" in messages.getvalue()
    assert get_logger().level == root_level
    assert config.output.verbose == 1


def test_nested_scopes_restore_after_base_exception_and_keep_warning_state(messages):
    parent = Output(2)
    showwarning = warnings.showwarning
    filters = list(warnings.filters)
    with parent.scope():
        with pytest.raises(KeyboardInterrupt):
            with Output(0).scope():
                assert not Config(verbose=4).output.progress
                raise KeyboardInterrupt
        assert Output.current() is parent
        assert warnings.showwarning is showwarning
        assert warnings.filters == filters
    assert Output.current() is None


def test_concurrent_runs_keep_separate_logging_and_progress(messages):
    barrier = Barrier(2)
    logger = get_logger(__name__)

    def run(level):
        with Output(level).scope():
            barrier.wait(timeout=10)
            logger.info("info for %s", level)
            logger.debug("debug for %s", level)
            result = Config(verbose=3).output.progress
            barrier.wait(timeout=10)
            return result

    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(run, level) for level in (0, 4)]
        assert [future.result(timeout=15) for future in futures] == [False, True]
    output = messages.getvalue()
    assert "info for 4" in output and "debug for 4" in output
    assert "info for 0" not in output and "debug for 0" not in output
    assert Output.current() is None


def test_decorated_async_tasks_keep_separate_output_policies():
    class Operation:
        config = Config(verbose=2)

        @with_output_settings
        async def run(self, verbose):
            await asyncio.sleep(0)
            return self.config.output.verbose

    operation = Operation()

    async def gather():
        return await asyncio.gather(operation.run(0), operation.run(4))

    assert asyncio.run(gather()) == [0, 4]
    assert Output.current() is None


def test_decorator_handles_defaults_positional_and_forwarded_overrides():
    class Operation:
        config = Config(verbose=0)

        @with_output_settings
        def run(self, verbose=2, sampler_progress=None):
            """Calculate using the requested output settings."""
            return self.config.output

        @with_output_settings
        def forward(self, **kwargs):
            return self.run(**kwargs)

    operation = Operation()
    assert operation.run() == Output(2)
    assert operation.run("high", False) == Output(3, False)
    assert operation.forward(verbose=4, sampler_progress=True) == Output(4, True)
    assert operation.config.verbose == 0
    assert operation.config.sampler_progress is None
    assert inspect.signature(Operation.run) == inspect.signature(Operation.run.__wrapped__)
    assert Operation.run.__doc__ == Operation.run.__wrapped__.__doc__
    assert Output.current() is None


def test_decorator_inherits_parent_and_restores_after_failure():
    class Operation:
        config = Config(verbose=4)

        @with_output_settings
        def run(self, verbose=None):
            assert self.config.output.verbose == 1
            raise KeyboardInterrupt

    operation = Operation()
    with output_settings(operation.config, verbose=2, per_order=True) as parent:
        with pytest.raises(KeyboardInterrupt):
            operation.run(verbose=4)
        assert Output.current() is parent
        with pytest.raises(ValueError):
            operation.run(verbose=5)
        assert Output.current() is parent
    assert Output.current() is None
    assert operation.config.output.verbose == 4


def test_generator_methods_require_settings_during_iteration():
    with pytest.raises(TypeError, match="inside generator"):
        @with_output_settings
        def generate(self):
            yield self.config.output


def test_decorator_preserves_class_level_argument_validation():
    @beartype
    class Operation:
        config = Config(verbose=0)

        @with_output_settings
        def run(self, count: int) -> int:
            return count

    operation = Operation()
    assert operation.run(count=3) == 3
    for args, kwargs in [(("invalid",), {}), ((), {"count": "invalid"})]:
        with pytest.raises(BeartypeCallHintParamViolation):
            operation.run(*args, **kwargs)
        assert Output.current() is None


def test_user_logging_limit_applies_inside_verbose_run(messages):
    logger = get_logger(__name__)
    set_log_level("WARNING")
    with Output(4).scope():
        logger.debug("hidden debug")
        logger.info("hidden info")
        logger.warning("visible warning")
    output = messages.getvalue()
    assert "visible warning" in output
    assert "hidden" not in output


def test_nested_alpha_calls_obey_quiet_scope(synthetic_spectrum, capfd):
    wave, _, _, _, velocities, lines = synthetic_spectrum
    with Output(0).scope():
        LSD.calc_alpha(wave, lines["wavelengths"], lines["depths"], velocities, verbose=4)
    captured = capfd.readouterr()
    assert not captured.out and not captured.err


@pytest.mark.parametrize("value, expected", [(None, 2), (True, 2), (False, 0),
                                             ("off", 0), ("MED", 2), ("debug", 4)])
def test_normalization_is_shared_by_config_and_runtime(value, expected):
    assert normalize_verbose(value) == expected
    assert Config(verbose=value).verbose == expected
    assert Output(value).verbose == expected


@pytest.mark.parametrize("value", [-1, 5, 1.5, [], "loud"])
def test_invalid_runtime_verbosity_is_rejected(value):
    with pytest.raises(ValueError):
        Output(value)


def test_sampler_diagnostics_preserve_process_streams(harps_result, monkeypatch):
    import sys
    import matplotlib.pyplot as plt
    from ACID_code import Plot

    streams = sys.stdout, sys.stderr
    original = harps_result.sampler.get_autocorr_time

    def autocorrelation(**kwargs):
        assert (sys.stdout, sys.stderr) == streams
        return original(**kwargs)

    monkeypatch.setattr(harps_result.sampler, "get_autocorr_time", autocorrelation)
    figure, _ = Plot(harps_result.data).walkers(return_fig=True)
    plt.close(figure)


def test_standalone_plot_uses_its_config_without_changing_global_logging(messages, tmp_path):
    import matplotlib.pyplot as plt
    from ACID_code import Data, Plot

    data = Data()
    data.config = Config(verbose=0, figure_dir=str(tmp_path))
    Plot(data).masking_lines()
    assert (tmp_path / "masking_lines.png").exists()
    assert not messages.getvalue()
    data.config.verbose = 2
    Plot(data).masking_lines()
    assert "Saved figure" in messages.getvalue()
    assert Output.current() is None
    plt.close("all")
