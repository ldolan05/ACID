"""Runtime output policy shared by standalone calls and batches.

Config stores user settings. Output scopes control presentation without changing
Config, warning filters, or global logger levels. Nested calls inherit the scope.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import inspect
from numbers import Integral
import logging
from beartype import beartype

from .errors import ACIDInputError


_active_output = ContextVar("acid_output", default=None)
VERBOSITY_LOG_LEVELS = {0: logging.CRITICAL + 1, 1: logging.WARNING,
                        2: logging.INFO, 3: logging.INFO, 4: logging.DEBUG}


def normalize_verbose(value, default=2):
    """Return a verbosity integer without changing any output settings."""
    if value is None or value is True:
        value = default
    elif value is False:
        value = 0
    elif isinstance(value, str):
        aliases = (("none", "no", "false", "off", "n", "0"),
                   ("low", "lo", "l", "1"), ("medium", "med", "m", "2"),
                   ("high", "all", "hi", "h", "3"), ("debug", "dbg", "d", "4"))
        value = next((i for i, names in enumerate(aliases) if value.lower() in names), None)
        if value is None:
            raise ACIDInputError("verbose string not recognised; use none, low, medium, high, or debug")
    if not isinstance(value, Integral) or not 0 <= value <= 4:
        raise ACIDInputError("verbose must be an integer between 0 and 4, a boolean, or a verbosity string")
    return int(value)


@dataclass(frozen=True)
class Output:
    """Resolved output settings for an operation; safe to reuse across scopes."""
    verbose: int = 2
    sampler_progress: bool | None = None

    def __post_init__(self):
        object.__setattr__(self, "verbose", normalize_verbose(self.verbose))
        if self.sampler_progress is not None and not isinstance(self.sampler_progress, bool):
            raise ACIDInputError("sampler_progress must be a boolean or None")

    @property
    def progress(self):
        return self.verbose >= 2

    @property
    def sampling(self):
        return self.progress if self.sampler_progress is None else self.sampler_progress

    @property
    def plots(self):
        return self.verbose >= 3

    @property
    def debug(self):
        return self.verbose == 4

    @property
    def log_level(self):
        return VERBOSITY_LOG_LEVELS[self.verbose]

    @classmethod
    def current(cls, config=None, *, verbose=None, sampler_progress=None):
        """Use the active scope, otherwise resolve the supplied Config's settings."""
        active = _active_output.get()
        if active is not None:
            return active
        if config is None:
            return None if verbose is None else cls(verbose, sampler_progress)
        return cls(config.verbose if verbose is None else verbose,
                   config.sampler_progress if sampler_progress is None else sampler_progress)

    @classmethod
    def for_order(cls, config, verbose):
        """At batch levels 0--2, reserve routine progress for the outer order bar."""
        verbose = normalize_verbose(verbose)
        if cls.current() is not None:
            verbose = min(verbose, cls.current().verbose)
        level = verbose if verbose >= 3 else min(verbose, 1)
        sampling = False if verbose <= 2 else config.sampler_progress
        return cls(level, sampling)

    @contextmanager
    def scope(self):
        """Apply this policy to nested calls in the current thread/task only."""
        token = _active_output.set(self)
        try:
            yield self
        finally:
            _active_output.reset(token)


@contextmanager
def output_settings(config, *, verbose=None, sampler_progress=None, per_order=False):
    """Apply normal output settings, or quieter settings for one order in a batch."""
    if verbose is not None:
        verbose = normalize_verbose(verbose)
    if sampler_progress is not None and not isinstance(sampler_progress, bool):
        raise ACIDInputError("sampler_progress must be a boolean or None")
    if per_order:
        output = Output.for_order(config, config.verbose if verbose is None else verbose)
    else:
        output = Output.current(config, verbose=verbose, sampler_progress=sampler_progress)
    with output.scope():
        yield output


def with_output_settings(method):
    """Apply a configured method's output settings, including call-time overrides.

    Works with synchronous and asynchronous methods. Generators must enter
    output_settings inside their body so settings remain active during iteration.
    """
    if inspect.isgeneratorfunction(method) or inspect.isasyncgenfunction(method):
        raise TypeError("Use output_settings inside generator methods")
    # Class-level beartype cannot check named arguments through a *args wrapper.
    method = beartype(method)
    signature = inspect.signature(method)
    extra_name = next((p.name for p in signature.parameters.values()
                       if p.kind == inspect.Parameter.VAR_KEYWORD), None)

    def settings(self, args, kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        extra = bound.arguments.get(extra_name, {})
        options = {name: bound.arguments.get(name, extra.get(name))
                   for name in ("verbose", "sampler_progress")}
        return output_settings(getattr(self, "config", self), **options)

    if inspect.iscoroutinefunction(method):
        @wraps(method)
        async def wrapper(self, *args, **kwargs):
            with settings(self, args, kwargs):
                return await method(self, *args, **kwargs)
    else:
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with settings(self, args, kwargs):
                return method(self, *args, **kwargs)
    return wrapper

