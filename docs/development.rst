.. _development:

Development and Issues
======================

I appreciate any and all help with the development of ACID and I am always open to suggestions and contributions. 
If you have any suggestions for features, improvements, or find any bugs, please raise an issue in the GitHub repository. 
If you would like to contribute code, please fork the repository and make a pull request with your changes.
I am also open to collaborations, so if you have an idea for a project using ACID, please feel free to reach out to me via email or GitHub.

Sampling
--------

``Sampler`` owns sampler selection, worker pools, continuation, and resume.
It shares the prepared ``Data`` instance with ``Acid`` and ``Result``:

.. code-block:: python

    from ACID_code import Sampler, Result

    sampler = Sampler(acid.data)  # after acid.ACID(..., run_mcmc=False)
    sampler.run_mcmc(1000, state=acid.data.initial_state)
    sampler.continue_sampling(max_steps=5000)
    result = Result(acid.data)

``run_mcmc`` and ``run_mcmc_until_converged`` dispatch to the configured backend
in ``mcmc/emcee.py``, ``mcmc/dynesty.py``, or ``mcmc/pymc.py``. Dynesty handles convergence internally
and uses ``nsteps`` as its live-point count. Existing ``Acid`` sampling methods
forward to this interface.

``MCMC`` in ``mcmc/mcmc.py`` defines both profile models, the log probability,
and their priors once. Its array and linear-algebra modules default to NumPy
and SciPy. With ``use_jax=True``, inputs are converted to JAX arrays and the same
probability methods are compiled. ``mcmc/jax.py`` only handles optional imports
and double-precision setup. New sampler implementations should reuse this model.

``mcmc/pymc.py`` lazily builds a PyTensor bridge to that JAX posterior and calls
PyMC's NumPyro NUTS implementation. It converts retained samples and diagnostics
to an emcee backend, so downstream code does not need PyMC objects. The bridge
follows `PyMC's JAX wrapping interface <https://www.pymc.io/projects/examples/en/latest/howto/wrapping_jax_function.html>`_.

Output and verbosity
--------------------

Keep stored settings in ``Config`` and use ``config.output`` for runtime output
decisions. It provides ``progress``, ``sampling``, ``plots``, and ``debug``.
This lets the same code work in standalone operations and quieter batch runs.
Decorate a method that uses ``self.config`` to apply its output settings;
nested calls inherit them automatically:

.. code-block:: python

    from ACID_code.diagnostics.logging import get_logger
    from ACID_code.diagnostics.output import with_output_settings
    from tqdm import tqdm

    logger = get_logger(__name__)

    class Calculation:
        @with_output_settings
        def run(self, items, verbose=None):
            logger.info("Starting calculation")
            for item in tqdm(items, disable=not self.config.output.progress):
                self.process(item)
            if self.config.output.debug:
                self.store_debug_data()

The decorator reads ``verbose`` and ``sampler_progress`` overrides from method
arguments, including positional arguments and forwarded keyword arguments. It
also works on Config methods and async methods. For part of a method, use
``with output_settings(config, verbose=verbose)`` instead. Add ``per_order=True``
for an individual order in a batch, suppressing its routine output at levels
0--2. Generator methods should use the context manager inside their body.

Within a decorated method or output context, call ``logger.info(...)`` directly
for routine status messages and ``logger.debug(...)`` for debugging details.
The logger filters these automatically; extra ``if config.verbose`` checks are
unnecessary. Outside a context, logging uses the package's default level rather
than any particular Config's verbosity. Orchestration methods such as DataList
therefore need explicit contexts around output for each stage.

Use ``config.output.progress`` for calculation bars, ``config.output.sampling``
for sampler bars, and ``config.output.plots`` for automatic diagnostic plots.
Read ``config.output`` rather than stored ``config.verbose`` for these decisions
so a quieter parent operation is respected. Explicit plotting requests still
run at any verbosity. Use ``warnings.warn`` for warnings; Python warning filters
control them independently of verbosity.

The runtime policy is implemented in ``diagnostics/output.py``. Scopes restore
their parent on normal return or exceptions, and are local to each thread or
async task. They do not change Config values, warning filters, or global logger
levels. Config creation and saved-data loading therefore have no output side
effects. Continue to use ``warnings.warn`` for warnings and ``get_logger`` for
log messages; explicit user logging configuration provides a global severity
limit even inside a verbose scope.

New worker threads or processes should receive the resolved ``Output`` object
and use ``with output_settings(output)`` if their work needs to inherit the
caller's policy.
Forked sampler processes inherit their parent's active scope; the likelihood
workers themselves do not create progress bars.
