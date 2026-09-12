.. _installation:

Installation and Setup
======================

Installing the package
------------------------

ACID has been tested in Python 3.13 and 3.14, running fastest on 3.14. It is recommended to install ACID in a new conda environment to avoid dependency conflicts.

In this example we create an environment named 'acid' and activate it using:

.. code-block:: bash

    conda create -n acid python=3.14
    conda activate acid

Once the environment has been activated ACID can be installed using pip_:

.. _pip: https://pip.pypa.io/en/stable/ 

.. code-block:: bash

    pip install ACID_code

.. _source:

This will install ACID into your environment with all of its dependencies.

Optional JAX acceleration
-------------------------

The MCMC log-probability calculation has an optional JAX backend. JAX is not a required dependency and the normal NumPy/SciPy backend remains the default.
For CPU use, install JAX using its standard installation command::

    pip install -U jax

When developing from a clone of ACID, ``pip install -e ".[jax]"`` is a convenience which installs the editable ACID checkout and the same CPU JAX dependency together.
The ``jax`` package selects a compatible ``jaxlib`` dependency, so they should not normally be listed or versioned separately.

Accelerator installations depend on the cluster hardware. For example, current NVIDIA CUDA 13 installations use::

    pip install -U "jax[cuda13]"

Consult the `JAX installation documentation <https://docs.jax.dev/en/latest/installation.html>`_ for CUDA 12, ROCm, TPU, driver, operating-system, and architecture requirements.
Once installed, enable the backend for an ACID run with ``use_jax=True``. If JAX cannot be imported, ACID will issue a warning and continue with NumPy/SciPy.

With ``parallel=True``, ACID uses a thread pool when JAX is requested.
Forking after JAX has initialised its runtime can deadlock workers during compilation or execution.
Threads share one model and compiled kernel per sampler, and work with interactive scripts without a ``__main__`` guard.
``cores`` controls the number of worker threads; JAX also manages its own internal execution threads.
This applies in SLURM too; the existing environment-variable requirements for parallel runs still apply.
Compare with ``parallel=False`` on your workload, since coordinating workers can outweigh the benefit for fast kernels.

With ``use_jax=False``, ACID uses the original ``fork`` process pool, even if JAX was used earlier in the session.

To try batching emcee walkers in a single JAX call, pass ``use_jax=True, vectorize=True``.
This uses ``jax.jit(jax.vmap(...))`` and emcee's vectorized interface, bypassing pools regardless of ``parallel`` and ``cores``.
It defaults to False because batching can be slower on CPU. Without JAX it falls back to evaluating the walkers with NumPy;
dynesty ignores ``vectorize``.
emcee usually evaluates subsets of the ensemble, so the small batches in a 15-walker run can still favour threads.

The measured speedup applies to each log-probability evaluation after its first JIT compilation. A complete MCMC step also includes sampler proposal and coordination overhead,
so its overall improvement will be smaller and depends on the fraction of runtime spent evaluating log probability.

.. _cloning:

Cloning the repository
------------------------
In order to use the example data (for the tutorials) or the test suite, ACID will need to be installed from the source (i.e. directly from the GitHub repository).
This can be done by cloning the source repository. All examples and tests attempt first to import from your pip installation.
If this fails they will attempt to import from the local source directory instead.

.. code-block:: bash

    git clone https://github.com/ldolan05/ACID.git
    cd ACID

.. _test:

Testing the installation
-------------------------

Test your installation by running our test file in the test directory. This may take a while (~1-2 min) but should run without any errors if the installation has gone smoothly.
The test file will attempt to run all of the methods and functions in ACID. If any of these fail and you believe that this is due to the source code and not your installation, please raise an issue on GitHub.

.. code-block:: bash
    
    conda install pytest
    python tests/tests.py
