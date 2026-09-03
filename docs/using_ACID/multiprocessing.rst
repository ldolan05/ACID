.. _multiprocessing:

Multiprocessing
===============

Best Practices
--------------

The default multiprocessing setting is True for ACID. ACID uses loky's affinity-aware CPU count and will not create more workers than the sampler can use for each proposal batch.
This avoids launching a machine-wide set of idle worker interpreters on high-core-count Linux systems.
According to emcee documentation, they recommend setting the environment variable: OMP_NUM_THREADS=1. Our testing also showed this setting absolutely necessary for
ACID to avoid large transfer overheads. ACID applies the equivalent one-thread limits for MKL, OpenBLAS, BLIS, Apple's vecLib, and NumExpr before starting loky workers.

On most standard machines, you can set these two variables to false just before the start of multiprocessing (which ACID does), but in some
environments, for unknown reasons, eg. some HPC environments, they must be set either in the terminal with:

.. code-block:: bash

   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1

or right at the top of the page before ALL other imports:

.. code-block:: python

   import os
   os.environ["OMP_NUM_THREADS"] = "1"
   os.environ["MKL_NUM_THREADS"] = "1"
   ... import numpy as np ... etc.

If they are not set before importing numpy, they will not correctly apply, and running MCMC in parallel will be excrutiatingly slow. If you ever experience unexpectdly
slow mcmc times, this is likely why.

We simply just recommend for all users to make sure these are set correctly before ACID is run. ACID will raise an exception in SLURM environments if they are not
set when multiprocessing is True, but in other environments, if they are not set, ACID will run but will be extremely slow. You have been warned!

Reminder: you can always turn off multiprocessing in ACID by setting parallel=False:

.. code-block:: python
   
   acid = Acid(...)
   result = acid.ACID(
      ..., # other inputs
      parallel=False
   )

Platform support
----------------

ACID uses loky's cross-platform process executor. Unlike Python's standard ``spawn`` multiprocessing context, it does not import the user's main script in each worker.
Parallel ACID calls therefore do not require an ``if __name__ == "__main__"`` guard on Windows or macOS.
Loky workers also reuse the parent's Matplotlib configuration directory, avoiding a separate font-cache build in every process when the default directory is not writable.

Inside a SLURM allocation, detected from ``SLURM_JOB_ID``, ACID instead uses Python's ``fork`` multiprocessing context. This retains the original HPC behaviour and avoids
loky's fork/exec startup having to transfer the complete ACID ``Data`` instance into each newly executed worker. The fallback is automatic; no additional configuration is required.
