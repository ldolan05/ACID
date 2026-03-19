.. _multiprocessing:

Multiprocessing
===============

Best Practices
--------------

The default multiprocessing setting is True for ACID, which means that ACID will automatically use all available CPU cores to run the MCMC sampler in parallel.
According to emcee documentation, they recommend setting the environment variable: OMP_NUM_THREADS=1. Our testing also showed this setting absolutely necessary for
ACID to avoid large transfer overheads. We also recommend setting the environment variable: MKL_NUM_THREADS=1 for similar reasons. 

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

Windows Support
---------------

Windows is not supported for multiprocessing as the mp context cannot be set to fork. If a windows system is detected, you will be warned that multiprocessing is not
supported and then automatically turned off. This project has otherwise not received extensive testing on Windows, there may be other unforseen issues.