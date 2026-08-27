.. ACID documentation master file, created by
   sphinx-quickstart on Tue Oct 31 11:39:06 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A.C.I.D (Accurate Continuum fItting and Deconvolution)
------------------------------------------------------

GitHub link: https://github.com/ldolan05/ACID

Since the original ACID code was developed for Lucy Dolan's PhD in 2024, it has had development continued by me (Benjamin Cadell) as part of my PhD from October 2025.
The version was originally forked from the original code and renamed ACID_v2_. The code has now been merged into this (the original) repository and will continue here.
The ACID_v2 repository will continue to exist on GitHub and will be kept up for reference, but all new development will appear here.

Since 2024, the most significant changes to ACID have been:

   - Updating packages and code to work with newer and stable versions of python.
   - Improving memory management so that ACID can be run on MacOS without crashes (i.e. extending compatibility to all POSIX systems)
   - Adding additional options to ACID to tailor the output, including verbosity settings, MCMC number of steps, multiprocessing switch, and many more.
   - Utilising classes for both ACID and the result of ACID, and adding analysis methods that can be found in the documentation.
   - Methods to extract fits formats for common instruments (e.g. ESPRESSO, HARPS, UVES) and to load them directly into ACID.
   - Updated documentation and examples

An more complete list of changes can be found in the CHANGELOG.md file in the repository. 

The documentation will be kept up to date until at least 2029.

For a full outline of ACID's algorithm and implementation, see the paper_ or view the package on GitHub_.

Please note that ACID now works as a class, and its usage is slightly different to the original ACID. Please see the User Guide and the Using ACID page for more information.

.. _emcee: https://emcee.readthedocs.io/en/
.. _paper: https://doi.org/10.1093/mnras/stae710
.. _GitHub: https://github.com/ldolan05/ACID
.. _ACID_v2: https://github.com/Benjamin-Cadell/ACID_v2

User Guide
----------

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   using_ACID
   API
   development

License and Attribution
-----------------------

Copyright 2023, Lucy Dolan.

ACID is free software made available under the MIT License.

If you make use of ACID in your work, please cite the original work by L. Dolan (Dolan et al. 2024): https://doi.org/10.1093/mnras/stae710.
