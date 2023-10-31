.. ACID documentation master file, created by
   sphinx-quickstart on Tue Oct 31 11:39:06 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A.C.I.D (Accurate Continuum fItting and Deconvolution)
==============================================================

ACID is a technique that builds on traditional Least-Squares Deconvolution (LSD) by simultaneously fitting the stellar continuum and stellar line profile and performing LSD in effective optical depth. 

In a basic sense, ACID simulatenously fits the stellar continuum and profile using a combination of LSD and MCMC techniques (MCMC implementation through emcee_). The spectra are then continuum corrected using this continuum fit. LSD is then run on the continuum corrected spectra to return high-resolution line profiles for each spectrum.

For a full outline of ACID's algorithm and implementation, see our paper or view the package on GitHub_.

.. _emcee: https://emcee.readthedocs.io/en/
.. _paper: https://github.com/ldolan05/ACID 
.. _GitHub: https://github.com/ldolan05/ACID 

User Guide
===================
.. toctree::
   :maxdepth: 2

   installation.rst
   using_ACID.rst
   tutorials.rst
   ACID.rst

License and Attribution
================================

Copyright 2022, Lucy Dolan.

If you make use of ACID in your work please cite our paper (L.Dolan et al, 2023).

