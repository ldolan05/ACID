A.C.I.D (Accurate Continuum fItting and Deconvolution)
==============================================================

[![Tests](https://github.com/ldolan05/ACID/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ldolan05/ACID/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/ldolan05/ACID/badge.svg?branch=main)](https://coveralls.io/github/ldolan05/ACID?branch=main)
[![Documentation Status](https://app.readthedocs.org/projects/acid-code/badge/?version=stable)](https://acid-code.readthedocs.io/en/stable/?badge=stable)
[![PyPI version](https://img.shields.io/pypi/v/ACID-code.svg)](https://pypi.org/project/ACID-code/)
[![License](https://img.shields.io/github/license/ldolan05/ACID.svg)](https://github.com/ldolan05/ACID/blob/main/LICENSE)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1093%2Fmnras%2Fstae710-blue)](https://doi.org/10.1093/mnras/stae710)

The original ACID code was developed by Lucy Dolan as part of her PhD and published in 2024. Development has continued in this repository by Benjamin Cadell since October 2025 as part of his PhD.

Since 2024, the most signficant changes to ACID have been:
    - Updating packages and code to work with newer and stable versions of python.
    - Improving memory management so that ACID can be run on MacOS without crashes (ie extending compatibility to all POSIX systems)
    - Adding additional kwargs to ACID to tailor output, including verbosity settings, MCMC number of steps, multiprocessing switch, and more.
    - Utilising classes for both ACID and the result of ACID, allowing for analysis methods that can be found in the documentation.
    - Methods to extract fits formats for common instruments (e.g. ESPRESSO, HARPS, UVES) and to load them directly into ACID.
    - Updated documentation and examples

A more complete list of changes can be found in the [changelog](changelog.md).

The documentation will be kept up to date until at least 2029.

ACID is a technique that builds on traditional Least-Squares Deconvolution (LSD) by simultaneously fitting the stellar continuum and stellar line profile and performing LSD in effective optical depth. 

In a basic sense, ACID simulatenously fits the stellar continuum and profile using a combination of LSD and MCMC techniques. The spectra are then continuum corrected using this continuum fit. LSD is then run on the continuum corrected spectra to return high-resolution line profiles for each spectrum.

Installation
============

See the [Read the Docs documentation](https://acid-code.readthedocs.io/en/stable/installation.html) for installation instructions.
