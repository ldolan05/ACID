A.C.I.D (Accurate Continuum fItting and Deconvolution)
==============================================================

[![Tests](https://github.com/ldolan05/ACID/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ldolan05/ACID/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/ldolan05/ACID/badge.svg?branch=main)](https://coveralls.io/github/ldolan05/ACID?branch=main)
[![Documentation Status](https://app.readthedocs.org/projects/acid-code/badge/?version=latest)](https://acid-code.readthedocs.io/en/latest/?badge=latest)
[![PyPI stable](https://img.shields.io/pypi/v/ACID-code.svg?label=PyPI%20stable)](https://pypi.org/project/ACID-code/)
[![PyPI latest](https://img.shields.io/github/v/release/ldolan05/ACID?include_prereleases&sort=semver&label=PyPI%20latest)](https://pypi.org/project/ACID-code/)
[![License](https://img.shields.io/github/license/ldolan05/ACID.svg)](https://github.com/ldolan05/ACID/blob/main/LICENSE)
[![Paper DOI](https://img.shields.io/badge/DOI-10.1093%2Fmnras%2Fstae710-blue)](https://doi.org/10.1093/mnras/stae710)

ACID is a technique that builds on traditional Least-Squares Deconvolution (LSD) by simultaneously fitting the stellar continuum and stellar line profile and performing LSD in effective optical depth. 

In a basic sense, ACID simulatenously fits the stellar continuum and profile using a combination of LSD and MCMC techniques.
The spectra are then continuum corrected using this continuum fit.
LSD is then run on the continuum corrected spectra to return high-resolution line profiles for each spectrum.

The original ACID code was developed by Lucy Dolan as part of her PhD and published in 2024.
Development has continued in this repository by Benjamin Cadell since October 2025 as part of his PhD.

*Please note that since the new version of ACID has not yet been fully released, the development remains in pre-release. Therefore, the "stable" (upto 0.1.0) versions of this code are still the ones released in 2024. For documentation and installation of the latest version, use the "latest" (for 2.0 alpha) versions.*

A list of changes can be found in the [index](https://acid-code.readthedocs.io/en/latest/index.html) of the documentation or in the [changelog](changelog.md) for a full list of changes.

The documentation will be kept up to date until at least 2029.

Installation
============

See the [Read the Docs documentation](https://acid-code.readthedocs.io/en/latest/installation.html) for installation instructions.
