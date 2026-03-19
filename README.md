A.C.I.D (Accurate Continuum fItting and Deconvolution)
==============================================================

Since the original ACID code was developed for Lucy Dolan's PhD in 2024, it has had devlopment continued by Benjamin Cadell from October 2025. The version was
originally forked from the original code and renamed ([ACID_v2](https://github.com/Benjamin-Cadell/ACID_v2)). The code has now been merged to the original and development
will continue here. The ACID_v2 repository will continue to exist on GitHub and will be kept up for reference, but all new development will be on the merged codebase here.

Since 2024, the most signficant changes to ACID have been:
    - Updating packages and code to work with newer and stable versions of python.
    - Improving memory management so that ACID can be run on MacOS without crashes (ie extending compatibility to all POSIX systems)
    - Adding additional kwargs to ACID to tailor output, including verbosity settings, MCMC number of steps, multiprocessing switch, and more.
    - Utilising classes for both ACID and the result of ACID, allowing for analysis methods that can be found in the documentation.
    - Methods to extract fits formats for common instruments (e.g. ESPRESSO, HARPS, UVES) and to load them directly into ACID.
    - Updated documentation and examples

An more complete list of changes can be found in the CHANGELOG.md file in the repository. 

The documentation will be kept up to date until at least 2029.

ACID is a technique that builds on traditional Least-Squares Deconvolution (LSD) by simultaneously fitting the stellar continuum and stellar line profile and performing LSD in effective optical depth. 

In a basic sense, ACID simulatenously fits the stellar continuum and profile using a combination of LSD and MCMC techniques. The spectra are then continuum corrected using this continuum fit. LSD is then run on the continuum corrected spectra to return high-resolution line profiles for each spectrum.

Installation
============

See the [ReadTheDocs](https://acid-code.readthedocs.io/en/stable/installation.html) for the instructions on installation.