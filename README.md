A.C.I.D v2 (Accurate Continuum fItting and Deconvolution)
==============================================================

ACID_v2 (https://github.com/Benjamin-Cadell/ACID_v2) is a fork of ACID (https://github.com/ldolan05/ACID) from the work of Lucy Dolan for her PhD. ACID_v2 improves on ACID by:

    - Updating packages and code to work with newer and stable versions of python.

    - Improving memory management so that ACID can be run on MacOS without crashes (ie extending compatibility to all POSIX systems)

    - Adding additional kwargs to ACID to tailor output, including verbosity settings, MCMC number of steps, multiprocessing switch, and more.
    
    - Utilising classes for both ACID and the result of ACID, allowing for analysis methods that can be found in the doccumentation.

The mathematical functions and method remain the same as ACID and are outlined in Dolan et al. 2024 (https://academic.oup.com/mnras/article/529/3/2071/7624678).

The documentation will be kept up to date with the latest function descriptions until at least 2029. Please note that this install is incompatible with Windows systems.

ACID is a technique that builds on traditional Least-Squares Deconvolution (LSD) by simultaneously fitting the stellar continuum and stellar line profile and performing LSD in effective optical depth. 

In a basic sense, ACID simulatenously fits the stellar continuum and profile using a combination of LSD and MCMC techniques. The spectra are then continuum corrected using this continuum fit. LSD is then run on the continuum corrected spectra to return high-resolution line profiles for each spectrum.
