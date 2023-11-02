A.C.I.D (Accurate Continuum fItting and Deconvolution)
==============================================================

ACID is a technique that builds on traditional Least-Squares Deconvolution (LSD) by simultaneously fitting the stellar continuum and stellar line profile and performing LSD in effective optical depth. 

In a basic sense, ACID simulatenously fits the stellar continuum and profile using a combination of LSD and MCMC techniques. The spectra are then continuum corrected using this continuum fit. LSD is then run on the continuum corrected spectra to return high-resolution line profiles for each spectrum.

For a full outline of ACID's algorithm and implementation, see our paper: (link to paper will be added)

Documentation: https://acid-code.readthedocs.io/en/latest/index.html

