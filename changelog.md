# Changelog
All notable future changes to this project are documented here.

## [1.0.0] - 2025-11-20

### Notes
- I feel the changes made to ACID are in a space where I can call this a full release stable version. It is a huge overhaul of the final stable release before classes at 0.2.4
- The infrastructure for Acid and its Result have now moved to classes, with useful methods for analysis of the result.
- The LSD function has also moved to classes, however, it is not currently accessible to the user, which may change in a future update if its methods/attributes could be useful to the user.

### Added
- The Result class, which can plot the ACID profiles, MCMC walkers, corner plots, continue MCMC sampling, save and load the result
- The utils.py (accesible at top level of ACID_code_v2), which include calc_deltav, input rescaling, and SNR guessing.
- Improved ACID initialisation and function inputs, including verbosity, specification of cores, parallelisation, and other configurations.
- A few other optimisation improvements.
- Full backwards compatibility with ACIDv1 and ACIDv2 (0.2.4) with the ACID and ACID_HARPS functions.

### Changed
- Python requirement upgraded from 3.7 -> 3.13, allowing macOS users with Apple Silicon to run code.

### Fixed
- Calculation of the alpha matrix is chunked (with sizing dependent on available memory) to avoid crashes on macOS due to extremely high memory demands.

[1.0.0]: https://github.com/Benjamin-Cadell/ACID_v2/releases/tag/v1.0.0
