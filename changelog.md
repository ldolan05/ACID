# Changelog
All notable future changes to this project are documented here.

## [1.0.4] - 2025-12-02

### Changed
- Minor bugfixes, code improvements for readability, and performance optimisations.
- Reset OMP threads after sampler call, which should in the future speed everything up with numpy optimisations in python 3.14

### Added
- Ability to add a seed to Acid for consistent results across runs. The seed now determining the result of the sampler, while other random calls in Acid are done using a seperate seed instance (np.default_rng)
- Added ability to inject _input_data to skip sections of Acid, though this remains only available for the sampler and in testing and for development purposes for now.

## [1.0.3] - 2025-11-24

### Fixed
- Fixed the non-multiprocessing case, as this was broken with the partial function. Log prob is now called directly and mcmc_utils initialised directly.

## [1.0.2] - 2025-11-21

### Changed
- Vastly sped up the MCMC calculation function using a faster numpy method. And slightly improved log_prob calculation. The speedup should be of the order of 1.5x-2x faster.

## [1.0.1] - 2025-11-20

### Added
- Result.plot_forward_model() to plot the forward model overlayed on the original data with the finished ACID profile. See the API for documentation.

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
