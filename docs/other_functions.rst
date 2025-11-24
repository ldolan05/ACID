.. _other_functions:

Other ACID Functions
=====================

This page outlines some of the additional functions available in ACID_v2 outside of the main ACID functions,
some of which were highlighted in the Quickstart tutorial.

Saving the Result class
-----------------------

The Result class returned by ACID contains a method to save the entire Result object to a .pkl file for later use.
This can be done using the Result.save_result() method.

For the remainder of this page we will assume you have already run ACID and have a Result object called 'result'.

.. code-block:: python

    import ACID_code_v2 as acid
    from astropy.io import fits

    skips = 3 # Example of skipping every 3rd pixel to reduce computation time for this tutorial

    # Use all the same code as in the Quickstart tutorial to set up inputs and run ACID
    spec_file = fits.open('example/sample_spec_1.fits')
    wavelength = spec_file[0].data[::skips]   # Wavelengths in Angstroms
    spectrum = spec_file[1].data[::skips]     # Spectral Flux
    error = spec_file[2].data[::skips]        # Spectral Flux Errors
    sn = spec_file[3].data                    # SN of Spectrum
    linelist_path = 'example/example_linelist.txt' # Insert path to line list

    velocities = np.arange(-25, 25, acid.calc_deltav(wavelength))  # Velocity grid

    # Initiate Acid
    Acid = acid.Acid(velocities, linelist_path)
    # Run ACID
    result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=2000)

    result.save_result('example_result.pkl')

    # The result can be reloaded later using:
    loaded_result = acid.load_result('example_result.pkl')

    # And later for example, to plot the profiles again:
    # loaded_result.plot_profiles() (see plotting section below)

This result is used for the remainder of this page.

Plotting
---------

The Result class contains a number of plotting methods to visualise the results of ACID. These include:

- plot_profiles(): Plots the final LSD profiles returned by ACID. Can plot multiple profiles if multiple spectra were input.

- plot_walkers(): Plots the MCMC walkers for the continuum fit parameters.

- plot_corner(): Plots a corner plot of the posterior distributions of the continuum fit parameters.

These plotting functions have a number of keyword arguments to tailor the plots to your needs. See the documentation for more information on these.

Currently, plot_walkers plots the last 10 parameters to include continuum parameters and the last few velocity profile points.
Similarly, plot_corner only plots the last 10 parameters. Future versions may include options to select which parameters to plot.

.. code-block:: python

    import ACID_code_v2 as acid

    result = acid.Result.load_result('example_result.pkl')

    result.plot_profiles()
    result.plot_walkers()
    result.plot_corner()

These functions can be called directly from the Result object returned by ACID, or from a Result object loaded from a saved .pkl file.
Note however that if multiple orders or frames were calculated with ACID,
the plotting functions must be called from the Result object returned by ACID,
as the saved .pkl file cannot (currently) store multiple orders/frames.

Continuing sampling
---------------------

If you wish to continue sampling the MCMC after the initial run of ACID, you can do so using the Result.continue_sampling() method.
This method takes the same keyword arguments as the nsteps argument in ACID, allowing you to specify how many additional steps to run.

.. code-block:: python

    import ACID_code_v2 as acid

    result = acid.Result.load_result('example_result.pkl')

    # Continue sampling for an additional 2000 steps
    result.continue_sampling(nsteps=2000)

    # You can then plot the updated walkers and corner plots
    result.plot_walkers()
    result.plot_corner()

This allows you to extend the MCMC sampling if you feel that the initial number of steps was insufficient for convergence.
Note again that if multiple orders or frames were calculated, this method must be called from the Result object returned by ACID,
not from a saved .pkl file.
