.. _results:

Results and Plotting
====================

By default, the ACID method returns a Result object which contains many methods to analyse the results and simple tools to return the profile and errors.

.. code-block:: python

    import ACID_code as acid
    spec_file = fits.open('sample_spec_1.fits')
    wavelength = spec_file[0].data
    spectrum = spec_file[1].data
    error = spec_file[2].data
    sn = spec_file[3].data
    linelist = 'example_linelist.txt'
    acid = acid.Acid(velocities, linelist)

    # We will use the above result object for the rest of this page
    result = acid.ACID(wavelength, spectrum, error, sn, max_steps=5000)

Using the Results Class
-----------------------

The final profiles can be obtained by indexing the class with the first axis corresponding to the frame number, and the second axis corresponding to either
the profile values (0) or the errors (1).

.. code-block:: python

    profiles = results[0,0]
    errors = results[0,1]

    # They call also just be called from the class attributes, remember to index by frame number (0 if the ACID inputs were 1D):
    profiles = result.profiles[0]
    errors = result.profiles_errors[0]

Saving the Result
-----------------

The Result class returned by ACID contains a method to save the entire Result object to a .pkl file for later use.
This can be done using the Result.save_result() method.

.. code-block:: python

    # Save the result to a .pkl file
    result.save_result('example_result.pkl', store_sampler=True)
    ...
    # Later, load the result back:
    result = acid.Result.load_result('example_result.pkl')

The save_result method does not actually store the instance of the class as previously we had issues pickling class instances that stored samplers.
This method instead stores the Result internal dictionary as a pickle, including the backend of the emcee MCMC sampler (as a dictionary) if 
store_sampler=True (default is True). For this reason, if you try to open the dictionary yourself with pickle.load() and without using the class method, 
you will run into errors. If the sampler is not stored, some of the methods when loading the result will not work (eg. plotting walkers).

The :py:class:`Result` class also handles the storing of the :py:class:`Data` class (again, as a dictionary). See the :ref:`data` for more info.

Plotting
---------

The Result class contains a number of plotting methods to visualise the results of ACID. These include:

- plot_profiles(): Plots the final LSD profiles returned by ACID. Can plot multiple profiles if multiple spectra were input.

- plot_walkers(): Plots the MCMC walkers for the continuum fit parameters.

- plot_corner(): Plots a corner plot of the posterior distributions of the continuum fit parameters.

- plot_forward_model(): Plots the forward model fit to the data.

- plot_autocorrelation(): Plots the autocorrelation of the MCMC chains for the continuum fit parameters.

- plot_acf(): Plots the autocorrelation function for each parameter, averaged across walkers. This is less useful than the above, but kept as it was a part of the emcee example.

These plotting functions have a number of keyword arguments to tailor the plots to your needs. See the documentation for more information on these.

All of the plots which plot parameters (eg. walkers, corner, autocorrelation) will plot the parameters for the continuum parameters. Where deterministic_profile=False
and the sampler also fitted the profile, the parameters of the first, last and max profile point are shown to save space.

.. code-block:: python

    import ACID_code as acid

    result = acid.Result.load_result('example_result.pkl')

    result.plot_profiles()
    result.plot_walkers()
    result.plot_corner()
    result.plot_forward_model()
    result.plot_autocorrelation()
    result.plot_acf()

These functions can be called directly from the Result object returned by ACID, or from a Result object loaded from a saved .pkl file.
