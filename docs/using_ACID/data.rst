.. _data:

Using the Data class
====================

Getting the Data class
-----------------------

The :py:class:`ACID_code.Data` class stores the data for all calculations done in ACID, including the final resulting profiles. It is used to work
seamlessly with the :py:class:`ACID_code.Acid`, :py:class:`ACID_code.LSD`, and :py:class:`ACID_code.Result` classes to get configuration and store data.
The most useful examples of the usage of the class are shown below if desired by the user.

Remember that all attributes and methods of Data can be found in the :py:class:`ACID_code.Data` API.

.. code-block:: python

   from astropy.io import fits
   import ACID_code as acid

   spec_file = fits.open('sample_spec_1.fits')
   wavelength = spec_file[0].data   # Wavelengths in Angstroms
   spectrum = spec_file[1].data     # Spectral Flux
   error = spec_file[2].data        # Spectral Flux Errors
   sn = spec_file[3].data           # SN of Spectrum
   linelist = 'example_linelist.txt' # Insert path to line list

   Acid = acid.Acid(velocities=velocities, linelist=linelist)
   result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=5000)

   data = result.data  # Acid and Result share the Data class, so this is the same as Acid.data

Full line lists
---------------

Use ``full=True`` to also read the species/ion and Lande factor from VALD
columns 0 and 8. The default reads only wavelengths and depths.

.. code-block:: python

    lines = acid.LineList("example_linelist.txt", full=True)
    wavelengths, depths, species, lande = lines
    species = lines["spec_ion"]       # also lines[2]
    lande = lines["lande_factor"]     # also lines[3]
    Acid = acid.Acid(velocities=velocities, linelist=lines)

The column order is ``wavelengths``, ``depths``, ``spec_ion``, ``lande_factor``.
You can also provide a dictionary with these keys or a list/array of columns
in this order. Species are strings; the other columns are float arrays.
All columns share one wavelength sort and validity mask, so removing an invalid
row also removes its metadata. Non-finite numeric values, empty species,
nonpositive wavelengths, and depths outside ``[0, 1)`` are invalid.

``full=True`` requires all four columns. Metadata already present in an
in-memory input is preserved even with the default ``full=False``. This lets
``LineList(existing_lines)``, Data assignment, and save/load retain it.
Saved line lists remain dictionaries of arrays, including the extra columns.
Two-column line lists still unpack into just wavelengths and depths.

Plotting
---------
You can plot any of the three pre-mcmc set of plots with the following. See the API for the full list of optional inputs to each method.

.. code-block:: python

    # Plot the initial continuum fit
    data.plot_continuum_fit("initial")

    # Plot the continuum fit after masking
    data.plot_continuum_fit("masked")

    # Plot the residual mask results
    data.plot_residual_masking()

Getting run times
-----------------

You can also extract run times:

.. code-block:: python

    print(data.initialisation_time)  # time taken for initialization
    print(data.mcmc_time)  # time taken for MCMC sampling
    print(data.get_profiles_time)  # time taken to get profiles
    print(data.full_run_time)  # total time for the full run

Saving and Loading
-------------------

You can also save and load the data class using its method (It's not recommended to directly pickle yourself, 
see "Saving the Result" in :ref:`result` for an explanation why).

.. code-block:: python

    data.save("data.pkl")
    new_data = acid.Data.load("data.pkl")

The default ``config.order`` is ``None`` (unset). Loading a file such as
``order_20/data.pkl`` assigns order ``20`` if no order was saved. An explicitly
saved order, including ``0``, is preserved even if the containing folder has a
different order number. Files outside an ``order_<integer>`` directory keep an
unset order. Saving alone does not assign an order.

The loaded data can now be directly input into Acid to load the previous configuration and results.

.. code-block:: python

    # Load into a new Acid instance
    Acid = acid.Acid(data=new_data)

    # Or load into a Results instance (note that the sampler will not be available)
    result = acid.Result(new_data)

    # Or see the next section on DataLists for putting Data instances into lists for running Acid on echelle spectra
