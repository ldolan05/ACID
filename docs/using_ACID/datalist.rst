.. _datalist:

The DataList and loading FITS files
===================================

The :py:class:`ACID_code.DataList` class is a wrapper around a list of :py:class:`ACID_code.Data` instances, 
with some methods to easily run ACID on multiple frames of data (eg. for echelle spectra).

For now, loading FITS files should be done by you to get the wavelength, spectrum, error, and sn arrays into the correct format.
Over time, I may add another class to handle the loading of common FITS file formats for certain instruments (feel free to help with this).

The DataList class is the main way to run ACID on multiple orders of an echelle spectrum. It allows mapping the correct order number from the instrument
to the index of the DataList, and has a method to run ACID on all frames in the list with the same configuration. Also see the :py:class:`ACID_code.Data` API for more details.

The DataList can be initialized in two ways, either by inputting your spectra as arrays or by inputting a list of :py:class:`ACID_code.Data` instances.
The former is more useful for loading in spectra from FITS files, and the latter is more useful for loading in previously saved Data instances 
(see :ref:`data` for how to save and load Data instances).

Creating a DataList from Spectra
--------------------------------

The DataList class is initialized by inputting arrays of wavelengths, spectra, errors, and sn for each/one frame.
The dimensions should be (n_orders, n_frames, n_pixels) for multiple frames or (n_orders, n_pixels) for a single frame.
In other words, the last 1 or 2 dimensions should match the input format for :py:class:`ACID_code.Acid.ACID`.

Sometimes, fits files store their frames in shape (n_frames, n_orders, n_pixels), you can swap the axes with np.swapaxes(wavelengths, 0, 1) 
to get them in the correct shape. It is also possible to input orders with different numbers of pixels, in which case the wavelengths should be a list
of 2D arrays/lists, as numpy arrays do not allow ragged arrays.

.. code-block:: python

    from astropy.io import fits
    import glob

    # For the example, treat the multiple spectra as multiple orders with a single frame
    files = glob.glob('sample_spec_*.fits')  # three sample spectra in the current directory

    # create lists for wavelengths, spectra, errors and sn for all frames
    wavelengths = []
    spectra = []
    errors = []
    sn = []
    for file in files:
        spec_file = fits.open('%s'%file)
        wavelengths.append(spec_file[0].data)    # Wavelengths in Angstroms
        spectra.append(spec_file[1].data)        # Spectral Flux
        errors.append(spec_file[2].data)         # Spectral Flux Errors
        sn.append(spec_file[3].data)             # SN of Spectrum

    linelist = 'example_linelist.txt' # Insert path to line list
    velocities = np.arange(-25, 25, acid.calc_deltav(wavelengths[0]))

We can then tailor a few things to our needs if we wish. For example, the class uses a python list to index orders, but if we want to use those coming from the instrument,
we can also input it. We can also change order-specific configurations by inputting our own list of Config instances.

.. code-block:: python

    # Initialize the DataList, our wavelengths/spectra arrays are now 2D with shape (n_orders, n_pixels), n_orders = 3 in this example
    
    order_range = [20,21,22] # for example, if the orders in the fits files are labelled 20,21,22 instead of 0,1,2
    configs = acid.Config(max_steps=5000) # Use a single Config for all orders, with a custom max_steps value
    configs = [acid.Config(max_steps=5000) for _ in range(3)] # Or use a list of Configs, one for each order, with default values
    configs[1].update_hipri(poly_ord=5) # We can then, for example, for the second order only, increase the poly_ord to 5 instead of the default 3

    datalist = acid.DataList(
        wavelengths,
        spectra,
        errors,
        sn,
        velocities,
        linelist,
        verbose=2, # Set a global datalist-level verbosity
        # We highly recommend setting a directory to reduce memory load.
        save_dir="datalist/", # Can also specify and create save directory for the results, if None (default), results are only stored in memory.
        order_range=order_range,
        config=configs,
        ### Additional kwargs to pass to config for all orders ###
        cores=10, # You can use the **config_kwargs parameter to update more config parameters with low priority, meaning they will not overwrite the above configs
        )

    # If we want to do even more fine tuning, we can now index and change the datalist directly (now indexed to the order numbers provided)
    datalist[22].config.update_hipri(poly_ord=7) # For example, for the 22nd order, increase the poly_ord to 7 instead of the default 3 or the 5 we set for the second order

    # Run ACID on all orders in the datalist with the same configuration, if not already done
    datalist.run_ACID(allow_overwrite=False) # run but don't overwrite any existing results (this is actually default)

.. _fromdatalist:

Creating a DataList from a list of Data instances
-------------------------------------------------

The DataList can also be initialized by inputting a list of :py:class:`ACID_code.Data` instances by using the :py:function:`ACID_code.DataList.from_datalist` class method.
This is useful for loading in previously saved Data instances (see :ref:`data` for how to save and load Data instances).

.. code-block:: python

    data1 = acid.Data.load("data1") # Initialize a Data instance with the same format as the input data for the ACID method, and with a config
    data2 = acid.Data.load("data2") # Initialize more instances for another order/frame
    data3 = acid.Data.load("data3")
    data_list = [data1, data2, data3]

    datalist = acid.DataList.from_datalist(
        data_list,
        verbose=2, # Set a global datalist-level verbosity
        save_dir="datalist/", # Can also specify and create save directory for the results, if None (default), results are only stored in memory.
        )

    # Again, can tweak configs directly per order:
    datalist[22].config.update_hipri(poly_ord=7)

    # Run ACID if not already done
    datalist.run_ACID(allow_overwrite=False) # run but don't overwrite any existing results (this is actually default)

Using the DataList results
----------------------------

Once ACID has been run on the DataList with the :py:function:`ACID_code.DataList.run_ACID` method, 
the results for each order are stored in the DataList instance and can be accessed with the :py:function:`ACID_code.Data.result` property.

.. code-block:: python

    # All results can also be accessed via the data instances
    datalist[22].result.plot_profiles() # Access the result for the 22nd order, if it has been run, otherwise raises an error.

    # Combine the profiles across all orders, excluding the 21st order for example
    datalist.combine_profiles(exclude=21)
    datalist.plot_combined_profile() # Plot the combined profile across all orders, excluding the 21st order
    
    # Is a wrapper for ACID_code.Profiles.plot_fit() and passes the stored combined_profile through, kwargs are passed directly
    datalist.fit_profile()
