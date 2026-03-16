.. _other:

Other ACID Functions
=====================

This page outlines some of the additional functions available in ACID_v2 outside of the main ACID functions,
some of which were highlighted in the Quickstart tutorial.

Continuing sampling
-------------------

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

Performing LSD
------------
The ACID package can perform traditional LSD in a standalone LSD class. This can be useful for comparison to ACID results or for quick-look analysis.
The LSD class can be used as an example as follows:

.. code-block:: python

   import ACID_code_v2 as acid
   import numpy as np
   from astropy.io import fits
   import matplotlib.pyplot as plt

   spec_file = fits.open('example/sample_spec_1.fits')

   wavelength = spec_file[0].data   # Wavelengths in Angstroms
   spectrum = spec_file[1].data     # Spectral Flux
   error = spec_file[2].data        # Spectral Flux Errors
   sn = spec_file[3].data           # SN of Spectrum

   linelist = 'example/example_linelist.txt' # Insert path to line list

   # choose a velocity grid for the final profile(s)
   deltav = acid.calc_deltav(wavelength)  
   velocities = np.arange(-25, 25, deltav)  

   # Initiate LSD class
   LSD = acid.LSD() # Can be initiated with an instance of Acid 

   # Perform LSD
   LSD.run_LSD(wavelength, spectrum, error, sn, linelist, velocities)

   # Extract useful attributes
   profile = LSD.profile
   profile_errors = LSD.profile_errors

   # Example plot
   plt.errorbar(velocities, LSD.profile, yerr=LSD.profile_errors, ecolor='red')
   plt.title('LSD Profile')
   plt.xlabel('Velocity (km/s)')
   plt.ylabel('LSD Profile Value')
   plt.show()

See the LSD API for more information on available methods and attributes.