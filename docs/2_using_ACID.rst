.. _using_acid:

Tutorial - Using ACID
=====================

These tutorials requires use of the example data included in the example_ folder. See the repository cloning_ section to install the data.
You can find the script in example/tutorial_code.py

.. _source: https://github.com/Benjamin-Cadell/ACID_v2/tree/main/example
.. _cloning: https://acid-v2.readthedocs.io/en/stable/installation.html#cloning-the-repository

The architecture of ACID_v2 is slightly different to the original ACID code. ACID now works under the hood as a class (called Acid), rather than previously as a function.
The main method (ACID) of the Acid class is now a Result class with its own methods and attributes that allow useful analysis.
The legacy ACID and ACID_HARPS functions are still available for backwards compatibility, however it is recommended to use the Acid class for new applications.
The tutorials below walk through how to use ACID for a variety of applications using the new class structure.

Quickstart
---------------------

ACID returns LSD profiles based on input spectra. First, lets walk through an example for a single spectrum. 

ACID requires an input spectrum and stellar line list. An example spectrum and line list are contained in the 'example' directory of the source code.
In the 'example' directory we can set up our inputs are follows:

.. code-block:: python

   from astropy.io import fits

   spec_file = fits.open('sample_spec_1.fits')

   wavelength = spec_file[0].data   # Wavelengths in Angstroms
   spectrum = spec_file[1].data     # Spectral Flux
   error = spec_file[2].data        # Spectral Flux Errors
   sn = spec_file[3].data           # SN of Spectrum

   linelist = 'example_linelist.txt' # Insert path to line list

The stellar line list can also be obtained from VALD_ using their 'Extract Stellar' feature. You should input stellar parameters that correspond to your object and ensure that the wavelength range input covers the entire wavelength range of your spectrum. 
The detection threshold input to VALD must be less than 1/(3*SN) where SN is the signal-to-noise of the spectrum.

.. _VALD: http://vald.astro.uu.se/ 

We can then run ACID and plot the final results:

.. code-block:: python

   import numpy as np
   import ACID_code_v2 as acid

   # choose a velocity grid for the final profile(s)
   deltav = acid.calc_deltav(wavelength)   # velocity pixel size must not be smaller than the spectral pixel size - can use acid.calc_deltav function if unsure what this would be.
   velocities = np.arange(-25, 25, deltav)  

   # Initiate Acid
   Acid = acid.Acid(velocities=velocities, linelist_path=linelist_path)
   # Run ACID
   result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=2000)

   # Plot your final profile
   result.plot_profiles() # See documentation for more plot kwarg options

   # You can also plot walkers, corner plots, etc. See documentation.

   # If you feel the need to continue sampling, you can do so with:
   # result.continue_sampling(nsteps=2000) # And plot walkers to see the difference!

Multiple frames
---------------------

Multiple frames of data can be input to directly to ACID. ACID adjusts these frames and performs the continuum fit on a combined spectrum (constructed from all frames).
For this reason, frames must be from the same observation night where little variation is expected in the spectral continuum.
As in the previous example, we must first read in the data:

.. code-block:: python

   from astropy.io import fits
   import glob

   # finds sample files in 'example directory'. Each file is a different frame.
   files = glob.glob('sample_spec_*.fits')  
   
   # create lists for wavelengths, spectra, errors and sn for all frames
   wavelengths = []
   spectra = []
   errors = []
   sns = []

   for file in files:
      spec_file = fits.open('%s'%file)

      wavelengths.append(spec_file[0].data)    # Wavelengths in Angstroms
      spectra.append(spec_file[1].data)        # Spectral Flux
      errors.append(spec_file[2].data)         # Spectral Flux Errors
      sns.append(float(spec_file[3].data))     # SN of Spectrum

   linelist = 'example_linelist.txt' # Insert path to line list

Once the inputs have been constructed ACID can be applied and the results plotted. 

.. code-block:: python

   import ACID_code_v2 as acid
   import numpy as np

   # choose a velocity grid for the final profile(s)
   deltav = acid.calc_deltav(wavelength)  
   velocities = np.arange(-25, 25, deltav)  

   # run ACID function
   Acid = acid.Acid(velocities=velocities, linelist_path=linelist_path)
   result = Acid.ACID(wavelengths, spectra, errors, sns, nsteps=2000)

   # plot results
   result.plot_profiles()

   # Remember you can obtain the entire result array via:
   all_frames = result[:] # Which converts it to a numpy array of shape (frames, orders, 2, pixels)
   # or you can do
   all_frames = result.all_frames

Multiple Orders
---------------

As of ACID 1.4, the Acid class is designed to only handle a single order of ACID. This change was made to better handle the parallel processing of multiple
orders on a computer cluster. If you would like to run ACID for a full observation of an echelle spectrograph, see the Echelle Spectra section.

If you wish to run ACID for different wavelength ranges, then you can simply save the Result object for each wavelength range and process them separately. See
the Results and Plotting section.

Using Deterministic profile in MCMC fitting
-------------------------------------------

As of 1.4, ACID can infer the profile points at each MCMC step from the continuum parameters. This means the sampler does not fit the profile points, 
but instead fits only the continuum parameters and calculates the profile from the alpha matrix and the continuum model. The end result is the same, given
enough steps. As of 1.5, this setting now defaults to True and is recommended to be left on as it significantly decrease convergence time and computation 
time per step, while fully maintaining accuracy.

The feature can be disabled by setting deterministic_profile=False when calling ACID, which will match legacy behaviour.

.. code-block:: python

   # ... same code as before to set up data and run ACID ...

   Acid = acid.Acid(velocities=velocities, linelist_path=linelist)
   result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=2000, deterministic_profile=True)
   # Now runs ACID with deterministic profile points inferred from the continuum parameters.

Running ACID until convergence (section below) shows the vast improvement in convergence times with this setting.

Running ACID Until Convergence
------------------------------

As of version 1.4, ACID can also detect if the sampler has converged based on the computed autocorrelation time of the sampler. You can configure the following parameters
by passing them to the ACID function:

.. code-block:: text

   max_steps : IntLike, optional
      If set, the sampler will run until max_steps or convergence is reached by estimation using the emcee autocorrelation 
      time (tau). The sampler will check for convergence every 'check_interval' steps, and will require a minimum number 
      of checks ('min_checks') and a minimum tau factor ('min_tau_factor') before it can stop. The stopping criterion 
      is met when the change in tau is less than 'tau_tol' for all parameters. By default None, which means no maximum. 
      If a value is inputted, the nsteps parameter is ignored. The continue_sampling method in Result or Acid can still
      be used normally to continue sampling after either stopping criterion is reached.
   check_interval : IntLike, optional
      Interval (in steps) at which to check for MCMC convergence if max_steps is set, by default 1000. 
      Only used if max_steps is set.
   min_checks : IntLike, optional
      Minimum number of checks before MCMC can be stopped, by default 1. Only used if max_steps is set.
   min_tau_factor : IntLike, optional
      Minimum tau factor for MCMC stopping criterion, by default 50, which is the emcee recommendation, it's not
      recommend to set a value below 50 unless you want to force convergence for the deterministic_profile=False option.
      Only used if max_steps is set.
   tau_tol : float, optional
      Tolerance for tau convergence in MCMC stopping criterion, by default 0.05. Only used if max_steps is set.

.. code-block:: python

   # ... same code as before to set up data and run ACID ...

   Acid = acid.Acid(velocities=velocities, linelist_path=linelist)
   result = Acid.ACID(wavelength, spectrum, error, sn, max_steps=5000)

.. code-block:: text
   Iteration 1/5, last tolerance: inf>0.05, neff: 0.00<50: 100% 1000/1000 [00:04<00:00, 227.45it/s]
   Iteration 2/5, last tolerance: inf>0.05, neff: 0.00<50: 100% 1000/1000 [00:04<00:00, 234.49it/s]
   Iteration 3/5, last tolerance: 0.5674>0.05, neff: 7.00<50: 100% 1000/1000 [00:04<00:00, 219.90it/s]
   Iteration 4/5, last tolerance: 0.3913>0.05, neff: 7.00<50: 100% 1000/1000 [00:04<00:00, 228.78it/s]
   Iteration 5/5, last tolerance: 0.2756>0.05, neff: 7.00<50: 100% 1000/1000 [00:04<00:00, 222.20it/s]
   Not converged after reaching max steps of 5000. Final effective sample size: 7.00, final tolerance: 0.3184.
   Consider increasing max_steps.


The above code will still have a fully working sampler, which can plot the profiles as per normal (see the Results class, or the Other functions page for possible plotting options).
The sampler will give the following warning however:

.. code-block:: text

   The number of MCMC steps is less than 50 times the maximum autocorrelation time.
   The sampler may not have converged. Consider running more steps or checking the walker plots.
   The max autocorrelation time is 651.30, therefore the minimum number of steps should be roughly 32565.
   Disabling burnin from autocorrelation time, instead using burnin=steps-1000


If we turn on the deterministic profile feature, we see a significant improvement in convergence:

.. code-block:: python

   # ... same code as before to set up data and run ACID ...

   Acid = acid.Acid(velocities=velocities, linelist_path=linelist)
   result = Acid.ACID(wavelength, spectrum, error, sn, max_steps=5000, deterministic_profile=True)

.. code-block:: text

   Iteration 1/5, last tolerance: inf>0.05, neff: 0.00<50: 100% 1000/1000 [00:03<00:00, 282.63it/s]
   Iteration 2/5, last tolerance: inf>0.05, neff: 0.00<50: 100% 1000/1000 [00:03<00:00, 284.76it/s]
   Iteration 3/5, last tolerance: 0.0793>0.05, neff: 26.00<50: 100% 1000/1000 [00:03<00:00, 286.68it/s]
   Iteration 4/5, last tolerance: 0.0181<0.05, neff: 38.00<50: 100% 1000/1000 [00:03<00:00, 286.20it/s]
   Converged at step 4000. Final tolerance: 0.0066, final effective sample size: 51.00.
