.. _using_acid:

Tutorial - Using ACID
=======================

These tutorials requires use of the example data included in the example_ folder. You can find the script in example/tutorial_code.py

.. _source: https://github.com/Benjamin-Cadell/ACID_v2/tree/main/example

The architecture of ACID_v2 is different to the original ACID code. ACID now works under the hood as a class (called Acid), rather than previously as a function.
The main result of Acid class or ACID function is also now a Result class with its own methods and attributes that allow for simple analysis.
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
 

Multiple wavelength ranges
---------------------------

In this example we will only consider one frame, however this example can be combined with the previous example to apply ACID to multiple frames and orders.
Firstly, we will read in the data (exactly how we did in the quickstart tutorial).

.. code-block:: python

   from astropy.io import fits

   spec_file = fits.open('sample_spec_1.fits')

   wavelength = spec_file[0].data   # Wavelengths in Angstroms
   spectrum = spec_file[1].data     # Spectral Flux
   error = spec_file[2].data        # Spectral Flux Errors
   sn = spec_file[3].data           # SN of Spectrum

   linelist = 'example_linelist.txt' # Insert path to line list

We can then loop through our desired wavelength ranges, run ACID and plot the final results. In this example we will split the wavelength ranges into 1000Å chunks.
When looping over wavelength ranges we also need to provide the result array ('all_frames') to keep all results in the same array.

.. code-block:: python

   import ACID_code_v2 as acid
   import numpy as np

   # choose a velocity grid for the final profile(s)
   deltav = acid.calc_deltav(wavelength)  
   velocities = np.arange(-25, 25, deltav)

   # choose size of wavelength ranges (or chunks)
   wave_chunk = 25
   chunks_no = int(np.floor((max(wavelength)-min(wavelength))/wave_chunk))

   min_wave = min(wavelength)
   max_wave = min_wave+wave_chunk
   
   # create result array of shape (no. of frames, no. of chunks, 2, no. of velocity pixels)
   result = np.zeros((1, chunks_no, 2, len(velocities)))
   
   # Initiate Acid
   Acid = acid.Acid(velocities=velocities, linelist_path=linelist_path)
   for i in range(chunks_no):

      # use indexing to select correct chunk of spectrum
      idx = np.logical_and(wavelength>=min_wave, wavelength<=max_wave)

      # You can recursively call ACID to fill in each order
      # In the future, this loop will be handled internally by ACID and this
      # example will be updated accordingly.
      result = Acid.ACID(wavelength[idx], spectrum[idx], error[idx], sn,
                         all_frames=result, order=i, nsteps=2000)

      min_wave += wave_chunk
      max_wave += wave_chunk

   # reset min and max wavelengths
   min_wave = min(wavelength)
   max_wave = min_wave+wave_chunk

   # plot results
   result.plot_profiles()
   # Reember you can save and load results via:
   # result.save_result("example/multi_order_result.pkl")
   # result = acid.Result.load_result("example/multi_order_result.pkl")

HARPS data
------------

ACID can also be directly applied to HARPS data from DRS pipeline 3.5. To apply ACID in this way all files must be contained in the same directory.

If applying to 's1d' files, the corresponding 'e2ds' files must also be contained in this directory. 

If applying to 'e2ds' files, the corresponding blaze files must be present in this directory as indicated in the FITS header of the e2ds file.

This application only requires a filelist of the HARPS FITS files, a line list that covers the entire wavelength range and a chosen velocity range.
For 'e2ds' spectra the resolution of the profiles are optimized when the velocity pixel size is equal to the spectral resolution, i.e. 0.82 km/s.

.. code-block:: python

   import glob
   import numpy as np

   file_type = 'e2ds'
   e2ds_files = glob.glob('tests/data/*e2ds_A*.fits') # Returns list of HARPS files
   linelist_path = 'example/example_linelist.txt'
   save_path = 'no save'
   order_range = np.arange(41, 43) # Specify which orders to run ACID on (here we do 41 and 42 as an example)

   # choose a velocity grid for the final profile(s)
   deltav = 0.82     # velocity pixel size for HARPS e2ds data from DRS pipeline 3.5
   velocities = np.arange(-25, 25, deltav)

These inputs can be input into the HARPS function of ACID (ACID_HARPS):

.. code-block:: python

   import ACID_code_v2 as acid

   # run ACID function
   Acid = acid.Acid(velocities=velocities, linelist_path=linelist_path)

   # Due to legacy behaviour, the function returns BJDs, profiles and errors separately when indexed,
   # not all_frames as in other examples. All frames can still be accessed via result.all_frames

   result = Acid.ACID_HARPS(filelist=e2ds_files, file_type='e2ds', save_path=save_path, nsteps=2000,
                            order_range=order_range)

   # BJDs, profiles, profile_errors = result
   all_frames = result.all_frames
   result.plot_profiles()

ACID computes and returns the Barycentric Julian Date, average profile and errors for each frame. The average profile is computed using a weighted mean across all orders.
