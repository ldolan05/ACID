.. _using_acid:

Using ACID
-----------

These tutorials requires use of the example data included in the example_ folder.

.. _source: https://github.com/Benjamin-Cadell/ACID_v2/tree/main/example

Quickstart
=============

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
   import matplotlib.pyplot as plt
   import ACID_code_v2 as acid

   # choose a velocity grid for the final profile(s)
   deltav = acid.calculate_deltav(wavelength)   # velocity pixel size must not be smaller than the spectral pixel size - can use calculate_deltav function if unsure what this would be.
   velocities = np.arange(-25, 25, deltav)  

   # run ACID function
   result = acid.ACID(wavelength, spectrum, error, linelist, sn, velocities)
   
   # Plot results using the plot_profiles method
   result.plot_profiles()

   # Or plot profiles as with legacy ACID:
   # extract profile and errors
   profile = result[0, 0, 0]
   profile_error = result[0, 0, 1]
   # plot results
   plt.figure()
   plt.errorbar(velocities, profile, profile_error)
   plt.xlabel('Velocities (km/s)')
   plt.ylabel('Flux')
   plt.show()

Multiple frames
=============================

Multiple frames of data can be input to directly to ACID. ACID adjust these frames and performs the continuum fit on a combined spectrum (constructed from all frames).
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
   import matplotlib.pyplot as plt

   # choose a velocity grid for the final profile(s)
   deltav = acid.calc_deltav(wavelength)  
   velocities = np.arange(-25, 25, deltav)  

   # run ACID function
   result = acid.ACID(wavelengths, spectra, errors, linelist, sns, velocities)
   
   # plot results
   plt.figure()

   for frame in range(len(files)):
      profile = result[frame, 0, 0]
      profile_error = result[frame, 0, 1]
      plt.errorbar(velocities, profile, profile_error, label = '%s'%frame)

   plt.xlabel('Velocities (km/s)')
   plt.ylabel('Flux')
   plt.legend()
   plt.show()
 

Multiple wavelength ranges
=========================================

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
   import matplotlib.pyplot as plt

   # choose a velocity grid for the final profile(s)
   deltav = acid.calc_deltav(wavelength)  
   velocities = np.arange(-25, 25, deltav)

   # choose size of wavelength ranges (or chunks)
   wave_chunk = 25
   chunks_no = int(np.floor((max(wavelength)-min(wavelength))/wave_chunk))

   min_wave = min(wavelength)
   max_wave = min_wave+wave_chunk
   
   # create result array of shape (no. of frames, no. of chunks, 2, no. of velocity pixels)
   results = np.zeros((1, chunks_no, 2, len(velocities)))
   
   for i in range(chunks_no):

      # use indexing to select correct chunk of spectrum
      idx = np.logical_and(wavelength>=min_wave, wavelength<=max_wave)

      # run ACID function on specific chunk
      result = acid.ACID([wavelength[idx]], [spectrum[idx]], [error[idx]], linelist, [sn], velocities, all_frames=result, order=i)

      min_wave += wave_chunk
      max_wave += wave_chunk

   # reset min and max wavelengths
   min_wave = min(wavelength)
   max_wave = min_wave+wave_chunk

   # plot results
   plt.figure()
   for i in range(chunks_no): 

      # extract profile and errors
      profile = result[0, i, 0]
      profile_error = result[0, i, 1]

      plt.errorbar(velocities, profile, profile_error, label='(%s - %sÅ)'%(min_wave, max_wave))

      min_wave += wave_chunk
      max_wave += wave_chunk

   plt.xlabel('Velocities (km/s)')
   plt.ylabel('Flux')
   plt.legend()
   plt.show()

HARPS data
============

ACID can also be directly applied to HARPS data from DRS pipeline 3.5. To apply ACID in this way all files must be contained in the same directory.

If applying to 's1d' files, the corresponding 'e2ds' files must also be contained in this directory. 

If applying to 'e2ds' files, the corresponding blaze files must be present in this directory as indicated in the FITS header of the e2ds file.

This application only requires a filelist of the HARPS FITS files, a line list that covers the entire wavelength range and a chosen velocity range.
For 'e2ds' spectra the resolution of the profiles are optimized when the velocity pixel size is equal to the spectral resolution, i.e. 0.82 km/s.

.. code-block:: python

   import glob
   import numpy as np

   file_type = 'e2ds'
   filelist = glob.glob('/path/to/files/**%s**.fits')%file_type   # returns list of HARPS fits files
   linelist = '/path/to/files/example_linelist.txt'                            # Insert path to line list

   # choose a velocity grid for the final profile(s)
   deltav = 0.82     # velocity pixel size for HARPS e2ds data from DRS pipeline 3.5
   velocities = np.arange(-25, 25, deltav)  

These inputs can be input into the HARPS function of ACID (ACID_HARPS):

.. code-block:: python

   import ACID_code_v2 as acid

   # run ACID function
   BJDs, profiles, errors = acid.ACID_HARPS(filelist, linelist, velocities)

ACID computes and returns the Barycentric Julian Date, average profile and errors for each frame. The average profile is computed using a weighted mean across all orders.
