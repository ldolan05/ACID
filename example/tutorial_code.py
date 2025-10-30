#%%
from astropy.io import fits
import importlib, os, sys
import numpy as np
import matplotlib.pyplot as plt
os.chdir(os.path.dirname(__file__))
os.chdir("..")  # ensures we are in the main directory
try:
    import ACID_code_v2 as acid
except:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    sys.path.append(PROJECT_ROOT)
    from src import ACID_code_v2 as acid
    print("pip module failed to import, imported from local instead")
importlib.reload(acid)

spec_file = fits.open('example/sample_spec_1.fits')

wavelength = spec_file[0].data[::5]   # Wavelengths in Angstroms
spectrum = spec_file[1].data[::5]     # Spectral Flux
error = spec_file[2].data[::5]        # Spectral Flux Errors
sn = spec_file[3].data[::5]           # SN of Spectrum

linelist = 'example/example_linelist.txt' # Insert path to line list

# choose a velocity grid for the final profile(s)
deltav = 0.82   # velocity pixel size must not be smaller than the spectral pixel size
velocities = np.arange(-25, 25, deltav)

# run ACID function
result = acid.run_ACID([wavelength], [spectrum], [error], [sn], linelist, velocities,
                   parallel=True, cores=None, nsteps=4000, verbose=True)

# extract profile and errors
profile = result[0, 0, 0]
profile_error = result[0, 0, 1]

# plot results
plt.figure()
plt.errorbar(velocities, profile, profile_error)
plt.xlabel('Velocities (km/s)')
plt.ylabel('Flux')
plt.show()