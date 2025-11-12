#%%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import importlib, os, sys, re
os.chdir(os.path.dirname(__file__))
os.chdir("..")  # ensures we are in the main directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
from src import ACID_code_v2 as acid
from src.ACID_code_v2 import utils
importlib.reload(acid)

spec_file = fits.open('tests/test_data/sky_subtracted_s1d_A_26.fits')

spectrum = spec_file[0].data     # Spectral Flux
spectrum = np.array(spectrum, dtype='float')

hdr = spec_file[0].header
crval1 = hdr.get("CRVAL1")
cdelt1 = hdr.get("CDELT1") or hdr.get("CD1_1") or hdr.get("CDELT")
naxis1 = hdr.get("NAXIS1", spectrum.size)
wavelength = crval1 + np.arange(naxis1) * cdelt1

# --- Estimate errors ---
def rolling_sigma_error(wavelength, flux):
    flux = np.asarray(flux)
    n = len(flux)
    errors = np.zeros(n)

    for i in range(n):
        if i < 10:
            # Near start: use up to 15 points after current index
            start = 0
            end = min(i + 15, n)
        elif i > n - 11:
            # Near end: use up to 15 points before current index
            start = max(0, i - 15)
            end = n
        else:
            # Normal case: 10 before and 10 after
            start = i - 10
            end = i + 11  # +1 to include i+10

        # Calculate standard deviation in that window
        errors[i] = np.std(flux[start:end])

    return errors

error = rolling_sigma_error(wavelength, spectrum)

# --- Estimate S/N ---
sn = []
for i in range(len(spectrum)):
        sn_value = spectrum[i]/error[i]
        sn.append(sn_value)

sn = np.asarray(sn, dtype='float')


# cut out anything below 5000 angstroms because it's too noisy anyway
mask = (wavelength > 5000) & (wavelength < 5250)

# Apply the mask to all relevant arrays
wavelength = wavelength[mask]
spectrum   = spectrum[mask]
error      = error[mask]
sn         = sn[mask]

input_file = 'tests/test_data/linelist.txt'
output_file = 'tests/test_data/linelist_clean.csv'
output_file = 'example/example_linelist.txt'

"""
with open(input_file, 'r') as f:
    lines = f.readlines()

clean_lines = []
for line in lines:
    if not re.match(r"^'\w", line.strip()):
        continue
    
    parts = line.split(',')
    if len(parts) >= 10:
        try:
            wl = float(parts[1])      # column 1
            depth = float(parts[9])   # column 9
            # fabricate dummy values for the other 8 columns
            row = [0]*10
            row[1] = wl
            row[9] = depth
            clean_lines.append(','.join(map(str, row)) + '\n')
        except ValueError:
            continue

with open(output_file, 'w') as f:
    f.write("dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy,dummy\n")
    f.writelines(clean_lines)

print(f"✅ Fake 10-column line list saved to: {output_file}").
"""


linelist = output_file                            # Insert path to line list

# choose a velocity grid for the final profile(s)

velocities = np.arange(-50, 50, 0.82) # velocity pixel size for HARPS e2ds data from DRS pipeline 3.5

# run ACID function
# sn_scalar = np.nanmedian(sn[np.isfinite(sn) & (sn > 0)])
wavelength = wavelength[::15]
spectrum = spectrum[::15]
error = error[::15]

# wavelength, scaled_spec, scaled_error = utils.scale_spectra(wavelength, spectrum, error)
sn = acid.ACID().guess_SNR(wavelengths=wavelength, spectra=spectrum, errors=error)
# velocities = np.arange(-25, 25, acid.calc_deltav(wavelength))
# plt.errorbar(wavelength, scaled_spec, scaled_error)
# wavelength, spectrum, error = utils.scale_spectra(wavelength, spectrum, error)
# plt.show()
cl = acid.ACID()
# result = cl.run_ACID(wavelength, spectrum, error, sn, linelist, velocities, nsteps=2000)
result = acid.Result.load_result('tests/test_data/dusty_result.pkl')
result.plot_profile()

# # extract profile and errors
# profile = result[0, 0, 0]
# profile_error = result[0, 0, 1]

# # plot results
# plt.figure()
# plt.errorbar(velocities, profile, profile_error)
# plt.xlabel('Velocities (km/s)')
# plt.ylabel('Flux')
# plt.show()