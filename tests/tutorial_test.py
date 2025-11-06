#%%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob, os, importlib, sys, pickle
os.chdir(os.path.dirname(__file__))
os.chdir("..")  # ensures we are in the main directory
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)
from src import ACID_code_v2 as acid_v2
from ACID_code import ACID as acid_v1


spec_file = fits.open('example/sample_spec_1.fits')

wavelength = spec_file[0].data   # Wavelengths in Angstroms
spectrum = spec_file[1].data     # Spectral Flux
error = spec_file[2].data        # Spectral Flux Errors
sn = spec_file[3].data           # SN of Spectrum

linelist = 'example/example_linelist.txt' # Insert path to line list

# choose a velocity grid for the final profile(s)
deltav = acid_v1.calc_deltav(wavelength)  # velocity pixel size must not be smaller than the spectral pixel size
velocities = np.arange(-25, 25, deltav)

result = acid_v2.run_ACID(wavelength, spectrum, error, sn, linelist, velocities, nsteps=2000)

# v1_data = pickle.load(open('tests/test_data/v1.pkl', 'rb'))
# v2_data = pickle.load(open('tests/test_data/v2.pkl', 'rb'))

# # print(v1_data[3])
# # print(v2_data[3])
# # print(np.allclose(v1_data[3], v2_data[3]))
# # print(v1_data[3] - v2_data[3])
# # print(v2_data[3].shape)
# mat_v1 = v1_data[3]
# mat_v2 = v2_data[3]
# diff_matrix = np.abs(mat_v1 - mat_v2)
# positive_mask = diff_matrix > 0
# positive_indices = np.nonzero(positive_mask)[1]
# positive_differences = diff_matrix[positive_mask]
# x = positive_indices
# plt.plot(x, positive_differences, '.', label='Alpha matrix Differences')
# plt.yscale('log')
# plt.legend(loc="lower left")
# plt.show()


# run ACID function
# velocities, profile, profile_errors, alpha, x, y, yerr, poly_inputs, fluxes_order1, flux_error_order1 = acid_v1.ACID([wavelength], [spectrum], [error], linelist, [sn], velocities)
# velocities2, profile2, profile_errors2, alpha2, x2, y2, yerr2, poly_inputs2, fluxes_order2, flux_error_order2 = acid_v2.run_ACID(wavelength, spectrum, error, sn, linelist, velocities)

# print(np.all(velocities==velocities2))
# print(np.all(profile==profile2))
# print(np.all(profile_errors==profile_errors2))
# print(np.all(alpha==alpha2))
# print(np.all(x==x2))
# print(np.all(y==y2))
# print(np.all(yerr==yerr2))
# print(np.all(poly_inputs==poly_inputs2))
# print(np.all(fluxes_order1==fluxes_order2))
# print(np.all(flux_error_order1==flux_error_order2))
# pickle.dump([velocities, profile, profile_errors, alpha, x, y, yerr, poly_inputs, fluxes_order1, flux_error_order1], open('tests/test_data/v1.pkl', 'wb'))
# pickle.dump([velocities2, profile2, profile_errors2, alpha2, x2, y2, yerr2, poly_inputs2, fluxes_order2, flux_error_order2], open('tests/test_data/v2.pkl', 'wb'))

# def quickstart():
#     spec_file = fits.open('example/sample_spec_1.fits')

#     wavelength = spec_file[0].data   # Wavelengths in Angstroms
#     spectrum = spec_file[1].data     # Spectral Flux
#     error = spec_file[2].data        # Spectral Flux Errors
#     sn = spec_file[3].data           # SN of Spectrum

#     linelist = 'example/example_linelist.txt' # Insert path to line list

#     # choose a velocity grid for the final profile(s)
#     deltav = acid_v1.calc_deltav(wavelength)  # velocity pixel size must not be smaller than the spectral pixel size
#     velocities = np.arange(-25, 25, deltav)

#     # run ACID function
#     result_v1 = acid_v1.ACID(wavelength, spectrum, error, sn, linelist, velocities, nsteps=2000)
#     result_v2 = acid_v2.run_ACID(wavelength, spectrum, error, sn, linelist, velocities, nsteps=2000)

#     # extract profile and errors
#     profile = result_v1[0, 0, 0]
#     profile_error = result_v1[0, 0, 1]

#     # plot results
#     plt.figure()
#     plt.errorbar(velocities, profile, profile_error)
#     plt.xlabel('Velocities (km/s)')
#     plt.ylabel('Flux')
#     plt.show()

# quickstart()

# def multiple_frames():

#     # finds sample files in 'example directory'. Each file is a different frame.
#     files = glob.glob('example/sample_spec_*.fits')

#     # create lists for wavelengths, spectra, errors and sn for all frames
#     wavelengths = []
#     spectra = []
#     errors = []
#     sns = []

#     for file in files:
#         spec_file = fits.open('%s'%file)

#         wavelengths.append(spec_file[0].data)    # Wavelengths in Angstroms
#         spectra.append(spec_file[1].data)        # Spectral Flux
#         errors.append(spec_file[2].data)         # Spectral Flux Errors
#         sns.append(float(spec_file[3].data))     # SN of Spectrum

#     linelist = 'example/example_linelist.txt' # Insert path to line list

#     # choose a velocity grid for the final profile(s)
#     deltav = acid.calc_deltav(wavelengths[0])
#     velocities = np.arange(-25, 25, deltav)

#     # run ACID function
#     result = acid.run_ACID(wavelengths, spectra, errors, sns, linelist, velocities, nsteps=2000)

#     # plot results
#     plt.figure()

#     for frame in range(len(files)):
#         profile = result[frame, 0, 0]
#         profile_error = result[frame, 0, 1]
#         plt.errorbar(velocities, profile, profile_error, label = '%s'%frame)

#     plt.xlabel('Velocities (km/s)')
#     plt.ylabel('Flux')
#     plt.legend()
#     plt.show()

# def multiple_orders():
#     spec_file = fits.open('example/sample_spec_1.fits')

#     wavelength = spec_file[0].data   # Wavelengths in Angstroms
#     spectrum = spec_file[1].data     # Spectral Flux
#     error = spec_file[2].data        # Spectral Flux Errors
#     sn = spec_file[3].data           # SN of Spectrum

#     linelist = 'example/example_linelist.txt' # Insert path to line list

#     # choose a velocity grid for the final profile(s)
#     deltav = acid.calc_deltav(wavelength)  
#     velocities = np.arange(-25, 25, deltav)

#     # choose size of wavelength ranges (or chunks)
#     wave_chunk = 25
#     chunks_no = int(np.floor((max(wavelength)-min(wavelength))/wave_chunk))
#     min_wave = min(wavelength)
#     max_wave = min_wave+wave_chunk

#     # create result array of shape (no. of frames, no. of chunks, 2, no. of velocity pixels)
#     result = np.zeros((1, chunks_no, 2, len(velocities)))

#     for i in range(chunks_no):

#         # use indexing to select correct chunk of spectrum
#         idx = np.logical_and(wavelength>=min_wave, wavelength<=max_wave)

#         # run ACID function on specific chunk
#         result = acid.run_ACID([wavelength[idx]], [spectrum[idx]], [error[idx]], [sn], linelist,
#                            velocities, all_frames=result, order=i, nsteps=2000)

#         min_wave += wave_chunk
#         max_wave += wave_chunk

#     # reset min and max wavelengths
#     min_wave = min(wavelength)
#     max_wave = min_wave+wave_chunk

#     # plot results
#     plt.figure()
#     for i in range(chunks_no):

#         # extract profile and errors
#         profile = result[0, i, 0]
#         profile_error = result[0, i, 1]

#         plt.errorbar(velocities, profile, profile_error, label='(%s - %sÅ)'%(min_wave, max_wave))

#         min_wave += wave_chunk
#         max_wave += wave_chunk

#     plt.xlabel('Velocities (km/s)')
#     plt.ylabel('Flux')
#     plt.legend()
#     plt.show()


# multiple_frames()
# multiple_orders()