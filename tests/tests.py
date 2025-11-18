#%%
from astropy.io import fits
import os, glob, importlib, sys
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
acid._reload_all()

def test_run_e2ds():

    e2ds_files = glob.glob('tests/data/*e2ds_A*.fits')
    linelist = 'example/example_linelist.txt'
    save_path = 'no save'

    velocities = np.arange(-25, 25, 0.82)

    # run ACID on e2ds files
    ACID_results_e2ds = acid.run_ACID_HARPS(e2ds_files, linelist, velocities=velocities, save_path=save_path,
                                            order_range=np.arange(41, 43), nsteps=2000)
    return ACID_results_e2ds

def test_run_s1d():

    s1d_files = glob.glob('tests/data/*s1d_A*.fits')
    linelist = 'example/example_linelist.txt'
    save_path = 'no save'

    velocities = np.arange(-25, 25, 0.82)

    # run ACID on s1d files
    ACID_results_s1d = acid.run_ACID_HARPS(s1d_files, linelist, velocities=velocities, save_path=save_path,
                                       order_range = np.arange(41, 43), file_type = 's1d', nsteps=2000)
    return ACID_results_s1d

skips = 5

def quickstart():
    spec_file = fits.open('example/sample_spec_1.fits')

    wavelength = spec_file[0].data[::skips]   # Wavelengths in Angstroms
    spectrum = spec_file[1].data[::skips]     # Spectral Flux
    error = spec_file[2].data[::skips]        # Spectral Flux Errors
    sn = spec_file[3].data           # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list

    # choose a velocity grid for the final profile(s)
    deltav = acid.calc_deltav(wavelength)  # velocity pixel size must not be smaller than the spectral pixel size
    velocities = np.arange(-25, 25, deltav)

    # run ACID function
    result = acid.run_ACID(wavelength, spectrum, error, linelist, sn, velocities, nsteps=2000)

    # extract profile and errors
    profile = result[0, 0, 0]
    profile_error = result[0, 0, 1]

    # plot results
    plt.figure()
    plt.errorbar(velocities, profile, profile_error)
    plt.xlabel('Velocities (km/s)')
    plt.ylabel('Flux')
    plt.show()
    return result

def multiple_frames():

    # finds sample files in 'example directory'. Each file is a different frame.
    files = glob.glob('example/sample_spec_*.fits')

    # create lists for wavelengths, spectra, errors and sn for all frames
    wavelengths = []
    spectra = []
    errors = []
    sns = []

    for file in files:
        spec_file = fits.open('%s'%file)

        wavelengths.append(spec_file[0].data[::skips])    # Wavelengths in Angstroms
        spectra.append(spec_file[1].data[::skips])        # Spectral Flux
        errors.append(spec_file[2].data[::skips])         # Spectral Flux Errors
        sns.append(float(spec_file[3].data))     # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list

    # choose a velocity grid for the final profile(s)
    deltav = acid.calc_deltav(wavelengths[0])
    velocities = np.arange(-25, 25, deltav)

    # run ACID function
    result = acid.run_ACID(wavelengths, spectra, errors, linelist, sns, velocities, nsteps=2000)

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
    return result

def multiple_orders():
    spec_file = fits.open('example/sample_spec_1.fits')

    wavelength = spec_file[0].data[::skips]   # Wavelengths in Angstroms
    spectrum = spec_file[1].data[::skips]     # Spectral Flux
    error = spec_file[2].data[::skips]        # Spectral Flux Errors
    sn = spec_file[3].data           # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list

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

    for i in range(chunks_no):

        # use indexing to select correct chunk of spectrum
        idx = np.logical_and(wavelength>=min_wave, wavelength<=max_wave)

        # run ACID function on specific chunk
        result = acid.run_ACID([wavelength[idx]], [spectrum[idx]], [error[idx]], linelist,
                               [sn], velocities, all_frames=result, order=i, nsteps=2000)

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
    return result

def classes_test():
    spec_file = fits.open('example/sample_spec_1.fits')

    wavelength = spec_file[0].data[::skips]   # Wavelengths in Angstroms
    spectrum = spec_file[1].data[::skips]     # Spectral Flux
    error = spec_file[2].data[::skips]        # Spectral Flux Errors
    sn = spec_file[3].data           # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list

    # choose a velocity grid for the final profile(s)
    deltav = acid.calc_deltav(wavelength)  # velocity pixel size must not be smaller than the spectral pixel size
    velocities = np.arange(-25, 25, deltav)

    # run ACID function
    ACID = acid.ACID(velocities=velocities, linelist_path=linelist)
    result = ACID.run_ACID(wavelength, spectrum, error, sn, nsteps=2000)
    result.plot_corner()
    result.plot_profiles()
    result.plot_walkers()
    print(result[0,0,0][:5])
    return result

# q_res = quickstart()
# mf_res = multiple_frames()
# mo_res = multiple_orders()
# res_e2ds = test_run_e2ds()
# res_s1d = test_run_s1d()
classes_res = classes_test()
classes_res.continue_sampling(nsteps=3000)
classes_res.plot_walkers()

print("All tests passed!")

#%%
