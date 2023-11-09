from astropy.io import fits
import ACID_code.ACID as acid
import numpy as np
import matplotlib.pyplot as plt
import glob

def quickstart():
    spec_file = fits.open('example/sample_spec_1.fits')

    wavelength = spec_file[0].data   # Wavelengths in Angstroms
    spectrum = spec_file[1].data     # Spectral Flux
    error = spec_file[2].data        # Spectral Flux Errors
    sn = spec_file[3].data           # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list

    # choose a velocity grid for the final profile(s)
    deltav = acid.calc_deltav(wavelength)  # velocity pixel size must not be smaller than the spectral pixel size
    velocities = np.arange(-25, 25, deltav)

    # run ACID function
    result = acid.ACID([wavelength], [spectrum], [error], linelist, [sn], velocities)

    # extract profile and errors
    profile = result[0, 0, 0]
    profile_error = result[0, 0, 1]

    # plot results
    plt.figure()
    plt.errorbar(velocities, profile, profile_error)
    plt.xlabel('Velocities (km/s)')
    plt.ylabel('Flux')
    plt.close('all')

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

        wavelengths.append(spec_file[0].data)    # Wavelengths in Angstroms
        spectra.append(spec_file[1].data)        # Spectral Flux
        errors.append(spec_file[2].data)         # Spectral Flux Errors
        sns.append(float(spec_file[3].data))     # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list

    # choose a velocity grid for the final profile(s)
    deltav = acid.calc_deltav(wavelengths[0])
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
    plt.close('all')

def mulitple_orders():
    spec_file = fits.open('example/sample_spec_1.fits')

    wavelength = spec_file[0].data   # Wavelengths in Angstroms
    spectrum = spec_file[1].data     # Spectral Flux
    error = spec_file[2].data        # Spectral Flux Errors
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
    plt.close('all')

def test_run_e2ds():

    e2ds_files = glob.glob('tests/data/*e2ds_A*.fits')
    linelist = 'example/example_linelist.txt'
    save_path = 'no save'


    velocities = np.arange(-25, 25, 0.82)

    # run ACID on e2ds files
    ACID_results_e2ds = acid.ACID_HARPS(e2ds_files, linelist, vgrid = velocities, save_path = save_path, order_range = np.arange(41, 43))


def test_run_s1d():

    s1d_files = glob.glob('tests/data/*s1d_A*.fits')
    linelist = 'example/example_linelist.txt'
    save_path = 'no save'

    velocities = np.arange(-25, 25, 0.82)

    # run ACID on s1d files
    ACID_results_s1d = acid.ACID_HARPS(s1d_files, linelist, vgrid = velocities, save_path = save_path, order_range = np.arange(41, 43), file_type = 's1d')

quickstart()
multiple_frames()
mulitple_orders()
test_run_e2ds()
test_run_s1d()
