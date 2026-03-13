#%%
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import glob, os, sys
from pathlib import Path
from time import time
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = next(p for p in SCRIPT_DIR.parents if (p / "pyproject.toml").exists())
sys.path.append(str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
from src import  ACID_code_v2 as acid
np.random.seed(0) # Set random seed for reproducibility in tests
start = time()
skips = 5

def legacy_test():

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
        result = acid.ACID(wavelength, spectrum, error, linelist, sn, velocities, nsteps=2000)

        # extract profile and errors
        profile = result[0, 0]
        profile_error = result[0, 1]

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
            sns.append(float(spec_file[3].data[0]))     # SN of Spectrum

        linelist = 'example/example_linelist.txt' # Insert path to line list

        # choose a velocity grid for the final profile(s)
        deltav = acid.calc_deltav(wavelengths[0])
        velocities = np.arange(-25, 25, deltav)

        # run ACID function
        result = acid.ACID(wavelengths, spectra, errors, linelist, sns, velocities, nsteps=2000)

        # plot results
        plt.figure()

        for frame in range(len(files)):
            profile = result[frame, 0]
            profile_error = result[frame, 1]
            plt.errorbar(velocities, profile, profile_error, label = '%s'%frame)

        plt.xlabel('Velocities (km/s)')
        plt.ylabel('Flux')
        plt.legend()
        plt.show()
        return result

    _ = quickstart() # generic quick test
    _ = multiple_frames()

def class_test():
    def classes_test(skips, nsteps):
        spec_file = fits.open('example/sample_spec_1.fits')

        wavelength = spec_file[0].data   # Wavelengths in Angstroms
        spectrum = spec_file[1].data     # Spectral Flux
        error = spec_file[2].data        # Spectral Flux Errors
        sn = spec_file[3].data           # SN of Spectrum

        linelist = 'example/example_linelist.txt' # Insert path to line list

        # choose a velocity grid for the final profile(s)
        velocities = np.arange(-25, 25, 0.82)

        # run ACID function
        Acid = acid.Acid(velocities=velocities, linelist_path=linelist)
        result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=nsteps, skips=skips)
        return result

    nsteps1 = 5000
    nsteps2 = 3000
    result = classes_test(skips=3, nsteps=nsteps1) # test the classes and result handling, use lower skips
    os.makedirs("tests/test_data", exist_ok=True) # create test data directory if it doesn't exist
    result.save_result(filename="tests/test_data/classes_test.pkl")
    result = acid.Result.load_result("tests/test_data/classes_test.pkl")
    result.plot_corner()
    result.plot_profiles()
    result.plot_walkers()
    result.plot_forward_model()
    result.continue_sampling(nsteps=nsteps2)
    assert result.sampler.get_chain().shape[0] == nsteps1 + nsteps2 and result.data.nsteps == nsteps1 + nsteps2, \
    f"Continue sampling did not add the correct number of steps to the chain.\n" \
    f"Expected {nsteps1 + nsteps2}, got {result.sampler.get_chain().shape[0]} for sampler and {result.data.nsteps} for data."  
    result.plot_walkers()
    result.plot_autocorrelation()
    result.plot_acf()
    acid.Profiles(velocities=np.arange(-25, 25, 0.82), flux=result[0,0]).plot_fit("all")
    result.plot_profiles()
    del result # clean up memory after test

def verbosity_test():
    def no_verbosity():
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
        Acid = acid.Acid(velocities=velocities, linelist_path=linelist, verbose=False)
        result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=2000)
        return result

    def high_verbosity():
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
        Acid = acid.Acid(velocities=velocities, linelist_path=linelist, verbose=3)
        result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=2000)
        return result
    
    print("Starting no verbosity, until no verbosity is printed, nothing should be output")
    res_nv = no_verbosity()
    res_nv.continue_sampling(nsteps=1000)
    del res_nv # clean up memory after test
    print("End no verbosity, starting high verbosity")
    res_hv = high_verbosity()
    del res_hv # clean up memory after test

def deterministic_test():
    def deterministic_profile_fit(skips=5, nsteps=2000):
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
            sns.append(float(spec_file[3].data[0]))     # SN of Spectrum

        linelist_path = 'example/example_linelist.txt' # Insert path to line list

        # choose a velocity grid for the final profile(s)
        velocities = np.arange(-25, 25, 0.82)

        # run ACID function
        Acid = acid.Acid(velocities=velocities, linelist_path=linelist_path)
        result = Acid.ACID(wavelengths, spectra, errors, sns, nsteps=nsteps, parallel=True, deterministic_profile=True, skips=skips)
        result.plot_walkers()
        return result
    
    res_deterministic_fit = deterministic_profile_fit(skips=3, nsteps=5000)
    res = res_deterministic_fit
    res.continue_sampling(nsteps=1000)
    res.plot_walkers()
    res.plot_profiles()
    res.plot_corner()
    res.plot_autocorrelation()
    res.plot_acf()

def data_and_convergence_test():
    spec_file = fits.open('example/sample_spec_1.fits')

    wavelength = spec_file[0].data   # Wavelengths in Angstroms
    spectrum = spec_file[1].data     # Spectral Flux
    error = spec_file[2].data        # Spectral Flux Errors
    sn = spec_file[3].data           # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list

    # choose a velocity grid for the final profile(s)
    velocities = np.arange(-25, 25, 0.82)

    # run ACID function
    Acid = acid.Acid(velocities=velocities, linelist_path=linelist)
    result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=2000, skips=skips)

    data = result.data

    Acid = acid.Acid(data=data) # data does not store the sampler
    result2 = Acid.ACID(nsteps=1000, parallel=True)

    assert result2.data.nsteps == 3000, "Continue sampling did not add the correct" \
    " number of steps to the chain when using data class."

    Acid.ACID(max_steps=5000) # test new convergence function

    # test convergence check interval
    Acid = acid.Acid(data=data)
    data.config.deterministic_profile = True
    Acid.ACID(max_steps=5000)

    return

def test_edge_cases():
    spec_file = fits.open('example/sample_spec_1.fits')

    wavelength = spec_file[0].data   # Wavelengths in Angstroms
    spectrum = spec_file[1].data     # Spectral Flux
    error = spec_file[2].data        # Spectral Flux Errors
    # sn = spec_file[3].data         # SN of Spectrum

    linelist = 'example/example_linelist.txt' # Insert path to line list
    full_linelist = np.genfromtxt('%s'%linelist, skip_header=4, delimiter=',', usecols=(1,9), invalid_raise=False)
    wl = full_linelist[:,0]
    depths = full_linelist[:,1]
    wl[199] = np.nan # introduce a nan value to test handling of nans in linelist
    depths[200] = np.nan # introduce a nan value to test handling of nans in linelist
    spectrum[201] = np.nan # introduce a nan value to test handling of nans in spectrum
    error[190] = np.nan # introduce a nan value to test handling of nans in error array

    # choose a velocity grid for the final profile(s)
    # velocities = np.arange(-25, 25, 0.82)

    # run ACID function
    Acid = acid.Acid(linelist_path=linelist)
    result = Acid.ACID(wavelength, spectrum, error, nsteps=2000, 
                       skips=skips, parallel=False, deterministic_profile=True)

    data = result.data
    # continue sampling
    Acid = acid.Acid(data=data)
    result = Acid.ACID(max_steps=5000) # test continue sampling with no parallelisation

    result.continue_sampling(max_steps=5000)
    result.plot_profiles()
    result.plot_walkers()
    result.plot_autocorrelation()

    # Plot continuum fit with the data class
    result.data.plot_continuum_fit("initial")
    result.data.plot_continuum_fit("masked")
    result.data.plot_residual_masking()

    # per pixel snr, and no run_mcmc test
    sn = np.random.normal(loc=100, scale=10, size=spectrum.shape) # create a random sn array with the same shape as the spectrum
    Acid = acid.Acid(linelist_path=linelist)
    result = Acid.ACID(wavelength, spectrum, error, sn, nsteps=2000, run_mcmc=False,
                       skips=skips, parallel=False, deterministic_profile=True)
    assert result is None, "When run_mcmc is set to False, the ACID function should return None, but it did not."

    # Guess sn for multiple frames, plot multiple frames
    files = glob.glob('example/sample_spec_*.fits')
    wavelengths = []
    spectra = []
    errors = []
    sns = []
    for file in files:
        spec_file = fits.open('%s'%file)
        wavelengths.append(spec_file[0].data[::skips])    # Wavelengths in Angstroms
        spectra.append(spec_file[1].data[::skips])        # Spectral Flux
        errors.append(spec_file[2].data[::skips])         # Spectral Flux Errors
        sns.append(float(spec_file[3].data[0]))     # SN of Spectrum
    deltav = acid.calc_deltav(wavelengths[0])
    velocities = np.arange(-25, 25, deltav)
    result = acid.ACID(wavelengths, spectra, errors, linelist, sns, velocities, nsteps=2000)
    result.plot_profiles()
    pass

print("Starting tests, this will take a 4-6 minutes to run, and a bunch of output will be printed.")

# The first five tests use legacy ACID inputs and calls
legacy_test()

# Now test classes
class_test()

# Test verbosities
verbosity_test()

# Test deterministic profile fit
deterministic_test()

# Add a skipping calculations using the data class test
data_and_convergence_test()

# Test edge cases, including no parallelization
test_edge_cases()


print("All tests passed!")
print(f"Total time: {time() - start:.2f} seconds")

# %%
