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
importlib.reload(acid)

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

res_e2ds = test_run_e2ds()
res_s1d = test_run_s1d()
print("All tests passed!")
