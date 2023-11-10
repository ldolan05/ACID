from astropy.io import fits
import ACID_code.ACID as acid
import numpy as np
import matplotlib.pyplot as plt
import glob

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

test_run_e2ds()
test_run_s1d()
