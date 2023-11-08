import ACID_code.ACID as acid
import numpy as np
import glob

# import os

# files = glob.glob('/Users/lucydolan/Documents/GitHub/ACID/tests/data//*/*/*/**.fits') 
# print(files)

# for file in files:
#     os.system('mv "%s" "%s"'%(file, '/Users/lucydolan/Documents/GitHub/ACID/tests/data/'))

def test_run_e2ds():

    e2ds_files = glob.glob('/Users/lucydolan/Documents/HD189733/July2007/*/*/*/*e2ds*A*.fits')
    
    linelist = 'example/example_linelist.txt'
    save_path = 'no save'

    if len(e2ds_files)==0:
        e2ds_files = glob.glob('./*e2ds*.fits')
        s1d_files = glob.glob('./*s1d*.fits')
        linelist = '/home/lsd/Documents/Starbase/novaprime/Documents/fulllinelist0001.txt'
        save_path = './'

    velocities = np.arange(-25, 25, 0.82)

    # run ACID on e2ds files
    ACID_results_e2ds = acid.ACID_HARPS(e2ds_files, linelist, vgrid = velocities, save_path = save_path, order_range = np.arange(41, 43))


def test_run_s1d():

    s1d_files = glob.glob('/Users/lucydolan/Starbase/problem_frames/*s1d*.fits')
    linelist = 'example/example_linelist.txt'
    save_path = 'no save'

    velocities = np.arange(-25, 25, 0.82)

    # run ACID on s1d files
    ACID_results_s1d = acid.ACID_HARPS(s1d_files, linelist, vgrid = velocities, save_path = save_path, order_range = np.arange(41, 43), file_type = 's1d')

test_run_e2ds()
test_run_s1d()